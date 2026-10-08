/*
 * libembedding - detail/gguf/gguf_sparse_session.hpp
 * Turns text into a sparse vector using a SPLADE GGUF model and ggml.
 *
 * The pipeline, end to end:
 *
 *   text
 *     -> WordPiece ids            (TokenizerWrapper, shared with the ONNX backend)
 *     -> ggml graph: BERT encoder + MLM head
 *     -> log(1 + relu(logits)) max-pooled over the document's real tokens
 *     -> special tokens removed
 *     -> top_k / min_weight / storage_format   (caller, via sparse_prune_and_sort)
 *
 * Two properties are worth stating because they are easy to break:
 *
 *   1. The head is a `ggml_mul_mat` against the tied token embedding matrix, not
 *      a scalar loop. A loop is O(tokens x hidden x vocab) and this model's
 *      vocabulary is 30522 wide; the op also does the Q8_0 dequantisation, which
 *      is why nothing here ever reads a quantised weight element by element.
 *
 *   2. Every document of a batch is one flat row of tokens and is separated from
 *      its neighbours by the attention mask, not by a second graph. A padded
 *      batch and a ragged one therefore produce the same numbers, which is what
 *      makes batching safe rather than merely fast.
 *
 * Not thread-safe: embed() writes into one scratch buffer for the logits, so two
 * threads sharing a session would overwrite each other's results. The dense
 * llama.cpp pool exists for that reason and there is no sparse equivalent; one
 * session per thread, or serialise the calls.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_SPARSE_SESSION_HPP
#define LIBEMBEDDING_GGUF_SPARSE_SESSION_HPP

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../sparse_postprocess.hpp"
#include "../tokenizer_impl.hpp"
#include "gguf_bert_graph.hpp"

/* Order matters. gguf_spec.hpp includes gguf_inspect.h, and gguf_inspect.h's
 * implementation block includes gguf_inspect_entry.hpp, which calls
 * lembed::gguf::classify(). Entering through gguf_inspect.h first makes
 * gguf_spec.hpp finish parsing before the entry points are compiled; entering
 * through gguf_spec.hpp does the opposite and the entry points fail to see
 * classify(). */
#include "../../gguf_inspect.h"
#include "gguf_spec.hpp"
#include "gguf_weights.hpp"

namespace lembed {
namespace gguf {

/* lembed::detail is a sibling of lembed::gguf, not a parent, so the shared
 * tokenisation and pooling types are pulled in explicitly rather than by a
 * namespace directive. Naming them is what keeps it obvious that this runtime
 * reuses them instead of reimplementing them. */
using detail::EncodingBatch;
using detail::SparseResult;
using detail::TokenizerWrapper;
using detail::splade_pool;

/* Reads a SPLADE GGUF and embeds text with it.
 *
 * Owns the weights, the ggml CPU backend and the tokenizer; nothing is shared
 * with the ONNX backend except the tokenizer's algorithm and the SPLADE pooling
 * helper, both deliberately so that the two cannot drift apart. */
class SparseSession {
public:
    SparseSession() = default;

    SparseSession(const SparseSession&) = delete;
    SparseSession& operator=(const SparseSession&) = delete;
    SparseSession(SparseSession&&) = delete;
    SparseSession& operator=(SparseSession&&) = delete;

    ~SparseSession() {
        if (graph_.galloc) {
            ggml_gallocr_free(graph_.galloc);
            graph_.galloc = nullptr;
        }
        if (graph_.ctx) {
            ggml_free(graph_.ctx);
            graph_.ctx = nullptr;
        }
    }

    /* Loads a SPLADE GGUF.
     *
     * The capability check is the same one lembed_gguf_inspect() performs, and it
     * runs before anything is allocated: a dense file, a file with no vocabulary
     * or a file with no MLM head is refused here with the reason, rather than
     * accepted and turned into a plausible vector over the wrong tokens.
     *
     * `max_length` of 0 means "as long as the position table allows". A longer
     * sequence would index past the position embeddings, so the request is capped
     * at the model's own context length rather than refused.
     *
     * Returns false with the reason in `err`. */
    bool load_from_file(const char* path, int n_threads, int max_length,
                        std::string& err) {
        std::unique_ptr<Weights> weights(new Weights());
        if (!weights->open(path, err)) return false;

        /* Before anything is allocated: telling somebody their dense file has no
         * sparse head is cheap, and doing it after a hundred megabytes of weights
         * have been copied is not. */
        if (!accept(weights->probe(), err)) return false;

        if (!weights->load_tensors(n_threads, err)) return false;

        const SpladeHparams& hp = weights->hparams();

        std::vector<std::string> tokens;
        if (!weights->probe().get_arr_str("tokenizer.ggml.tokens", tokens)) {
            err = "GGUF has no tokenizer.ggml.tokens vocabulary array";
            return false;
        }
        {
            std::string model;
            if (weights->probe().get_str("tokenizer.ggml.model", model) &&
                model != "bert" && model != "bert-wordpiece") {
                err = "GGUF declares tokenizer model '" + model +
                      "', which this runtime does not implement (expected bert)";
                return false;
            }
        }
        if ((int)tokens.size() != hp.n_vocab) {
            char msg[192];
            snprintf(msg, sizeof(msg),
                     "GGUF vocabulary holds %d entries but %s.vocab_size is %d",
                     (int)tokens.size(), hp.arch.c_str(), hp.n_vocab);
            err = msg;
            return false;
        }

        max_length_ = (max_length > 0) ? std::min(max_length, hp.n_pos) : hp.n_pos;

        int pad_id = -1, unk_id = -1, cls_id = -1, sep_id = -1;
        read_token_id(weights->probe(), "tokenizer.ggml.pad_token_id", pad_id);
        read_token_id(weights->probe(), "tokenizer.ggml.unk_token_id", unk_id);
        read_token_id(weights->probe(), "tokenizer.ggml.cls_token_id", cls_id);
        read_token_id(weights->probe(), "tokenizer.ggml.separator_token_id", sep_id);
        if (sep_id < 0) {
            read_token_id(weights->probe(), "tokenizer.ggml.eos_token_id", sep_id);
        }

        /* [CLS] and [SEP] go on as a pair or not at all: encode_single() emits
         * both, and half a delimiter pair is a different sequence, not an
         * approximation of the right one. */
        bool add_bos = true, add_eos = true;
        weights->probe().get_bool("tokenizer.ggml.adding_bos_token", add_bos);
        weights->probe().get_bool("tokenizer.ggml.adding_eos_token", add_eos);
        const bool add_special = add_bos && add_eos;

        if (!tok_.load_vocab(tokens, pad_id, unk_id, add_special ? cls_id : -1,
                             add_special ? sep_id : -1, max_length_, add_special,
                             err)) {
            return false;
        }
        if (tok_.pad_token_id() < 0 || tok_.pad_token_id() >= hp.n_vocab) {
            err = "GGUF pad token id is outside the vocabulary";
            return false;
        }

        vocab_size_ = hp.n_vocab;
        special_ids_ = weights->special_ids();
        path_ = path ? path : "";

        /* The vocabulary is the last thing read out of the header, so the file's
         * blob -- as large as the file itself -- can go now. What stays is the
         * weight buffer. */
        weights->release_header();
        weights_ = std::move(weights);
        return true;
    }

    /* Embeds a batch of texts, one SparseResult per input, in order.
     *
     * The results are SPLADE-pooled and stripped of special tokens but *not*
     * pruned: top_k / min_weight / storage_format belong to the caller and are
     * applied by sparse_prune_and_sort(), which is also what the ONNX backend
     * uses. */
    std::vector<SparseResult> embed(const std::vector<std::string>& texts,
                                    int batch_size) {
        std::vector<SparseResult> out;
        out.reserve(texts.size());
        if (texts.empty() || !weights_) return out;

        if (batch_size <= 0) batch_size = 1;
        for (size_t start = 0; start < texts.size(); start += (size_t)batch_size) {
            const size_t end = std::min(texts.size(), start + (size_t)batch_size);
            std::vector<std::string> batch;
            batch.reserve(end - start);
            for (size_t i = start; i < end; i++) batch.push_back(texts[i]);
            for (SparseResult& r : embed_batch(batch)) out.push_back(std::move(r));
        }
        return out;
    }

    int vocab_size() const { return vocab_size_; }
    int max_length() const { return max_length_; }
    const std::string& path() const { return path_; }

private:
    static void read_token_id(const Probe& probe, const char* key, int& out) {
        int32_t v = 0;
        if (probe.get_i32(key, v) && v >= 0) out = (int)v;
    }

    /* Refuses anything that is not a SPLADE head over a known vocabulary.
     *
     * This is deliberately the P1.5b classification and not a second one: the
     * runtime must not be able to accept a file the inspector would call dense. */
    bool accept(const Probe& probe, std::string& err) {
        lembed_gguf_desc_t d;
        classify(probe, d);

        if (!probe.valid()) {
            err = "not a readable GGUF file";
            return false;
        }
        if (d.diagnostic[0] != '\0') {
            err = d.diagnostic;
            return false;
        }
        if (!(d.capabilities & LEMBED_GGUF_CAP_SPLADE)) {
            err = "GGUF carries no SPLADE head (mlm_transform.weight is absent)";
            return false;
        }
        if (d.sparse_formula != LEMBED_GGUF_SPARSE_FORMULA_SPLADE) {
            err = "GGUF sparse formula is not SPLADE";
            return false;
        }
        if (!d.has_vocab_size) {
            err = "GGUF has no <architecture>.vocab_size to pool over";
            return false;
        }
        if (d.missing_hparams[0] != '\0') {
            err = std::string("GGUF is missing ") + d.missing_hparams;
            return false;
        }
        if (d.capabilities & LEMBED_GGUF_CAP_SPARSE_LINEAR) {
            err = "GGUF carries both a SPLADE head and a sparse_linear head; "
                  "refusing to guess which one this file is meant for";
            return false;
        }
        return true;
    }

    std::vector<SparseResult> embed_batch(const std::vector<std::string>& texts) {
        const int batch = (int)texts.size();
        std::vector<SparseResult> results((size_t)batch);

        EncodingBatch enc = tok_.encode_batch(texts);
        const int seq_len = enc.seq_length;
        if (seq_len <= 0) return results;

        /* One flat row of tokens: [document][position] concatenated, document
         * major. The batch is an axis of its own in the graph, so the mask is
         * [seq_len, seq_len, 1, batch] and is what keeps the documents apart --
         * nothing else has to know where one ends and the next begins, except
         * the pooling, which does, by walking each document's own slice. */
        const int n_tokens = batch * seq_len;
        std::vector<int32_t> tokens((size_t)n_tokens);
        std::vector<int32_t> positions((size_t)n_tokens);
        std::vector<int64_t> valid((size_t)n_tokens, 0);

        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < seq_len; j++) {
                const size_t t = (size_t)(b * seq_len + j);
                positions[t] = (int32_t)j;
                tokens[t] = (int32_t)enc.input_ids[(size_t)b][j];
                valid[t] = enc.attention_mask[(size_t)b][j];
            }
        }

        const float* logits = run_graph(tokens, positions, valid, seq_len, batch);
        if (!logits) return results;

        for (int b = 0; b < batch; b++) {
            const int begin = b * seq_len;
            splade_pool(logits + (size_t)begin * (size_t)vocab_size_, vocab_size_,
                        valid.data() + begin, seq_len, results[(size_t)b]);
            drop_special_tokens(results[(size_t)b]);
        }
        return results;
    }

    /* Removes the vocabulary entries that must never carry weight: the padding
     * and the sequence delimiters, as the file declares them. On a BERT
     * vocabulary that is [PAD], [CLS] and [SEP]; on another one it is whatever
     * the metadata says, which is why this list comes from the file and not from
     * a constant. */
    void drop_special_tokens(SparseResult& r) const {
        if (special_ids_.empty()) return;
        size_t keep = 0;
        for (size_t i = 0; i < r.indices.size(); i++) {
            const int32_t id = r.indices[i];
            if (std::find(special_ids_.begin(), special_ids_.end(), id) !=
                special_ids_.end()) {
                continue;
            }
            r.indices[keep] = id;
            r.values[keep] = r.values[i];
            keep++;
        }
        r.indices.resize(keep);
        r.values.resize(keep);
    }

/* Builds the graph, allocates it, feeds it and returns a pointer to the
     * logits in the session's scratch buffer, valid until the next call. Null on
     * a compute failure.
     *
     * The graph context is cached and reused across calls. It is rebuilt only when
     * the shape (seq_len, batch) changes. */
    const float* run_graph(const std::vector<int32_t>& tokens,
                           const std::vector<int32_t>& positions,
                           const std::vector<int64_t>& valid,
                           int seq_len, int batch) {
        bool rebuild = (graph_.ctx == nullptr) ||
                       (graph_.cached_seq_len != seq_len) ||
                       (graph_.cached_batch != batch);

        if (rebuild) {
            /* Free old graph if exists */
            if (graph_.galloc) {
                ggml_gallocr_free(graph_.galloc);
                graph_.galloc = nullptr;
            }
            if (graph_.ctx) {
                ggml_free(graph_.ctx);
                graph_.ctx = nullptr;
            }

            ggml_init_params iparams{};
            iparams.mem_size = ggml_graph_overhead() + ggml_tensor_overhead() * 2048;
            iparams.mem_buffer = nullptr;
            iparams.no_alloc = true;

            graph_.ctx = ggml_init(iparams);
            if (!graph_.ctx) return nullptr;

            graph_.t_tokens = ggml_new_tensor_1d(graph_.ctx, GGML_TYPE_I32,
                                                 seq_len * batch);
            graph_.t_pos = ggml_new_tensor_1d(graph_.ctx, GGML_TYPE_I32,
                                              seq_len * batch);
            graph_.t_mask = ggml_new_tensor_4d(graph_.ctx, GGML_TYPE_F32, seq_len,
                                               seq_len, 1, batch);
            graph_.gf = ggml_new_graph(graph_.ctx);
            if (!graph_.t_tokens || !graph_.t_pos || !graph_.t_mask || !graph_.gf) {
                ggml_free(graph_.ctx);
                graph_.ctx = nullptr;
                return nullptr;
            }

            graph_.logits = build_mlm_head(graph_.ctx, build_bert_encoder(
                                                 graph_.ctx, *weights_, graph_.t_tokens, graph_.t_pos,
                                                 graph_.t_mask, seq_len, batch));
            if (!graph_.logits) {
                ggml_free(graph_.ctx);
                graph_.ctx = nullptr;
                return nullptr;
            }
            ggml_build_forward_expand(graph_.gf, graph_.logits);

            graph_.galloc = ggml_gallocr_new(weights_->buffer_type());
            if (!graph_.galloc || !ggml_gallocr_alloc_graph(graph_.galloc, graph_.gf)) {
                if (graph_.galloc) ggml_gallocr_free(graph_.galloc);
                ggml_free(graph_.ctx);
                graph_.ctx = nullptr;
                return nullptr;
            }

            graph_.cached_seq_len = seq_len;
            graph_.cached_batch = batch;
        }

        /* After allocation: until then the input tensors have no data to write
         * into. The mask is written from mask.size(), not from n_tokens * n_tokens:
         * those two are the same only when the batch is one, and writing the
         * larger count into the smaller tensor is a heap overflow. */
        const std::vector<float> mask = build_mask(valid, seq_len, batch);

        ggml_backend_tensor_set(graph_.t_tokens, tokens.data(), 0,
                                (size_t)seq_len * (size_t)batch * sizeof(int32_t));
        ggml_backend_tensor_set(graph_.t_pos, positions.data(), 0,
                                (size_t)seq_len * (size_t)batch * sizeof(int32_t));
        ggml_backend_tensor_set(graph_.t_mask, mask.data(), 0,
                                (size_t)mask.size() * sizeof(float));

        bool ok = ggml_backend_graph_compute(weights_->backend(), graph_.gf) ==
                  GGML_STATUS_SUCCESS;

        if (ok) {
            const size_t n_values =
                (size_t)vocab_size_ * (size_t)seq_len * (size_t)batch;
            scratch_.resize(n_values);
            ggml_backend_tensor_get(graph_.logits, scratch_.data(), 0,
                                    n_values * sizeof(float));
        }

        return ok ? scratch_.data() : nullptr;
    }

    /* The MLM head, in the graph.
     *
     * The decoder projection is the tied token embedding matrix, so the file
     * stores no separate [vocab, hidden] tensor and none is invented here.
     * `mlm_bias` is the only vocabulary-sized vector. */
    ggml_tensor* build_mlm_head(ggml_context* ctx, ggml_tensor* h) {
        const SpladeHparams& hp = weights_->hparams();
        const HeadWeights& head = weights_->head();

        ggml_tensor* x = ggml_add(ctx, ggml_mul_mat(ctx, head.mlm_transform_w, h),
                                  head.mlm_transform_b);
        x = ggml_gelu(ctx, x);
        x = bert_layernorm(ctx, x, head.mlm_ln_w, head.mlm_ln_b, hp.eps);
        return ggml_add(ctx, ggml_mul_mat(ctx, head.tok_embd, x), head.mlm_bias);
    }

    /* Per-document attention mask, [seq_len, seq_len, 1, batch], added to the
     * scores before the softmax.
     *
     * A query at (document b, position i) may attend to a key at (document b,
     * position j) when that key is not padding. Everything else is -infinity.
     *
     * The head axis is left at 1: a BERT mask has no per-head component, so
     * every head reads the same plane, and ggml's softmax indexes it with
     * i02 % ne12 (ggml-cpu/ops.cpp:5499-5500), so a single plane is read for
     * every head and never out of bounds.
     *
     * A query row that is padding attends to its own document's real tokens
     * instead of to itself, so the softmax is well defined and the row's column
     * is discarded at pooling. The "mask a padding row against itself only"
     * special case is gone: it existed to avoid a divide by zero, and it is
     * unnecessary now that every document carries at least one real token.
     *
     * The one case that cannot happen is a document with no real token at all:
     * its rows would be all -infinity and the softmax a NaN. [CLS] and [SEP]
     * guarantee that never happens, but the guard is two lines and a silent NaN
     * is worse than a dead branch. */
    std::vector<float> build_mask(const std::vector<int64_t>& valid,
                                  int seq_len, int batch) const {
        const float neg = -std::numeric_limits<float>::infinity();
        std::vector<float> mask((size_t)seq_len * (size_t)seq_len * (size_t)batch);
        for (int b = 0; b < batch; b++) {
            const int64_t* vb = valid.data() + (size_t)b * (size_t)seq_len;
            float* plane_b = mask.data() + (size_t)b * (size_t)seq_len *
                                                (size_t)seq_len;
            bool any_valid = false;
            for (int i = 0; i < seq_len; i++) {
                float* row = plane_b + (size_t)i * (size_t)seq_len;
                for (int j = 0; j < seq_len; j++) {
                    if (vb[j] != 0) {
                        row[j] = 0.0f;
                        any_valid = true;
                    } else {
                        row[j] = neg;
                    }
                }
            }
            if (!any_valid) {
                for (int i = 0; i < seq_len; i++)
                    plane_b[(size_t)i * (size_t)seq_len + i] = 0.0f;
            }
        }
        return mask;
    }

    std::unique_ptr<Weights> weights_;
    TokenizerWrapper tok_;
    std::string path_;
    std::vector<int32_t> special_ids_;
    std::vector<float> scratch_;
    int vocab_size_ = 0;
    int max_length_ = 0;

    /* Cached graph context for reuse across embed_batch() calls.
     * The graph structure depends only on (seq_len, batch, vocab_size, n_embd, n_head, n_layers),
     * so it can be built once and reused. Only input tensor data changes per batch. */
    struct CachedGraph {
        ggml_context* ctx = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_tensor* t_tokens = nullptr;
        ggml_tensor* t_pos = nullptr;
        ggml_tensor* t_mask = nullptr;
        ggml_tensor* logits = nullptr;
        int cached_seq_len = 0;
        int cached_batch = 0;
    } graph_;
};

} /* namespace gguf */
} /* namespace lembed */

#endif /* LIBEMBEDDING_GGUF_SPARSE_SESSION_HPP */