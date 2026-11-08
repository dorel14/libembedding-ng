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
 * Parallelism comes from workers, not from more threads per graph. One graph
 * stops scaling past a few threads -- ggml synchronises every thread after each
 * of its ~460 operations, and a BERT graph over a few hundred tokens is many
 * short operations -- whereas independent graphs do not wait for each other.
 * So a session runs N workers, each with its own backend, graph and scratch,
 * all reading one shared, read-only copy of the weights; the batches of a call
 * are handed out to them. Measured on 4 cores / 8 threads: 4 workers x 2
 * threads beat 1 x 4 by 19-31 % and 1 x 8 by 22-23 %.
 *
 * embed() itself is not re-entrant: two threads must not call it on the same
 * session at the same time (the workers' graphs and the tokenizer's scratch are
 * per session). One session per calling thread, or serialise the calls.
 *
 * Auteur: David Orel
 * Version: 1.12.1
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_SPARSE_SESSION_HPP
#define LIBEMBEDDING_GGUF_SPARSE_SESSION_HPP

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifdef LIBEMBEDDING_GGUF_PROFILE
#include <chrono>
#include <map>
#endif

#include "../ggml_cpu_backend.hpp"
#include "../sparse_batching.hpp"
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
using detail::sparse_batch_end;
using detail::sparse_batch_order;
using detail::splade_pool;

#ifdef LIBEMBEDDING_GGUF_PROFILE
struct SparseProfile {
    double tokenization_ms = 0.0;
    double input_pack_ms = 0.0;
    double graph_build_ms = 0.0;
    double mask_build_ms = 0.0;
    double input_upload_ms = 0.0;
    double graph_exec_ms = 0.0;
    double logits_download_ms = 0.0;
    double pooling_ms = 0.0;
    uint64_t batches = 0;
    uint64_t graph_builds = 0;
    uint64_t graph_cache_hits = 0;
    uint64_t real_tokens = 0;
    uint64_t padded_tokens = 0;
    uint64_t graph_nodes = 0;
    int max_seq_len = 0;
    int graph_seq_len = 0;
    int graph_batch = 0;

    void reset() { *this = SparseProfile{}; }

    /* Folds a worker's profile into this one. Stage times are summed, so with
     * several workers they are CPU-side time across workers, not wall time. */
    void merge(const SparseProfile& o) {
        tokenization_ms += o.tokenization_ms;
        input_pack_ms += o.input_pack_ms;
        graph_build_ms += o.graph_build_ms;
        mask_build_ms += o.mask_build_ms;
        input_upload_ms += o.input_upload_ms;
        graph_exec_ms += o.graph_exec_ms;
        logits_download_ms += o.logits_download_ms;
        pooling_ms += o.pooling_ms;
        batches += o.batches;
        graph_builds += o.graph_builds;
        graph_cache_hits += o.graph_cache_hits;
        real_tokens += o.real_tokens;
        padded_tokens += o.padded_tokens;
        max_seq_len = std::max(max_seq_len, o.max_seq_len);
        if (o.graph_nodes) {
            graph_nodes = o.graph_nodes;
            graph_seq_len = o.graph_seq_len;
            graph_batch = o.graph_batch;
        }
    }
};
#endif

/* Reads a SPLADE GGUF and embeds text with it.
 *
 * Owns the weights, the ggml CPU backend and the tokenizer; nothing is shared
 * with the ONNX backend except the tokenizer's algorithm and the SPLADE pooling
 * helper, both deliberately so that the two cannot drift apart. */
class SparseSession {
public:
    /* Default cap on batch * seq_len; see set_max_batch_tokens(). Shared with
     * the ONNX path: on ggml too, 512 gave the best throughput of 128..2048 on
     * short and long documents and kept peak RSS under ~420 MiB, where an
     * uncapped batch reached 1.6-2.2 GiB (docs/gguf_sparse_runtime.md). */
    static constexpr int kDefaultMaxBatchTokens = detail::kSparseMaxBatchTokens;

    SparseSession() = default;

    SparseSession(const SparseSession&) = delete;
    SparseSession& operator=(const SparseSession&) = delete;
    SparseSession(SparseSession&&) = delete;
    SparseSession& operator=(SparseSession&&) = delete;

    ~SparseSession() = default;

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
     * `n_threads` is the total thread budget and `workers` the number of
     * independent graphs sharing it; 0 picks each automatically (see
     * resolve_parallelism()).
     *
     * Returns false with the reason in `err`. */
    bool load_from_file(const char* path, int n_threads, int workers,
                        int max_length, std::string& err) {
        std::unique_ptr<Weights> weights(new Weights());
        if (!weights->open(path, err)) return false;

        /* Before anything is allocated: telling somebody their dense file has no
         * sparse head is cheap, and doing it after a hundred megabytes of weights
         * have been copied is not. */
        if (!accept(weights->probe(), err)) return false;

        if (!weights->load_tensors(err)) return false;

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

        resolve_parallelism(n_threads, workers, total_threads_);
        std::vector<std::unique_ptr<Worker>> pool;
        for (int i = 0; i < workers; i++) {
            std::unique_ptr<Worker> w(new Worker());
            w->backend = lembed::detail::ggml_cpu_backend_init();
            if (!w->backend) {
                err = lembed::detail::ggml_cpu_device()
                          ? "cannot initialise the ggml CPU backend"
                          : lembed::detail::ggml_cpu_backend_missing_message();
                return false;
            }
            pool.push_back(std::move(w));
        }

        weights_ = std::move(weights);
        workers_ = std::move(pool);
        return true;
    }

    /* The thread budget and how it is split.
     *
     * total: `n_threads`, or every logical CPU when 0.
     * workers: `workers`, or total / 2 when 0 -- two threads per worker, which on
     * a hyper-threaded CPU is one worker per physical core, the configuration
     * that measured fastest (4 x 2 on 4 cores / 8 threads). Never more workers
     * than threads. embed() shares the threads among the workers it uses. */
    static void resolve_parallelism(int n_threads, int& workers, int& total) {
        const int logical = (int)std::thread::hardware_concurrency();
        total = (n_threads > 0) ? n_threads : std::max(1, logical);
        if (workers <= 0) workers = std::max(1, total / 2);
        workers = std::min(workers, total);
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
        if (texts.empty() || !weights_) return out;
        out.resize(texts.size());

        if (batch_size <= 0) batch_size = 1;
        std::vector<std::vector<int>> token_ids(texts.size());
#ifdef LIBEMBEDDING_GGUF_PROFILE
        auto tokenization_start = std::chrono::steady_clock::now();
#endif
        for (size_t i = 0; i < texts.size(); i++) {
            token_ids[i] = tok_.encode(texts[i]);
        }
#ifdef LIBEMBEDDING_GGUF_PROFILE
        profile_.tokenization_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tokenization_start).count();
#endif

        /* Every batch is cut up front, on this thread, so the workers only ever
         * read token ids: TokenizerWrapper::encode() is not thread-safe, its
         * padding step is. */
        const std::vector<size_t> order = sparse_batch_order(token_ids, batch_size);
        std::vector<std::pair<size_t, size_t>> batches;
        for (size_t start = 0; start < order.size();) {
            const size_t end = sparse_batch_end(
                start, order.size(), token_ids[order[start]].size(), batch_size,
                max_batch_tokens_);
            batches.emplace_back(start, end);
            start = end;
        }
        last_batch_count_ = (int)batches.size();

        /* A call that is one batch -- a query, typically -- runs on one worker
         * with the whole thread budget: splitting its threads would only make
         * it slower. Otherwise as many workers as there are batches, each with
         * an equal share of the threads. */
        const size_t active = std::min(workers_.size(), batches.size());
        const int threads = (active <= 1) ? total_threads_
                                          : std::max(1, total_threads_ / (int)active);
        for (size_t i = 0; i < active; i++) {
            lembed::detail::ggml_cpu_backend_set_n_threads(workers_[i]->backend, threads);
        }

        std::atomic<size_t> next{0};
        std::exception_ptr failure;
        std::mutex failure_mutex;
        auto drain = [&](Worker& w) {
            try {
                for (size_t b = next++; b < batches.size(); b = next++) {
                    const size_t start = batches[b].first, end = batches[b].second;
                    std::vector<std::vector<int>> batch;
                    batch.reserve(end - start);
                    for (size_t i = start; i < end; i++) {
                        batch.push_back(std::move(token_ids[order[i]]));
                    }
                    std::vector<SparseResult> results = embed_batch(w, batch);
                    for (size_t i = start; i < end; i++) {
                        out[order[i]] = std::move(results[i - start]);
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                next = batches.size(); /* stop handing out work */
            }
        };

        std::vector<std::thread> helpers;
        helpers.reserve(active > 0 ? active - 1 : 0);
        for (size_t i = 1; i < active; i++) {
            helpers.emplace_back(drain, std::ref(*workers_[i]));
        }
        if (active > 0) drain(*workers_[0]);
        for (std::thread& t : helpers) t.join();
        if (failure) std::rethrow_exception(failure);
        return out;
    }

    /* Workers and threads in use after load_from_file(). */
    int workers() const { return (int)workers_.size(); }
    int total_threads() const { return total_threads_; }

    /* Caps batch * seq_len, the number of token columns one graph computes.
     * 0 removes the cap and leaves batch_size as the only bound. */
    void set_max_batch_tokens(int n) { max_batch_tokens_ = std::max(0, n); }
    int max_batch_tokens() const { return max_batch_tokens_; }

    /* Graphs the last embed() ran: the token budget can cut more batches than
     * ceil(texts / batch_size), so callers counting batches ask here. */
    int last_batch_count() const { return last_batch_count_; }

    int vocab_size() const { return vocab_size_; }
    int max_length() const { return max_length_; }
    const std::string& path() const { return path_; }

#ifdef LIBEMBEDDING_GGUF_PROFILE
    /* The session's own stages (tokenisation) plus every worker's. */
    SparseProfile profile() const {
        SparseProfile merged = profile_;
        for (const auto& w : workers_) merged.merge(w->profile);
        return merged;
    }
    void reset_profile() {
        profile_.reset();
        for (auto& w : workers_) w->profile.reset();
    }

    void print_graph_operations(FILE* out) const {
        if (!out || workers_.empty() || !workers_[0]->graph.gf) return;
        ggml_cgraph* gf = workers_[0]->graph.gf;
        std::map<std::string, uint64_t> counts;
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor* node = ggml_graph_node(gf, i);
            counts[ggml_op_name(node->op)]++;
        }
        std::fprintf(out, "Graph nodes: %d\n", n_nodes);
        std::fprintf(out, "%-28s %10s\n", "GGML operation", "Nodes");
        for (const auto& entry : counts) {
            std::fprintf(out, "%-28s %10llu\n", entry.first.c_str(),
                         (unsigned long long)entry.second);
        }
    }
#endif

private:
    /* Cached graph context, reused across batches of the same shape.
     * The graph structure depends only on (seq_len, batch); vocab_size, n_embd,
     * n_head and n_layers are fixed per session after load_from_file(). */
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

        CachedGraph() = default;
        CachedGraph(const CachedGraph&) = delete;
        CachedGraph& operator=(const CachedGraph&) = delete;
        ~CachedGraph() { reset(); }

        /* Frees the allocator before the context (the allocator references
         * tensors owned by the context), then nulls every pointer and resets
         * the cached dimensions. Safe to call repeatedly. */
        void reset() {
            if (galloc) ggml_gallocr_free(galloc);
            if (ctx) ggml_free(ctx);
            galloc = nullptr;
            ctx = nullptr;
            gf = nullptr;
            t_tokens = nullptr;
            t_pos = nullptr;
            t_mask = nullptr;
            logits = nullptr;
            cached_seq_len = 0;
            cached_batch = 0;
        }
    };

    /* Everything one batch writes to: a backend with its own thread count, a
     * graph and pooling scratch. The weights are not here -- they are shared,
     * read-only, by every worker of the session. */
    struct Worker {
        ggml_backend_t backend = nullptr;
        CachedGraph graph;
        std::vector<float> scores;      /* splade_pool() per-vocabulary maxima */
        std::vector<float> logits_copy; /* only for a non-host graph buffer */
#ifdef LIBEMBEDDING_GGUF_PROFILE
        SparseProfile profile;
#endif

        Worker() = default;
        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;
        ~Worker() {
            graph.reset(); /* before the backend that computed it */
            if (backend) ggml_backend_free(backend);
        }
    };

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

    std::vector<SparseResult> embed_batch(
        Worker& w, const std::vector<std::vector<int>>& token_ids) {
        const int batch = (int)token_ids.size();
        std::vector<SparseResult> results((size_t)batch);

#ifdef LIBEMBEDDING_GGUF_PROFILE
        using ProfileClock = std::chrono::steady_clock;
        auto stage_start = ProfileClock::now();
#endif
        EncodingBatch enc = tok_.encode_batch(token_ids);
        const int seq_len = enc.seq_length;
        if (seq_len <= 0) return results;

        /* One flat row of tokens: [document][position] concatenated, document
         * major. The batch is an axis of its own in the graph, so the mask is
         * [seq_len, seq_len, 1, batch] and is what keeps the documents apart --
         * nothing else has to know where one ends and the next begins, except
         * the pooling, which does, by walking each document's own slice. */
#ifdef LIBEMBEDDING_GGUF_PROFILE
        stage_start = ProfileClock::now();
#endif
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
#ifdef LIBEMBEDDING_GGUF_PROFILE
        w.profile.input_pack_ms += std::chrono::duration<double, std::milli>(
            ProfileClock::now() - stage_start).count();
        w.profile.batches++;
        w.profile.max_seq_len = std::max(w.profile.max_seq_len, seq_len);
        w.profile.padded_tokens += (uint64_t)n_tokens;
        for (int64_t token_valid : valid) {
            if (token_valid != 0) {
                w.profile.real_tokens++;
                w.profile.padded_tokens--;
            }
        }
#endif

        const float* logits = run_graph(w, tokens, positions, valid, seq_len, batch);

#ifdef LIBEMBEDDING_GGUF_PROFILE
        stage_start = ProfileClock::now();
#endif
        for (int b = 0; b < batch; b++) {
            const int begin = b * seq_len;
            splade_pool(logits + (size_t)begin * (size_t)vocab_size_, vocab_size_,
                        valid.data() + begin, seq_len, results[(size_t)b],
                        w.scores);
            drop_special_tokens(results[(size_t)b]);
        }
#ifdef LIBEMBEDDING_GGUF_PROFILE
        w.profile.pooling_ms += std::chrono::duration<double, std::milli>(
            ProfileClock::now() - stage_start).count();
#endif
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

    /* Builds the worker's graph, allocates it, feeds it and returns a pointer to
     * the logits, valid until the worker's next call. Throws if the graph
     * cannot be built or computed so callers cannot mistake a failed inference
     * for a valid empty sparse vector.
     *
     * The graph is cached per worker and rebuilt only when the shape
     * (seq_len, batch) changes. */
    const float* run_graph(Worker& w, const std::vector<int32_t>& tokens,
                           const std::vector<int32_t>& positions,
                           const std::vector<int64_t>& valid,
                           int seq_len, int batch) {
        CachedGraph& g = w.graph;
        const bool rebuild = (g.ctx == nullptr) || (g.cached_seq_len != seq_len) ||
                             (g.cached_batch != batch);

        if (rebuild) {
#ifdef LIBEMBEDDING_GGUF_PROFILE
            const auto graph_build_start = std::chrono::steady_clock::now();
#endif
            g.reset();

            ggml_init_params iparams{};
            iparams.mem_size = ggml_graph_overhead() + ggml_tensor_overhead() * 2048;
            iparams.mem_buffer = nullptr;
            iparams.no_alloc = true;

            g.ctx = ggml_init(iparams);
            if (!g.ctx)
                throw std::runtime_error("failed to initialize sparse GGUF graph");

            g.t_tokens = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, seq_len * batch);
            g.t_pos = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, seq_len * batch);
            g.t_mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, seq_len, seq_len, 1,
                                          batch);
            g.gf = ggml_new_graph(g.ctx);
            if (!g.t_tokens || !g.t_pos || !g.t_mask || !g.gf) {
                g.reset();
                throw std::runtime_error("failed to create sparse GGUF graph inputs");
            }

            g.logits = build_mlm_head(g.ctx, build_bert_encoder(
                                                 g.ctx, *weights_, g.t_tokens, g.t_pos,
                                                 g.t_mask, seq_len, batch));
            if (!g.logits) {
                g.reset();
                throw std::runtime_error("failed to build sparse GGUF forward graph");
            }
            ggml_build_forward_expand(g.gf, g.logits);

            g.galloc = ggml_gallocr_new(weights_->buffer_type());
            if (!g.galloc || !ggml_gallocr_alloc_graph(g.galloc, g.gf)) {
                g.reset();
                throw std::runtime_error("failed to allocate sparse GGUF graph tensors");
            }

            g.cached_seq_len = seq_len;
            g.cached_batch = batch;
#ifdef LIBEMBEDDING_GGUF_PROFILE
            w.profile.graph_build_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - graph_build_start).count();
            w.profile.graph_builds++;
            w.profile.graph_nodes = (uint64_t)ggml_graph_n_nodes(g.gf);
            w.profile.graph_seq_len = seq_len;
            w.profile.graph_batch = batch;
        } else {
            w.profile.graph_cache_hits++;
#endif
        }

        /* After allocation: until then the input tensors have no data to write
         * into. The mask is written from mask.size(), not from n_tokens * n_tokens:
         * those two are the same only when the batch is one, and writing the
         * larger count into the smaller tensor is a heap overflow. */
#ifdef LIBEMBEDDING_GGUF_PROFILE
        auto stage_start = std::chrono::steady_clock::now();
#endif
        const std::vector<float> mask = build_mask(valid, seq_len, batch);
#ifdef LIBEMBEDDING_GGUF_PROFILE
        w.profile.mask_build_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
#endif

#ifdef LIBEMBEDDING_GGUF_PROFILE
        stage_start = std::chrono::steady_clock::now();
#endif
        ggml_backend_tensor_set(g.t_tokens, tokens.data(), 0,
                                (size_t)seq_len * (size_t)batch * sizeof(int32_t));
        ggml_backend_tensor_set(g.t_pos, positions.data(), 0,
                                (size_t)seq_len * (size_t)batch * sizeof(int32_t));
        ggml_backend_tensor_set(g.t_mask, mask.data(), 0,
                                (size_t)mask.size() * sizeof(float));
#ifdef LIBEMBEDDING_GGUF_PROFILE
        w.profile.input_upload_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
        stage_start = std::chrono::steady_clock::now();
#endif

        if (ggml_backend_graph_compute(w.backend, g.gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("sparse GGUF graph computation failed");
        }
#ifdef LIBEMBEDDING_GGUF_PROFILE
        w.profile.graph_exec_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
#endif

        /* The graph lives in a CPU buffer, so the logits are read where they
         * are: copying [vocab, batch * seq_len] floats out first cost a full
         * pass over ~60 MB per 512-token batch and the same again in memory,
         * per worker. The copy stays as the fallback for a non-host buffer. */
        if (ggml_backend_buffer_is_host(g.logits->buffer)) {
            return static_cast<const float*>(g.logits->data);
        }
        const size_t n_values =
            (size_t)vocab_size_ * (size_t)seq_len * (size_t)batch;
        w.logits_copy.resize(n_values);
#ifdef LIBEMBEDDING_GGUF_PROFILE
        stage_start = std::chrono::steady_clock::now();
#endif
        ggml_backend_tensor_get(g.logits, w.logits_copy.data(), 0,
                                n_values * sizeof(float));
#ifdef LIBEMBEDDING_GGUF_PROFILE
        w.profile.logits_download_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stage_start).count();
#endif
        return w.logits_copy.data();
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

    /* Declared before the workers so it is destroyed after them: every
     * worker's graph points at these tensors. */
    std::unique_ptr<Weights> weights_;
    TokenizerWrapper tok_;
    std::string path_;
    std::vector<int32_t> special_ids_;
    int vocab_size_ = 0;
    int max_length_ = 0;
    int max_batch_tokens_ = kDefaultMaxBatchTokens;
    int last_batch_count_ = 0;
    int total_threads_ = 1;
    std::vector<std::unique_ptr<Worker>> workers_;
#ifdef LIBEMBEDDING_GGUF_PROFILE
    SparseProfile profile_; /* the session's own stages: tokenisation */
#endif
};

} /* namespace gguf */
} /* namespace lembed */

#endif /* LIBEMBEDDING_GGUF_SPARSE_SESSION_HPP */