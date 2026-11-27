/*
 * libembedding - detail/llama_session_impl.hpp
 * llama.cpp backend for GGUF embedding models
 *
 * Auteur: David Orel
 * Version: 1.16.7
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_LLAMA_SESSION_IMPL_HPP
#define LIBEMBEDDING_DETAIL_LLAMA_SESSION_IMPL_HPP

#include <llama.h>
#include <ggml.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml_cpu_backend.hpp"
#include "worker_autotune_impl.hpp"
#include "llama_scheduler_impl.hpp"
#include "tokenizer_impl.hpp"

namespace lembed { namespace detail {

inline std::string llama_last_error;

/* llama.cpp's WPM tokenizer applies NFD through a 1:1 codepoint table, so a
 * Hangul syllable collapses to its leading consonant jamo and the vowel and
 * final are lost (HF decomposes into 2-3 jamo). For WPM (BERT) vocabs holding
 * Hangul syllables, pre-decompose with the library's HF-exact BERT
 * normalization (NFD, strip Mn, lowercase: the same defaults llama.cpp uses for
 * WPM). Jamo are untouched by llama's own pass. Returns true and fills `out`
 * only in that case; other vocab types and other texts are left alone. */
inline bool llama_wpm_hangul_prepare(const struct llama_vocab* vocab, const char* text,
                                     std::string& out) {
    if (!vocab || !text || llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_WPM) return false;
    bool hangul = false;
    for (const unsigned char* p = (const unsigned char*)text; *p && !hangul; ++p) {
        /* Syllables U+AC00..U+D7A3 are 3-byte: EA B0.. / EB / EC / ED ..9E */
        if (*p < 0xEA || *p > 0xED) continue;
        const unsigned char b1 = p[1];
        hangul = *p != 0xED ? (*p != 0xEA || b1 >= 0xB0) : b1 <= 0x9E;
    }
    if (!hangul) return false;
    out = bert_normalize(std::string(text));
    return true;
}

inline void llama_log_callback(ggml_log_level level, const char* text, void* userdata) {
    (void)userdata;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "[llama.cpp] %s", text);
        llama_last_error = text;
    }
}

inline void llama_ensure_log_callback() {
    static std::once_flag log_init;
    std::call_once(log_init, []() {
        llama_log_set(llama_log_callback, nullptr);
    });
}

/* Builds the cross-encoder input for a (query, document) pair from already
 * tokenized segments, the way HF tokenizers do for a real text pair:
 *   WordPiece / BERT : [CLS] q [SEP] d [SEP]
 *   others (XLM-R)   : <s> q </s></s> d </s>
 * Special ids come from the GGUF vocab (pass -1 for "absent"; absent ones are
 * skipped). Truncation is HF "longest_first": drop one token at a time from
 * the longer segment until the whole sequence fits max_len, specials kept. */
inline std::vector<llama_token> build_rerank_pair_tokens(
        std::vector<llama_token> q, std::vector<llama_token> d,
        llama_token bos, llama_token sep, llama_token eos,
        bool wordpiece, int max_len) {
    std::vector<llama_token> head, mid, tail;
    if (bos >= 0) head.push_back(bos);
    if (wordpiece) {
        llama_token s = (sep >= 0) ? sep : eos;
        if (s >= 0) { mid.push_back(s); tail.push_back(s); }
    } else if (eos >= 0) {
        mid.push_back(eos); mid.push_back(eos); tail.push_back(eos);
    }
    size_t overhead = head.size() + mid.size() + tail.size();
    size_t budget = (max_len > 0 && (size_t)max_len > overhead)
                        ? (size_t)max_len - overhead : 0;
    while (q.size() + d.size() > budget) {
        if (q.size() > d.size()) q.pop_back(); else d.pop_back();
    }
    std::vector<llama_token> out;
    out.reserve(overhead + q.size() + d.size());
    out.insert(out.end(), head.begin(), head.end());
    out.insert(out.end(), q.begin(), q.end());
    out.insert(out.end(), mid.begin(), mid.end());
    out.insert(out.end(), d.begin(), d.end());
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

/* Thrown when a GGUF loads fine but cannot be used by this backend. Callers map
 * it to LEMBED_ERROR_UNSUPPORTED instead of the generic LEMBED_ERROR_LLAMA. */
struct LlamaUnsupportedError : public std::runtime_error {
    explicit LlamaUnsupportedError(const std::string& what)
        : std::runtime_error(what) {}
};

/* The dense and reranker paths only call llama_encode(). It serves encoder
 * stacks (T5) and bidirectional BERT-style models (non-causal attention). A
 * decoder-only model with causal attention (Qwen3-Embedding, ...) loads but
 * then crashes at the first embed (access violation), so refuse it once the
 * context exists. llama_model_has_encoder() alone is not enough: it is true
 * for T5 only, and false for BERT. */
inline void llama_require_encoder(const struct llama_model* model,
                                  const struct llama_context* ctx, const char* path,
                                  bool reranker = false) {
    if (!llama_model_has_encoder(model) && llama_get_causal_attn(ctx)) {
        std::string msg = reranker
            ? "decoder-only (causal) GGUF rerankers such as Qwen3-Reranker are not supported yet: "
            : "decoder-only GGUF embedding models are not supported yet: ";
        msg += path ? path : "";
        throw LlamaUnsupportedError(msg);
    }
}

/* Largest token count one llama_encode() call accepts for this context. A
 * non-causal encode asserts (GGML_ASSERT, process abort) when n_tokens exceeds
 * n_ubatch, and cannot exceed n_batch or n_ctx either. */
inline int llama_encode_token_limit(const struct llama_context* ctx) {
    int64_t lim = (int64_t)llama_n_ctx(ctx);
    const int64_t nb = (int64_t)llama_n_batch(ctx);
    const int64_t nu = (int64_t)llama_n_ubatch(ctx);
    if (nb > 0 && nb < lim) lim = nb;
    if (nu > 0 && nu < lim) lim = nu;
    return lim > 1 ? (int)lim : 1;
}

/* The pooled sequence vector is n_embd floats only for MEAN/CLS/LAST; RANK
 * yields n_cls_out floats and NONE no sequence vector, so reading n_embd would
 * overrun it. Throws instead. */
inline void llama_require_dense_pooling(const struct llama_context* ctx) {
    const enum llama_pooling_type pooling = llama_pooling_type(ctx);
    if (pooling == LLAMA_POOLING_TYPE_RANK || pooling == LLAMA_POOLING_TYPE_NONE) {
        throw std::runtime_error("GGUF model has no sequence pooling usable for dense embeddings");
    }
}

/* Truncates `tokens` to at most `limit`, keeping the trailing EOS/SEP when the
 * model appended one (same as HF truncation: specials survive). */
inline void llama_truncate_tokens(const struct llama_vocab* vocab,
                                  std::vector<llama_token>& tokens, int limit) {
    if (limit < 1) limit = 1;
    if ((size_t)limit >= tokens.size()) return;
    const llama_token last = tokens.back();
    const bool keep_end = (limit > 1) &&
        ((llama_vocab_get_add_eos(vocab) && last == llama_vocab_eos(vocab)) ||
         (llama_vocab_get_add_sep(vocab) && last == llama_vocab_sep(vocab)));
    tokens.resize((size_t)limit);
    if (keep_end) tokens.back() = last;
}

/* Dense batching: several texts go through one llama_encode() as distinct
 * sequences (seq_id 0..k-1) instead of one encode per text. A non-causal
 * encode computes full n_tokens x n_tokens attention (block-diagonal mask), so
 * packing is capped in total tokens to keep that quadratic term small.
 * n_seq_max is fixed at context creation.
 * Values come from a sweep on MiniLM Q8_0 (i7-1065G7, mixed corpus): 8
 * sequences / 256 tokens per encode gave the best throughput; 512 tokens cost
 * +35 MB RSS for no further gain, 128 tokens lost about 7 %. */
inline int llama_pack_seqs()   { return 8; }
inline int llama_pack_tokens() { return 256; }

/* Splits the non-empty token lists into consecutive groups of at most
 * max_seqs sequences and max_tokens total tokens (a lone longer text still
 * forms its own group; callers already truncated it to the encode limit). */
inline std::vector<std::vector<size_t>> llama_pack_groups(
        const std::vector<std::vector<llama_token>>& toks, int max_seqs, int max_tokens) {
    std::vector<std::vector<size_t>> groups;
    size_t used = 0;
    for (size_t i = 0; i < toks.size(); i++) {
        if (toks[i].empty()) continue;
        if (groups.empty() || (int)groups.back().size() >= max_seqs ||
            used + toks[i].size() > (size_t)max_tokens) {
            groups.emplace_back();
            used = 0;
        }
        groups.back().push_back(i);
        used += toks[i].size();
    }
    return groups;
}

/* One llama_encode() over the sequences `group` (indices into toks); the
 * pooled vector of group[k] is copied to out + group[k] * n_embd. Caller holds
 * the context exclusively and has checked the pooling type. */
inline void llama_encode_packed(struct llama_context* ctx,
                                const std::vector<std::vector<llama_token>>& toks,
                                const std::vector<size_t>& group, int n_embd, float* out) {
    size_t total = 0;
    for (size_t g : group) total += toks[g].size();
    llama_batch batch = llama_batch_init((int32_t)total, 0, 1);
    batch.n_tokens = 0;
    for (size_t k = 0; k < group.size(); k++) {
        const auto& t = toks[group[k]];
        for (size_t i = 0; i < t.size(); i++) {
            batch.token[batch.n_tokens] = t[i];
            batch.pos[batch.n_tokens] = (llama_pos)i;
            batch.seq_id[batch.n_tokens][0] = (llama_seq_id)k;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.logits[batch.n_tokens] = 0;
            batch.n_tokens++;
        }
    }
    llama_memory_clear(llama_get_memory(ctx), false);
    int ret = llama_encode(ctx, batch);
    llama_batch_free(batch);
    if (ret != 0) throw std::runtime_error("llama_encode failed for embedding extraction");
    for (size_t k = 0; k < group.size(); k++) {
        const float* p = llama_get_embeddings_seq(ctx, (llama_seq_id)k);
        if (!p) throw std::runtime_error("Failed to retrieve embeddings from llama context");
        memcpy(out + group[k] * (size_t)n_embd, p, (size_t)n_embd * sizeof(float));
    }
}

class LlamaSession {
public:
    LlamaSession()
        : model_(nullptr), ctx_(nullptr), n_ctx_(0), n_embd_(0),
          n_threads_(0), vocab_(nullptr) {}

    ~LlamaSession() {
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }

    LlamaSession(const LlamaSession&) = delete;
    LlamaSession& operator=(const LlamaSession&) = delete;
    LlamaSession(LlamaSession&&) = delete;
    LlamaSession& operator=(LlamaSession&&) = delete;

    void load_from_file(const char* model_path, int num_threads = 0, int n_ctx = 0,
                        int n_gpu_layers = 0, int n_batch = 0, bool verbose = false,
                        bool cross_encoder = false) {
        static std::once_flag backend_init;
        std::call_once(backend_init, [verbose]() {
            llama_ensure_log_callback();
            /* With CPU dispatch llama.cpp has no device until the CPU variant
             * is loaded; a no-op with a statically linked CPU backend. */
            ggml_cpu_backend_ensure_loaded();
            llama_backend_init();
        });

        llama_last_error.clear();
        struct llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = (n_gpu_layers < 0) ? 999 : n_gpu_layers;

        model_ = llama_model_load_from_file(model_path, mparams);
        if (!model_) {
            std::string msg = "Failed to load GGUF model: ";
            msg += model_path;
            if (!llama_last_error.empty()) {
                msg += " | llama.cpp error: ";
                msg += llama_last_error;
            }
            throw std::runtime_error(msg);
        }

        vocab_ = llama_model_get_vocab(model_);
        n_embd_ = llama_model_n_embd(model_);
        int32_t ctx_default = llama_model_n_ctx_train(model_);
        if (ctx_default <= 0) ctx_default = 4096;
        n_ctx_ = (n_ctx > 0) ? n_ctx : ctx_default;

        n_threads_ = (num_threads > 0) ? num_threads : 4;

        struct llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t)n_ctx_;
        cparams.n_threads = (int32_t)n_threads_;
        cparams.n_threads_batch = (int32_t)n_threads_;
        cparams.embeddings = true;
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
        /* Cross-encoder GGUFs carry a classifier head but often no pooling_type
         * key, so llama.cpp would default to NONE and expose no score. Only the
         * reranker asks for it: llama_model_n_cls_out() is 1 for every model
         * (hparams default), so it cannot tell a cross-encoder from a dense
         * embedder. Forcing RANK on a dense model made llama_get_embeddings_seq()
         * return one float and embed_tokens() read n_embd floats from it. */
        if (cross_encoder && llama_model_n_cls_out(model_) > 0) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_RANK;
        }
        if (n_batch > 0) {
            cparams.n_batch = (uint32_t)n_batch;
        }
        if (!cross_encoder) {
            /* Packed dense batches: one sequence per text. kv_unified keeps
             * n_ctx as requested (otherwise n_ctx is split and padded per
             * sequence, which would also change the truncation limit). */
            cparams.n_seq_max = (uint32_t)llama_pack_seqs();
            cparams.kv_unified = true;
        }

        ctx_ = llama_init_from_model(model_, cparams);
        if (!ctx_) {
            llama_model_free(model_);
            model_ = nullptr;
            throw std::runtime_error("Failed to create llama context for embeddings");
        }
        /* Dense and reranker alike: Qwen3-Reranker (causal) created fine but
         * segfaulted at the first rerank in 1.16.0, so refuse it cleanly. */
        try {
            llama_require_encoder(model_, ctx_, model_path, cross_encoder);
        } catch (...) {
            llama_free(ctx_);
            ctx_ = nullptr;
            llama_model_free(model_);
            model_ = nullptr;
            throw;
        }
        encode_limit_ = llama_encode_token_limit(ctx_);
    }

    int dimension() const { return n_embd_; }

    int max_context() const { return n_ctx_; }

    /* Max tokens one encode call accepts (context, n_batch and n_ubatch). */
    int max_encode_tokens() const { return encode_limit_ > 0 ? encode_limit_ : n_ctx_; }

    /* Sets the user max_length (0 = none); applied by tokenize(). */
    void set_max_length(int max_length) { max_length_ = max_length; }

    /* Tokenizes with special tokens, then truncates to the smallest of
     * max_tokens (if > 0), the max_length option, the model context and the
     * encode batch limit; a trailing EOS/SEP is kept. Returns the token count. */
    int tokenize(const char* text, std::vector<llama_token>& tokens, int max_tokens = 0) const {
        tokens.clear();
        if (!vocab_) return 0;
        int limit = max_encode_tokens();
        if (n_ctx_ > 0 && n_ctx_ < limit) limit = n_ctx_;
        if (max_length_ > 0 && max_length_ < limit) limit = max_length_;
        if (max_tokens > 0 && max_tokens < limit) limit = max_tokens;

        std::string prepared;
        if (llama_wpm_hangul_prepare(vocab_, text, prepared)) text = prepared.c_str();
        int text_len = (int)strlen(text);
        int needed = -llama_tokenize(vocab_, text, text_len, nullptr, 0, true, true);
        if (needed <= 0) return 0;
        tokens.resize((size_t)needed);

        int actual = llama_tokenize(vocab_, text, text_len,
                                    tokens.data(), (int32_t)tokens.size(), true, true);
        tokens.resize(actual > 0 ? (size_t)actual : 0);
        if (!tokens.empty()) llama_truncate_tokens(vocab_, tokens, limit);
        return (int)tokens.size();
    }

    std::vector<float> embed(const char* text) {
        std::vector<llama_token> tokens;
        tokenize(text, tokens);
        return embed_tokens(tokens);
    }

    /* Public accessor for benchmarking */
    struct llama_context* context() const { return ctx_; }

    std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts) {
        std::vector<const char*> ptrs;
        ptrs.reserve(texts.size());
        for (const auto& t : texts) ptrs.push_back(t.c_str());
        std::vector<float> flat((size_t)texts.size() * (size_t)n_embd_);
        embed_texts(ptrs.data(), (int)ptrs.size(), flat.data());
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            results.emplace_back(flat.begin() + (ptrdiff_t)(i * (size_t)n_embd_),
                                 flat.begin() + (ptrdiff_t)((i + 1) * (size_t)n_embd_));
        }
        return results;
    }

    /* Embeds n texts into out (n * dimension() floats, raw pooled vectors),
     * packing them into shared llama_encode() calls. Empty tokenizations give
     * a zero vector, like embed(). */
    void embed_texts(const char* const* texts, int n, float* out) {
        if (n <= 0) return;
        std::fill(out, out + (size_t)n * (size_t)n_embd_, 0.0f);
        if (!ctx_) return;
        llama_require_dense_pooling(ctx_);
        std::vector<std::vector<llama_token>> toks((size_t)n);
        for (int i = 0; i < n; i++) tokenize(texts[i], toks[(size_t)i]);
        int max_seqs = (int)llama_n_seq_max(ctx_);
        if (max_seqs < 1) max_seqs = 1;
        int max_tokens = llama_pack_tokens();
        if (max_tokens > max_encode_tokens()) max_tokens = max_encode_tokens();
        for (const auto& g : llama_pack_groups(toks, max_seqs, max_tokens)) {
            llama_encode_packed(ctx_, toks, g, n_embd_, out);
        }
    }

    std::vector<float> embed_tokens(const std::vector<llama_token>& tokens) {
        if (!ctx_ || tokens.empty()) return std::vector<float>((size_t)n_embd_, 0.0f);

        llama_require_dense_pooling(ctx_);

        /* Defence in depth: callers going through tokenize() are already within
         * the limit, but a longer sequence would abort the process. */
        if (tokens.size() > (size_t)max_encode_tokens()) {
            std::vector<llama_token> cut(tokens);
            llama_truncate_tokens(vocab_, cut, max_encode_tokens());
            return embed_tokens(cut);
        }

        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        batch.n_tokens = 0;

        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[batch.n_tokens] = tokens[i];
            batch.pos[batch.n_tokens] = (llama_pos)i;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.logits[batch.n_tokens] = 0;
            batch.n_tokens++;
        }

        llama_memory_clear(llama_get_memory(ctx_), false);

        int ret = llama_encode(ctx_, batch);
        if (ret != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_encode failed for embedding extraction");
        }

        const float* embd_ptr = llama_get_embeddings_seq(ctx_, 0);
        if (!embd_ptr) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to retrieve embeddings from llama context");
        }

        std::vector<float> result((size_t)n_embd_);
        memcpy(result.data(), embd_ptr, (size_t)n_embd_ * sizeof(float));

        llama_batch_free(batch);
        return result;
    }

    /* Tokenizes a segment without adding or parsing special tokens. */
    std::vector<llama_token> tokenize_plain(const char* text) const {
        std::vector<llama_token> toks;
        if (!vocab_ || !text || !text[0]) return toks;
        std::string prepared;
        if (llama_wpm_hangul_prepare(vocab_, text, prepared)) text = prepared.c_str();
        int len = (int)strlen(text);
        int needed = -llama_tokenize(vocab_, text, len, nullptr, 0, false, false);
        if (needed <= 0) return toks;
        toks.resize((size_t)needed);
        int n = llama_tokenize(vocab_, text, len, toks.data(),
                               (int32_t)toks.size(), false, false);
        toks.resize(n > 0 ? (size_t)n : 0);
        return toks;
    }

    /* Real sentence-pair encoding (see build_rerank_pair_tokens). Special
     * ids and the template family come from the GGUF vocab. */
    std::vector<llama_token> pair_tokens(const char* query, const char* document,
                                         int max_len) const {
        if (!vocab_) return {};
        const bool wpm = llama_vocab_type(vocab_) == LLAMA_VOCAB_TYPE_WPM;
        llama_token bos = llama_vocab_get_add_bos(vocab_) ? llama_vocab_bos(vocab_) : -1;
        llama_token sep = llama_vocab_sep(vocab_);
        llama_token eos = llama_vocab_eos(vocab_);
        int cap = n_ctx_;
        if (max_len > 0 && max_len < cap) cap = max_len;
        if (cap > max_encode_tokens()) cap = max_encode_tokens(); /* n_ubatch bounds one encode call */
        return build_rerank_pair_tokens(tokenize_plain(query), tokenize_plain(document),
                                        bos, sep, eos, wpm, cap);
    }

    /* Cross-encoder score from an already built token sequence. The GGUF
     * classifier head uses RANK pooling: the score is the pooled output of
     * sequence 0, not a vocabulary logit. */
    float classify_tokens(const std::vector<llama_token>& tokens) {
        if (!ctx_) throw std::runtime_error("LlamaSession: null context");
        if (tokens.empty()) return 0.0f;
        if (tokens.size() > (size_t)max_encode_tokens()) {
            std::vector<llama_token> cut(tokens);
            llama_truncate_tokens(vocab_, cut, max_encode_tokens());
            return classify_tokens(cut);
        }

        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        batch.n_tokens = 0;
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[batch.n_tokens] = tokens[i];
            batch.pos[batch.n_tokens] = (llama_pos)i;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.logits[batch.n_tokens] = 1;
            batch.n_tokens++;
        }

        llama_memory_clear(llama_get_memory(ctx_), false);
        int ret = llama_encode(ctx_, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            throw std::runtime_error("llama_encode failed for classification");
        }

        const float* score = llama_get_embeddings_seq(ctx_, 0);
        if (!score) {
            throw std::runtime_error("Failed to retrieve rank score from llama context");
        }
        return score[0];
    }

    float classify(const char* text) {
        std::vector<llama_token> tokens;
        tokenize(text, tokens);
        return classify_tokens(tokens);
    }

private:
    struct llama_model* model_;
    struct llama_context* ctx_;
    int n_ctx_;
    int n_embd_;
    int n_threads_;
    int encode_limit_ = 0;
    int max_length_ = 0;
    const struct llama_vocab* vocab_;
};

class LlamaSessionPool {
    struct Context {
        struct llama_context* ctx = nullptr;
        std::mutex mtx;
    };

    struct llama_model* model_ = nullptr;
    const struct llama_vocab* vocab_ = nullptr;
    int n_embd_ = 0;
    int n_ctx_ = 0;
    std::vector<std::unique_ptr<Context>> contexts_;
    std::atomic<size_t> next_idx_{0};
    std::unique_ptr<LlamaScheduler> scheduler_;

    int encode_limit_ = 0;
    int max_length_ = 0;

    /* Same truncation rules as LlamaSession::tokenize(). */
    int tokenize(const char* text, std::vector<llama_token>& tokens) const {
        tokens.clear();
        if (!vocab_) return 0;
        int limit = encode_limit_ > 0 ? encode_limit_ : n_ctx_;
        if (n_ctx_ > 0 && n_ctx_ < limit) limit = n_ctx_;
        if (max_length_ > 0 && max_length_ < limit) limit = max_length_;
        std::string prepared;
        if (llama_wpm_hangul_prepare(vocab_, text, prepared)) text = prepared.c_str();
        int text_len = (int)strlen(text);
        int needed = -llama_tokenize(vocab_, text, text_len, nullptr, 0, true, true);
        if (needed <= 0) return 0;
        tokens.resize((size_t)needed);
        int n = llama_tokenize(vocab_, text, text_len, tokens.data(), (int32_t)tokens.size(), true, true);
        tokens.resize(n > 0 ? (size_t)n : 0);
        if (!tokens.empty()) llama_truncate_tokens(vocab_, tokens, limit);
        return (int)tokens.size();
    }

public:
    LlamaSessionPool() = default;

    ~LlamaSessionPool() {
        for (auto& ctx : contexts_) {
            if (ctx && ctx->ctx) llama_free(ctx->ctx);
        }
        if (model_) llama_model_free(model_);
    }

    LlamaSessionPool(const LlamaSessionPool&) = delete;
    LlamaSessionPool& operator=(const LlamaSessionPool&) = delete;

    void load_from_file(const char* model_path, int n_sessions, int num_threads_per_session = 1,
                        int n_ctx = 0, int n_gpu_layers = 0, int n_batch = 0, bool verbose = false) {
        if (n_sessions <= 0) {
            throw std::runtime_error("LlamaSessionPool: n_sessions must be > 0");
        }

        static std::once_flag backend_init;
        std::call_once(backend_init, [verbose]() {
            llama_ensure_log_callback();
            ggml_cpu_backend_ensure_loaded();  // see LlamaSession::load_from_file
            llama_backend_init();
        });

        llama_last_error.clear();
        struct llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = (n_gpu_layers < 0) ? 999 : n_gpu_layers;

        model_ = llama_model_load_from_file(model_path, mparams);
        if (!model_) {
            std::string msg = "Failed to load GGUF: ";
            msg += model_path;
            if (!llama_last_error.empty()) {
                msg += " | llama.cpp error: ";
                msg += llama_last_error;
            }
            throw std::runtime_error(msg);
        }

        vocab_ = llama_model_get_vocab(model_);
        n_embd_ = llama_model_n_embd(model_);
        int32_t ctx_default = llama_model_n_ctx_train(model_);
        if (ctx_default <= 0) ctx_default = 4096;
        n_ctx_ = (n_ctx > 0) ? n_ctx : ctx_default;

        contexts_.clear();
        for (int i = 0; i < n_sessions; i++) {
            struct llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = (uint32_t)n_ctx_;
            cparams.n_threads = (int32_t)num_threads_per_session;
            cparams.n_threads_batch = (int32_t)num_threads_per_session;
            cparams.embeddings = true;
            cparams.type_k = GGML_TYPE_F16;
            cparams.type_v = GGML_TYPE_F16;
            if (n_batch > 0) {
                cparams.n_batch = (uint32_t)n_batch;
            }
            /* Packed dense batches, see LlamaSession::load_from_file. */
            cparams.n_seq_max = (uint32_t)llama_pack_seqs();
            cparams.kv_unified = true;

            auto ctx = std::make_unique<Context>();
            ctx->ctx = llama_init_from_model(model_, cparams);
            if (!ctx->ctx) throw std::runtime_error("Failed to create llama context");
            if (i == 0) {
                try {
                    llama_require_encoder(model_, ctx->ctx, model_path);
                } catch (...) {
                    llama_free(ctx->ctx);
                    ctx->ctx = nullptr;
                    llama_model_free(model_);
                    model_ = nullptr;
                    throw;
                }
                encode_limit_ = llama_encode_token_limit(ctx->ctx);
            }
            contexts_.push_back(std::move(ctx));
        }
    }

    /* Sets the user max_length (0 = none); applied by tokenize(). */
    void set_max_length(int max_length) { max_length_ = max_length; }

    void load_from_file_auto(const char* model_path, int n_ctx = 0, int n_gpu_layers = 0,
                             int n_batch = 0, bool verbose = false) {
        int workers = recommended_workers_for_model(model_path);
        load_from_file(model_path, workers, 1, n_ctx, n_gpu_layers, n_batch, verbose);
    }

    std::vector<float> embed(const char* text) {
        if (contexts_.empty()) {
            throw std::runtime_error("LlamaSessionPool: no contexts available");
        }

        size_t idx = next_idx_.fetch_add(1, std::memory_order_relaxed) % contexts_.size();
        auto& ctx = *contexts_[idx];

        std::unique_lock<std::mutex> lock(ctx.mtx);

        std::vector<llama_token> tokens;
        tokenize(text, tokens);
        if (tokens.empty()) return std::vector<float>((size_t)n_embd_, 0.0f);

        llama_require_dense_pooling(ctx.ctx);

        /* Defence in depth: tokenize() already truncates, but a longer
         * sequence would abort the process in llama_encode(). */
        const int max_tokens = encode_limit_ > 0 ? encode_limit_ : n_ctx_;
        if (tokens.size() > (size_t)max_tokens) llama_truncate_tokens(vocab_, tokens, max_tokens);

        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        batch.n_tokens = 0;
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[batch.n_tokens] = tokens[i];
            batch.pos[batch.n_tokens] = (llama_pos)i;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.logits[batch.n_tokens] = 0;
            batch.n_tokens++;
        }

        llama_memory_clear(llama_get_memory(ctx.ctx), false);
        int ret = llama_encode(ctx.ctx, batch);
        if (ret != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_encode failed");
        }

        const float* embd_ptr = llama_get_embeddings_seq(ctx.ctx, 0);
        if (!embd_ptr) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to retrieve embeddings");
        }

        std::vector<float> result((size_t)n_embd_);
        memcpy(result.data(), embd_ptr, (size_t)n_embd_ * sizeof(float));
        llama_batch_free(batch);
        return result;
    }

    std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts) {
        if (texts.empty()) return {};
        if (contexts_.empty()) {
            throw std::runtime_error("LlamaSessionPool: no contexts available");
        }
        std::vector<std::vector<llama_token>> toks(texts.size());
        for (size_t i = 0; i < texts.size(); i++) tokenize(texts[i].c_str(), toks[i]);
        int max_seqs = (int)llama_n_seq_max(contexts_[0]->ctx);
        if (max_seqs < 1) max_seqs = 1;
        int max_tokens = llama_pack_tokens();
        const int enc_limit = encode_limit_ > 0 ? encode_limit_ : n_ctx_;
        if (max_tokens > enc_limit) max_tokens = enc_limit;

        std::vector<float> flat(texts.size() * (size_t)n_embd_, 0.0f);
        for (const auto& g : llama_pack_groups(toks, max_seqs, max_tokens)) {
            size_t idx = next_idx_.fetch_add(1, std::memory_order_relaxed) % contexts_.size();
            auto& c = *contexts_[idx];
            std::unique_lock<std::mutex> lock(c.mtx);
            llama_require_dense_pooling(c.ctx);
            llama_encode_packed(c.ctx, toks, g, n_embd_, flat.data());
        }
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            out.emplace_back(flat.begin() + (ptrdiff_t)(i * (size_t)n_embd_),
                             flat.begin() + (ptrdiff_t)((i + 1) * (size_t)n_embd_));
        }
        return out;
    }

    std::vector<std::vector<float>> embed_batch_bucketed(const std::vector<std::string>& texts, int batch_size = 32) {
        if (texts.empty()) return {};

        std::vector<std::pair<size_t, std::string>> indexed;
        indexed.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            indexed.emplace_back(i, texts[i]);
        }

        std::sort(indexed.begin(), indexed.end(),
                  [this](const auto& a, const auto& b) {
                      size_t la = tokenize_length(a.second);
                      size_t lb = tokenize_length(b.second);
                      return la < lb;
                  });

        std::vector<std::vector<float>> out(texts.size());
        std::vector<std::string> batch_texts;
        std::vector<size_t> batch_indices;
        batch_texts.reserve(batch_size);
        batch_indices.reserve(batch_size);

        for (size_t i = 0; i < indexed.size(); ++i) {
            batch_texts.push_back(indexed[i].second);
            batch_indices.push_back(indexed[i].first);

            if ((int)batch_texts.size() >= batch_size || i == indexed.size() - 1) {
                auto batch_result = embed_batch(batch_texts);
                for (size_t j = 0; j < batch_indices.size(); ++j) {
                    out[batch_indices[j]] = std::move(batch_result[j]);
                }
                batch_texts.clear();
                batch_indices.clear();
            }
        }

        return out;
    }

    size_t tokenize_length(const std::string& text) const {
        std::vector<llama_token> tokens;
        return (size_t)tokenize(text.c_str(), tokens);
    }

    void enable_scheduler(int max_batch_size = 32, int max_wait_ms = 5) {
        if (!scheduler_) {
            auto self = this;
            scheduler_ = std::make_unique<LlamaScheduler>(
                [self](const std::vector<std::string>& batch,
                       std::vector<std::vector<float>>& results) {
                    results = self->embed_batch(batch);
                },
                max_batch_size,
                max_wait_ms
            );
        }
    }

    void disable_scheduler() {
        if (scheduler_) {
            scheduler_->stop();
            scheduler_.reset();
        }
    }

    bool has_scheduler() const { return scheduler_ != nullptr; }

    std::vector<float> embed_scheduled(const std::string& text) {
        if (!scheduler_) throw std::runtime_error("LlamaSessionPool: scheduler not enabled");
        return scheduler_->submit(text);
    }

    int dimension() const { return n_embd_; }
    int max_context() const { return n_ctx_; }
    int num_sessions() const { return (int)contexts_.size(); }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_LLAMA_SESSION_IMPL_HPP */




