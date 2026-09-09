/*
 * libembedding - detail/llama_session_impl.hpp
 * llama.cpp backend for GGUF embedding models
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_LLAMA_SESSION_IMPL_HPP
#define LIBEMBEDDING_DETAIL_LLAMA_SESSION_IMPL_HPP

#include <llama.h>
#include <ggml.h>

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

#include "worker_autotune_impl.hpp"
#include "llama_scheduler_impl.hpp"

namespace lembed { namespace detail {

inline std::string llama_last_error;

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
            llama_free_model(model_);
            model_ = nullptr;
        }
    }

    LlamaSession(const LlamaSession&) = delete;
    LlamaSession& operator=(const LlamaSession&) = delete;
    LlamaSession(LlamaSession&&) = delete;
    LlamaSession& operator=(LlamaSession&&) = delete;

    void load_from_file(const char* model_path, int num_threads = 0, int n_ctx = 0,
                        int n_gpu_layers = 0, int n_batch = 0, bool verbose = false) {
        static std::once_flag backend_init;
        std::call_once(backend_init, [verbose]() {
            llama_ensure_log_callback();
            llama_backend_init();
        });

        llama_last_error.clear();
        struct llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = (n_gpu_layers < 0) ? 999 : n_gpu_layers;

        model_ = llama_load_model_from_file(model_path, mparams);
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
        if (n_batch > 0) {
            cparams.n_batch = (uint32_t)n_batch;
        }

        ctx_ = llama_init_from_model(model_, cparams);
        if (!ctx_) {
            llama_free_model(model_);
            model_ = nullptr;
            throw std::runtime_error("Failed to create llama context for embeddings");
        }
    }

    int dimension() const { return n_embd_; }

    int max_context() const { return n_ctx_; }

    int tokenize(const char* text, std::vector<llama_token>& tokens, int max_tokens = 0) const {
        if (!vocab_) return 0;
        if (max_tokens <= 0) max_tokens = n_ctx_;

        int text_len = (int)strlen(text);
        int needed = -llama_tokenize(vocab_, text, text_len, nullptr, 0, true, true);
        tokens.resize((size_t)needed);

        int actual = llama_tokenize(vocab_, text, text_len,
                                    tokens.data(), (int32_t)tokens.size(), true, true);
        return actual;
    }

    std::vector<float> embed(const char* text) {
        std::vector<llama_token> tokens;
        tokenize(text, tokens);
        return embed_tokens(tokens);
    }

    /* Public accessor for benchmarking */
    struct llama_context* context() const { return ctx_; }

    std::vector<std::vector<float>> embed_batch(const std::vector<std::string>& texts) {
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (const auto& text : texts) {
            results.push_back(embed(text.c_str()));
        }
        return results;
    }

    std::vector<float> embed_tokens(const std::vector<llama_token>& tokens) {
        if (!ctx_ || tokens.empty()) return std::vector<float>((size_t)n_embd_, 0.0f);

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

    float classify(const char* text) {
        if (!ctx_) throw std::runtime_error("LlamaSession: null context");

        std::vector<llama_token> tokens;
        tokenize(text, tokens);
        if (tokens.empty()) return 0.0f;

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

        int ret = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            throw std::runtime_error("llama_decode failed for classification");
        }

        const float* logits = llama_get_logits(ctx_);
        if (!logits) {
            throw std::runtime_error("Failed to retrieve logits from llama context");
        }

        int last_token = (int)tokens.size() - 1;
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_));
        return logits[last_token * n_vocab];
    }

private:
    struct llama_model* model_;
    struct llama_context* ctx_;
    int n_ctx_;
    int n_embd_;
    int n_threads_;
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

    int tokenize(const char* text, std::vector<llama_token>& tokens) const {
        if (!vocab_) return 0;
        int max_tokens = n_ctx_;
        int text_len = (int)strlen(text);
        int needed = -llama_tokenize(vocab_, text, text_len, nullptr, 0, true, true);
        tokens.resize((size_t)needed);
        return llama_tokenize(vocab_, text, text_len, tokens.data(), (int32_t)tokens.size(), true, true);
    }

public:
    LlamaSessionPool() = default;

    ~LlamaSessionPool() {
        for (auto& ctx : contexts_) {
            if (ctx && ctx->ctx) llama_free(ctx->ctx);
        }
        if (model_) llama_free_model(model_);
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
            llama_backend_init();
        });

        llama_last_error.clear();
        struct llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = (n_gpu_layers < 0) ? 999 : n_gpu_layers;

        model_ = llama_load_model_from_file(model_path, mparams);
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

            auto ctx = std::make_unique<Context>();
            ctx->ctx = llama_init_from_model(model_, cparams);
            if (!ctx->ctx) throw std::runtime_error("Failed to create llama context");
            contexts_.push_back(std::move(ctx));
        }
    }

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
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (const auto& t : texts) out.push_back(embed(t.c_str()));
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
    int num_sessions() const { return (int)contexts_.size(); }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_LLAMA_SESSION_IMPL_HPP */




