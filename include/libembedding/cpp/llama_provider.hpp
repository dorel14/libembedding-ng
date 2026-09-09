/*
 * libembedding - cpp/llama_provider.cpp
 * llama.cpp backend implementation for embeddings.
 *
 * Author: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/cpp/provider.hpp>
#include <libembedding/llamacpp_backend.h>
#include <libembedding/text_embedding.h>
#include <libembedding/detail/status_helper.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lembed {

class LlamaEmbeddingProvider : public EmbeddingProvider {
public:
    LlamaEmbeddingProvider(const std::string& model,
                          const EmbeddingOptions& opts = {}) {
        lembed_text_options_t c_opts = lembed_text_options_default();

        c_opts.provider = parse_provider(opts.provider);
        c_opts.device_id = opts.device_id;
        c_opts.max_length = opts.max_length;
        c_opts.num_threads = opts.threads;
        c_opts.batch_size = opts.batch_size;
        c_opts.offline = opts.offline ? 1 : 0;
        c_opts.show_download_progress = opts.show_download_progress ? 1 : 0;

        int idx = lembed_find_text_model_by_code(model.c_str());
        if (idx >= 0) {
            c_opts.model = (lembed_text_model_t)idx;
            detail::check_status(lembed_text_embedding_create(&c_opts, &ctx_));
        } else if (std::filesystem::exists(model)) {
            detail::check_status(lembed_text_embedding_create_from_path(
                model.c_str(), &c_opts, &ctx_));
        } else {
            throw std::invalid_argument("Unknown model or path: " + model);
        }

        dim_ = lembed_text_embedding_dim(ctx_);
    }

    ~LlamaEmbeddingProvider() { close(); }

    LlamaEmbeddingProvider(const LlamaEmbeddingProvider&) = delete;
    LlamaEmbeddingProvider& operator=(const LlamaEmbeddingProvider&) = delete;

    LlamaEmbeddingProvider(LlamaEmbeddingProvider&& other) noexcept
        : ctx_(other.ctx_), dim_(other.dim_) {
        other.ctx_ = nullptr;
    }

    LlamaEmbeddingProvider& operator=(LlamaEmbeddingProvider&& other) noexcept {
        if (this != &other) {
            close();
            ctx_ = other.ctx_;
            dim_ = other.dim_;
            other.ctx_ = nullptr;
        }
        return *this;
    }

    std::vector<std::vector<float>> embed(const std::vector<std::string>& texts,
                                          int batch_size = 0) override {
        if (texts.empty()) return {};

        std::vector<const char*> c_texts;
        c_texts.reserve(texts.size());
        for (const auto& t : texts) c_texts.push_back(t.c_str());

        lembed_embeddings_t result = {0};
        detail::check_status(lembed_text_embedding_embed(
            ctx_, c_texts.data(), (int)texts.size(), batch_size, &result));

        std::vector<std::vector<float>> embeddings;
        try {
            embeddings.resize(result.num_embeddings);
            for (int i = 0; i < result.num_embeddings; i++) {
                embeddings[i].assign(
                    result.data + (size_t)i * result.dim,
                    result.data + (size_t)(i + 1) * result.dim);
            }
        } catch (...) {
            lembed_embeddings_free(&result);
            throw;
        }
        lembed_embeddings_free(&result);
        return embeddings;
    }

    int dimension() const override {
        return ctx_ ? lembed_text_embedding_dim(ctx_) : 0;
    }

    std::string name() const override {
        if (!ctx_) return "";
        const char* n = lembed_text_embedding_model_name(ctx_);
        return n ? std::string(n) : "";
    }

    void close() override {
        if (ctx_) {
            lembed_text_embedding_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    lembed_text_embedding_t* ctx_ = nullptr;
    int dim_ = 0;
};

} /* namespace lembed */