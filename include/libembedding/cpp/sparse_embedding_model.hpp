/*
 * libembedding - cpp/sparse_embedding_model.hpp
 * C++ wrapper for sparse text embedding.
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CPP_SPARSE_EMBEDDING_MODEL_HPP
#define LIBEMBEDDING_CPP_SPARSE_EMBEDDING_MODEL_HPP

#include <libembedding/sparse_text_embedding.h>
#include <libembedding/cpp/provider.hpp>
#include <libembedding/detail/status_helper.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lembed {

struct SparseEmbeddingOptions {
    std::string                  model_path = "prithvida/SPLADE_PP_en_v1";
    int                          threads = 0;
    int                          batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    bool                         offline = false;
    std::string                  cache_dir;
    std::string                  provider = "cpu";
    int                          device_id = 0;
    int                          max_length = 0;
    bool                         show_download_progress = true;
};

struct SparseVector {
    std::vector<int32_t> indices;
    std::vector<float>   values;
};

class SparseEmbeddingModel {
public:
    SparseEmbeddingModel(const std::string& model, const SparseEmbeddingOptions& opts = {}) {
        lembed_sparse_options_t c_opts = lembed_sparse_options_default();

        c_opts.provider = parse_provider(opts.provider);
        c_opts.device_id = opts.device_id;
        c_opts.max_length = opts.max_length;
        c_opts.num_threads = opts.threads;
        c_opts.batch_size = opts.batch_size;
        c_opts.offline = opts.offline ? 1 : 0;
        c_opts.show_download_progress = opts.show_download_progress ? 1 : 0;
        if (!opts.cache_dir.empty()) {
            cache_dir_buf_ = opts.cache_dir;
            c_opts.cache_dir = cache_dir_buf_.c_str();
        }

        int idx = lembed_find_sparse_model_by_code(model.c_str());
        if (idx >= 0) {
            c_opts.model = (lembed_sparse_model_t)idx;
            detail::check_status(lembed_sparse_text_embedding_create(&c_opts, &ctx_));
        } else if (std::filesystem::exists(model)) {
            detail::check_status(lembed_sparse_text_embedding_create_from_path(
                model.c_str(), &c_opts, &ctx_));
        } else {
            throw std::invalid_argument("Unknown model or path: " + model);
        }
    }

    ~SparseEmbeddingModel() { close(); }

    SparseEmbeddingModel(const SparseEmbeddingModel&) = delete;
    SparseEmbeddingModel& operator=(const SparseEmbeddingModel&) = delete;

    SparseEmbeddingModel(SparseEmbeddingModel&& other) noexcept
        : ctx_(other.ctx_), cache_dir_buf_(std::move(other.cache_dir_buf_)) {
        other.ctx_ = nullptr;
    }

    SparseEmbeddingModel& operator=(SparseEmbeddingModel&& other) noexcept {
        if (this != &other) {
            close();
            ctx_ = other.ctx_;
            cache_dir_buf_ = std::move(other.cache_dir_buf_);
            other.ctx_ = nullptr;
        }
        return *this;
    }

    std::vector<SparseVector> embed(const std::vector<std::string>& texts,
                                    int batch_size = 0) {
        if (texts.empty()) return {};

        std::vector<const char*> c_texts;
        c_texts.reserve(texts.size());
        for (const auto& t : texts) c_texts.push_back(t.c_str());

        lembed_sparse_embeddings_t result = {0};
        detail::check_status(lembed_sparse_text_embedding_embed(
            ctx_, c_texts.data(), (int)texts.size(), batch_size, &result));

        std::vector<SparseVector> embeddings;
        try {
            embeddings.resize(result.count);
            for (int i = 0; i < result.count; i++) {
                embeddings[i].indices.assign(
                    result.items[i].indices,
                    result.items[i].indices + result.items[i].length);
                embeddings[i].values.assign(
                    result.items[i].values,
                    result.items[i].values + result.items[i].length);
            }
        } catch (...) {
            lembed_sparse_embeddings_free(&result);
            throw;
        }
        lembed_sparse_embeddings_free(&result);
        return embeddings;
    }

    lembed_model_desc_t info() const {
        lembed_model_desc_t d{};
        if (ctx_) {
            const lembed_model_desc_t* p = lembed_sparse_text_embedding_desc(ctx_);
            if (p) d = *p;
        }
        return d;
    }

    std::string name() const {
        if (!ctx_) return "";
        const char* n = lembed_sparse_text_embedding_model_name(ctx_);
        return n ? std::string(n) : "";
    }

    int max_length() const {
        return ctx_ ? lembed_sparse_text_embedding_max_length(ctx_) : 0;
    }

    lembed_stats_t stats() const {
        lembed_stats_t s{};
        if (ctx_) lembed_sparse_text_embedding_stats(ctx_, &s);
        return s;
    }

    void close() {
        if (ctx_) {
            lembed_sparse_text_embedding_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    lembed_sparse_embedding_ctx_t* ctx_ = nullptr;
    std::string                    cache_dir_buf_;
};

} /* namespace lembed */

#endif /* LIBEMBEDDING_CPP_SPARSE_EMBEDDING_MODEL_HPP */




