/*
 * libembedding - cpp/reranker_model.hpp
 * C++ wrapper for document reranking.
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CPP_RERANKER_MODEL_HPP
#define LIBEMBEDDING_CPP_RERANKER_MODEL_HPP

#include <libembedding/reranker.h>
#include <libembedding/cpp/provider.hpp>
#include <libembedding/detail/status_helper.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lembed {

struct RerankerOptions {
    std::string                  model_path = "BAAI/bge-reranker-base";
    int                          threads = 0;
    int                          batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    bool                         offline = false;
    std::string                  cache_dir;
    std::string                  provider = "cpu";
    int                          device_id = 0;
    int                          max_length = 0;
    bool                         show_download_progress = true;
};

struct RerankResult {
    int   index;
    float score;
};

class RerankerModel {
public:
    RerankerModel(const std::string& model, const RerankerOptions& opts = {}) {
        lembed_reranker_options_t c_opts = lembed_reranker_options_default();

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

        // Resolve model by code/name, or load from local path
        int idx = -1;
        const lembed_model_info_t* models = nullptr;
        int count = 0;
        if (lembed_list_reranker_models(&models, &count) == LEMBED_OK) {
            for (int i = 0; i < count; i++) {
                std::string mn = models[i].model_name ? models[i].model_name : "";
                std::string mc = models[i].model_code ? models[i].model_code : "";
                if (model == mn || model == mc) { idx = i; break; }
            }
        }

        if (idx >= 0) {
            c_opts.model = (lembed_reranker_model_t)idx;
            detail::check_status(lembed_reranker_create(&c_opts, &ctx_));
        } else if (std::filesystem::exists(model)) {
            detail::check_status(lembed_reranker_create_from_path(
                model.c_str(), &c_opts, &ctx_));
        } else {
            throw std::invalid_argument("Unknown model or path: " + model);
        }
    }

    ~RerankerModel() { close(); }

    RerankerModel(const RerankerModel&) = delete;
    RerankerModel& operator=(const RerankerModel&) = delete;

    RerankerModel(RerankerModel&& other) noexcept
        : ctx_(other.ctx_), cache_dir_buf_(std::move(other.cache_dir_buf_)) {
        other.ctx_ = nullptr;
    }

    RerankerModel& operator=(RerankerModel&& other) noexcept {
        if (this != &other) {
            close();
            ctx_ = other.ctx_;
            cache_dir_buf_ = std::move(other.cache_dir_buf_);
            other.ctx_ = nullptr;
        }
        return *this;
    }

    std::vector<RerankResult> rerank(const std::string& query,
                                     const std::vector<std::string>& documents,
                                     int batch_size = 0) {
        if (documents.empty()) return {};

        std::vector<const char*> c_docs;
        c_docs.reserve(documents.size());
        for (const auto& d : documents) c_docs.push_back(d.c_str());

        lembed_rerank_results_t result = {0};
        detail::check_status(lembed_reranker_rerank(
            ctx_, query.c_str(), c_docs.data(), (int)documents.size(),
            batch_size, &result));

        std::vector<RerankResult> reranked;
        try {
            reranked.resize(result.count);
            for (int i = 0; i < result.count; i++) {
                reranked[i].index = result.items[i].index;
                reranked[i].score = result.items[i].score;
            }
        } catch (...) {
            lembed_rerank_results_free(&result);
            throw;
        }
        lembed_rerank_results_free(&result);
        return reranked;
    }

    lembed_model_desc_t info() const {
        lembed_model_desc_t d{};
        if (ctx_) {
            const lembed_model_desc_t* p = lembed_reranker_desc(ctx_);
            if (p) d = *p;
        }
        return d;
    }

    std::string name() const {
        if (!ctx_) return "";
        const char* n = lembed_reranker_model_name(ctx_);
        return n ? std::string(n) : "";
    }

    int max_length() const {
        return ctx_ ? lembed_reranker_max_length(ctx_) : 0;
    }

    lembed_stats_t stats() const {
        lembed_stats_t s{};
        if (ctx_) lembed_reranker_stats(ctx_, &s);
        return s;
    }

    void close() {
        if (ctx_) {
            lembed_reranker_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    lembed_reranker_t*    ctx_ = nullptr;
    std::string           cache_dir_buf_;
};

} /* namespace lembed */

#endif /* LIBEMBEDDING_CPP_RERANKER_MODEL_HPP */




