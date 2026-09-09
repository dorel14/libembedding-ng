/*
 * libembedding - cpp/image_embedding_model.hpp
 * C++ wrapper for image embedding.
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CPP_IMAGE_EMBEDDING_MODEL_HPP
#define LIBEMBEDDING_CPP_IMAGE_EMBEDDING_MODEL_HPP

#include <libembedding/image_embedding.h>
#include <libembedding/cpp/provider.hpp>
#include <libembedding/detail/status_helper.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lembed {

struct ImageEmbeddingOptions {
    std::string                  model_path = "openai/clip-vit-base-patch32";
    int                          threads = 0;
    int                          batch_size = 32;
    bool                         offline = false;
    std::string                  cache_dir;
    std::string                  provider = "cpu";
    int                          device_id = 0;
    bool                         show_download_progress = true;
    int                          dim = 0;
};

class ImageEmbeddingModel {
public:
    ImageEmbeddingModel(const std::string& model, const ImageEmbeddingOptions& opts = {}) {
        lembed_image_options_t c_opts = lembed_image_options_default();

        c_opts.provider = parse_provider(opts.provider);
        c_opts.device_id = opts.device_id;
        c_opts.num_threads = opts.threads;
        c_opts.batch_size = opts.batch_size;
        c_opts.offline = opts.offline ? 1 : 0;
        c_opts.show_download_progress = opts.show_download_progress ? 1 : 0;
        if (!opts.cache_dir.empty()) {
            cache_dir_buf_ = opts.cache_dir;
            c_opts.cache_dir = cache_dir_buf_.c_str();
        }
        c_opts.dim = opts.dim;

        if (std::filesystem::exists(model)) {
            detail::check_status(lembed_image_embedding_create_from_path(
                model.c_str(), &c_opts, &ctx_));
        } else {
            c_opts.model = resolve_image_model(model);
            detail::check_status(lembed_image_embedding_create(&c_opts, &ctx_));
        }

        dim_ = lembed_image_embedding_dim(ctx_);
    }

    ~ImageEmbeddingModel() { close(); }

    ImageEmbeddingModel(const ImageEmbeddingModel&) = delete;
    ImageEmbeddingModel& operator=(const ImageEmbeddingModel&) = delete;

    ImageEmbeddingModel(ImageEmbeddingModel&& other) noexcept
        : ctx_(other.ctx_), dim_(other.dim_), cache_dir_buf_(std::move(other.cache_dir_buf_)) {
        other.ctx_ = nullptr;
    }

    ImageEmbeddingModel& operator=(ImageEmbeddingModel&& other) noexcept {
        if (this != &other) {
            close();
            ctx_ = other.ctx_;
            dim_ = other.dim_;
            cache_dir_buf_ = std::move(other.cache_dir_buf_);
            other.ctx_ = nullptr;
        }
        return *this;
    }

    std::vector<std::vector<float>> embed_files(const std::vector<std::string>& paths,
                                                int batch_size = 0) {
        if (paths.empty()) return {};

        std::vector<const char*> c_paths;
        c_paths.reserve(paths.size());
        for (const auto& p : paths) c_paths.push_back(p.c_str());

        lembed_embeddings_t result = {0};
        detail::check_status(lembed_image_embedding_embed_files(
            ctx_, c_paths.data(), (int)paths.size(), batch_size, &result));

        return to_vector(result);
    }

    std::vector<std::vector<float>> embed_bytes(const std::vector<std::vector<uint8_t>>& images,
                                                int batch_size = 0) {
        if (images.empty()) return {};

        std::vector<const unsigned char*> c_data;
        std::vector<int> c_sizes;
        c_data.reserve(images.size());
        c_sizes.reserve(images.size());
        for (const auto& img : images) {
            c_data.push_back(img.data());
            c_sizes.push_back((int)img.size());
        }

        lembed_embeddings_t result = {0};
        detail::check_status(lembed_image_embedding_embed_bytes(
            ctx_, c_data.data(), c_sizes.data(), (int)images.size(), batch_size, &result));

        return to_vector(result);
    }

    int dimension() const {
        return ctx_ ? lembed_image_embedding_dim(ctx_) : 0;
    }

    lembed_model_desc_t info() const {
        lembed_model_desc_t d{};
        if (ctx_) {
            const lembed_model_desc_t* p = lembed_image_embedding_desc(ctx_);
            if (p) d = *p;
        }
        return d;
    }

    std::string name() const {
        if (!ctx_) return "";
        const char* n = lembed_image_embedding_model_name(ctx_);
        return n ? std::string(n) : "";
    }

    lembed_stats_t stats() const {
        lembed_stats_t s{};
        if (ctx_) lembed_image_embedding_stats(ctx_, &s);
        return s;
    }

    void close() {
        if (ctx_) {
            lembed_image_embedding_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    lembed_image_embedding_t* ctx_ = nullptr;
    int                       dim_ = 0;
    std::string               cache_dir_buf_;

    static lembed_image_model_t resolve_image_model(const std::string& name) {
        const lembed_model_info_t* models = nullptr;
        int count = 0;
        if (lembed_list_image_models(&models, &count) != LEMBED_OK) {
            throw std::invalid_argument("Unknown image model: " + name);
        }
        for (int i = 0; i < count; i++) {
            std::string mn = models[i].model_name ? models[i].model_name : "";
            std::string mc = models[i].model_code ? models[i].model_code : "";
            if (name == mn || name == mc) return (lembed_image_model_t)i;
        }
        throw std::invalid_argument("Unknown image model: " + name);
    }

    static std::vector<std::vector<float>> to_vector(const lembed_embeddings_t& result) {
        std::vector<std::vector<float>> embeddings;
        try {
            embeddings.resize(result.num_embeddings);
            for (int i = 0; i < result.num_embeddings; i++) {
                embeddings[i].assign(
                    result.data + (size_t)i * result.dim,
                    result.data + (size_t)(i + 1) * result.dim);
            }
        } catch (...) {
            lembed_embeddings_free(const_cast<lembed_embeddings_t*>(&result));
            throw;
        }
        lembed_embeddings_free(const_cast<lembed_embeddings_t*>(&result));
        return embeddings;
    }
};

} /* namespace lembed */

#endif /* LIBEMBEDDING_CPP_IMAGE_EMBEDDING_MODEL_HPP */




