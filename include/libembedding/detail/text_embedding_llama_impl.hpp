/*
 * libembedding - text_embedding_llama_impl.hpp
 * llama.cpp backend implementation
 * Include-only header (guarded by LIBEMBEDDING_IMPLEMENTATION)
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_TEXT_EMBEDDING_LLAMA_IMPL_HPP
#define LIBEMBEDDING_TEXT_EMBEDDING_LLAMA_IMPL_HPP

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "This header must be included only when LIBEMBEDDING_IMPLEMENTATION is defined"
#endif

#include "../text_embedding.h"
#include "llama_session_impl.hpp"
#include "../llamacpp_backend.h"
#include "normalize.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include <cstdlib>
#include <cstring>
#ifdef __cplusplus
}
extern "C++" {
#endif
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <filesystem>
#ifdef __cplusplus
}
extern "C" {
#endif

/* =========================================================================
 * llama.cpp backend: creation from GGUF path
 * ========================================================================= */
lembed_status_t lembed_text_embedding_create_from_gguf_path(
        const char* gguf_path,
        const lembed_text_options_t* options,
        lembed_text_embedding_t** out) {
    if (!gguf_path || !gguf_path[0] || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        auto* ctx = new lembed_text_embedding();
        ctx->backend_type = LEMBED_BACKEND_LLAMACPP;
        ctx->pooling = LEMBED_POOLING_MEAN;
        ctx->quantization = LEMBED_QUANTIZATION_NONE;
        ctx->max_length = (options->max_length > 0) ? options->max_length : 0;
        ctx->model_name_str = std::filesystem::path(gguf_path).filename().string();
        ctx->num_threads = (options->num_threads > 0) ? options->num_threads : 4;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = LEMBED_PROVIDER_LLAMACPP;
        ctx->device_id = options->device_id;
        ctx->output_key.clear();

        if (options->auto_workers) {
            ctx->llama.pool.load_from_file_auto(gguf_path,
                                                options->llama_n_ctx,
                                                options->llama_n_gpu_layers,
                                                options->llama_n_batch,
                                                options->llama_verbose != 0);
        } else {
            ctx->llama.session.load_from_file(gguf_path, ctx->num_threads,
                                              options->llama_n_ctx,
                                              options->llama_n_gpu_layers,
                                              options->llama_n_batch,
                                              options->llama_verbose != 0);
        }
        ctx->dim = ctx->llama.session.dimension();
        if (ctx->max_length == 0) ctx->max_length = ctx->llama.session.max_context();

        if (options->cache_size > 0) {
            ctx->cache = new lembed::detail::LRUCache((size_t)options->cache_size, 0);
        }

        lembed__text_update_desc(ctx);
        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_LLAMA;
    }
}

lembed_status_t lembed_text_embedding_create_from_gguf_model(
        const char* repo, const char* filename,
        const lembed_text_options_t* options,
        lembed_text_embedding_t** out) {
    if (!repo || !repo[0] || !filename || !filename[0] || !out)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    char* cached_path = nullptr;
    lembed_status_t s = lembed_ensure_gguf_model(
        repo, filename,
        options ? options->cache_dir : nullptr,
        options ? options->show_download_progress : 0,
        options ? options->offline : 0,
        &cached_path);
    if (s != LEMBED_OK) return s;

    s = lembed_text_embedding_create_from_gguf_path(cached_path, options, out);
    lembed_free_string(cached_path);
    return s;
}

/* =========================================================================
 * llama.cpp backend: embed helpers
 * ========================================================================= */
static inline void lembed__llama_embed_single(
        lembed_text_embedding_t* ctx,
        const char* text,
        float* out_data) {
    auto emb = ctx->llama.session.embed(text);
    std::memcpy(out_data, emb.data(), (size_t)ctx->dim * sizeof(float));
}

static inline void lembed__llama_embed_batch(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        float* result_data) {
    for (int i = 0; i < num_texts; i++) {
        lembed__llama_embed_single(ctx, texts[i], result_data + (size_t)i * ctx->dim);
    }
    lembed::detail::l2_normalize(result_data, num_texts, ctx->dim);
}

static inline void lembed__llama_embed_stream(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        void (*callback)(const float* embedding, int dim, void* userdata),
        void* userdata) {
    for (int i = 0; i < num_texts; i++) {
        float emb[LEMBED_MAX_DIM];
        lembed__llama_embed_single(ctx, texts[i], emb);
        lembed::detail::l2_normalize(emb, 1, ctx->dim);
        callback(emb, ctx->dim, userdata);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_LLAMA_IMPL_HPP */

