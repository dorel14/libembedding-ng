/*
 * libembedding - detail/reranker_llama_impl.hpp
 * llama.cpp backend for document reranking
 * Include-only header (guarded by LIBEMBEDDING_IMPLEMENTATION)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_RERANKER_LLAMA_IMPL_HPP
#define LIBEMBEDDING_RERANKER_LLAMA_IMPL_HPP

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "This header must be included only when LIBEMBEDDING_IMPLEMENTATION is defined"
#endif

#include "../reranker.h"
#include "llama_session_impl.hpp"
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
#include <cmath>
#include <filesystem>
#ifdef __cplusplus
}
extern "C" {
#endif

/* =========================================================================
 * llama.cpp backend: creation from GGUF path/name/URL
 * ========================================================================= */
lembed_status_t lembed_reranker_create_from_gguf_path(
        const char* gguf_path,
        const lembed_reranker_options_t* options,
        lembed_reranker_t** out) {
    if (!gguf_path || !gguf_path[0] || !options || !out)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        char* resolved_path = nullptr;
        lembed_status_t rs = lembed_resolve_gguf_path(
            gguf_path,
            options->cache_dir,
            options->offline,
            &resolved_path);
        if (rs != LEMBED_OK) return rs;

        auto* ctx = new lembed_reranker();
        ctx->backend_type = LEMBED_BACKEND_LLAMACPP;
        ctx->max_length = (options->max_length > 0) ? options->max_length : 512;
        ctx->model_name_str = std::filesystem::path(resolved_path).filename().string();
        ctx->num_threads = (options->num_threads > 0) ? options->num_threads : 4;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = LEMBED_PROVIDER_LLAMACPP;
        ctx->device_id = options->device_id;

        ctx->llama.session.load_from_file(resolved_path, ctx->num_threads,
                                          512, 0, 0, false);
        lembed_free_string(resolved_path);
        if (ctx->max_length == 0) ctx->max_length = ctx->llama.session.max_context();

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = 0;
        ctx->desc.max_length = ctx->max_length;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_LLAMA;
    }
}

lembed_status_t lembed_reranker_create_from_gguf_model(
        const char* repo, const char* filename,
        const lembed_reranker_options_t* options,
        lembed_reranker_t** out) {
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

    s = lembed_reranker_create_from_gguf_path(cached_path, options, out);
    lembed_free_string(cached_path);
    return s;
}

/* =========================================================================
 * llama.cpp backend: rerank helpers
 * ========================================================================= */
static inline float lembed__llama_classify_pair(
        lembed_reranker_t* ctx,
        const char* query,
        const char* document) {
    std::string pair_text = std::string(query) + " [SEP] " + document;
    return ctx->llama.session.classify(pair_text.c_str());
}

static inline void lembed__llama_rerank(
        lembed_reranker_t* ctx,
        const char* query,
        const char* const* documents,
        int num_documents,
        int batch_size,
        std::vector<float>& all_scores) {
    if (batch_size <= 0) batch_size = ctx->batch_size;

    int num_batches = lembed::detail::batch_count(num_documents, batch_size);
    for (int bi = 0; bi < num_batches; bi++) {
        auto range = lembed::detail::get_batch(bi, num_documents, batch_size);
        int bsz = range.end - range.start;

        for (int i = range.start; i < range.end; i++) {
            all_scores[i] = lembed__llama_classify_pair(ctx, query, documents[i]);
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_RERANKER_LLAMA_IMPL_HPP */
