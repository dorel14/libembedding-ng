/*
 * libembedding - detail/text_embedding_impl.hpp
 * Text embedding implementation (dispatcher + shared code)
 * Include-only header (guarded by LIBEMBEDDING_IMPLEMENTATION)
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_TEXT_EMBEDDING_IMPL_HPP
#define LIBEMBEDDING_TEXT_EMBEDDING_IMPL_HPP

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "This header must be included only when LIBEMBEDDING_IMPLEMENTATION is defined"
#endif

#include "../text_embedding.h"
#include "../model_registry.h"
#include "../downloader.h"
#include "onnx_session_impl.hpp"
#include "tokenizer_impl.hpp"
#include "pooling.hpp"
#include "normalize.hpp"
#include "batch.hpp"
#include "llama_session_impl.hpp"
#include "llama_scheduler_impl.hpp"
#include "worker_autotune_impl.hpp"
#include "embedding_cache_impl.hpp"

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
 * Internal struct definition - text embedding context (all backends)
 * ========================================================================= */
/**
 * @brief Main text embedding context struct.
 *
 * Contains ONNX backend (session + tokenizer), llama.cpp backend (session + pool),
 * pooling/quantization config, runtime metadata, stats counters, and optional LRU cache.
 */
struct lembed_text_embedding {
    lembed_backend_t backend_type;

    /* ONNX backend */
    struct OnnxBackend {
        lembed::detail::OnnxSession session;
        lembed::detail::TokenizerWrapper tokenizer;
    } onnx;

    /* llama.cpp backend */
    struct LlamaBackend {
        lembed::detail::LlamaSession session;
        lembed::detail::LlamaSessionPool pool;
    } llama;

    lembed_pooling_t pooling;
    lembed_quantization_t quantization;
    int dim;
    int max_length;
    std::string output_key;

    /* Runtime metadata */
    std::string model_name_str;
    int num_threads;
    int batch_size;
    lembed_execution_provider_t provider;
    int device_id;
    lembed_model_desc_t desc;
    lembed_model_desc_v2_t desc_v2;

    /* Stats counters */
    uint64_t texts_embedded = 0;
    uint64_t batches_run = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    double   total_latency_ms = 0.0;
    int      stats_calls = 0;

    /* Reusable per-batch buffers (ONNX only) */
    std::vector<int64_t> ids_buf_;
    std::vector<int64_t> mask_buf_;
    std::vector<int64_t> type_buf_;
    std::vector<float> pooled_buf_;

    /* Configuration */
    lembed_batch_strategy_t batch_strategy = LEMBED_BATCH_LENGTH_BUCKET;

    /* Embedding cache */
    lembed::detail::LRUCache* cache = nullptr;
};

/* =========================================================================
 * Helper functions
 * ========================================================================= */
static void lembed__text_update_desc(lembed_text_embedding_t* ctx) {
    /** @brief Update model descriptor fields from context members. */
    if (!ctx) return;
    ctx->desc.name = ctx->model_name_str.c_str();
    ctx->desc.dimension = ctx->dim;
    ctx->desc.max_length = ctx->max_length;
    ctx->desc.pooling = ctx->pooling;
    ctx->desc.num_threads = ctx->num_threads;
    ctx->desc.batch_size = ctx->batch_size;
    ctx->desc.provider = ctx->provider;
    ctx->desc.device_id = ctx->device_id;
    ctx->desc_v2.base = ctx->desc;
    ctx->desc_v2.quantization = (int)ctx->quantization;
    ctx->desc_v2.cache_size = ctx->cache ? (int)ctx->cache->capacity() : 0;
}

/* =========================================================================
 * Backend implementations (require struct definition)
 * ========================================================================= */
#include "text_embedding_onnx_impl.hpp"
#include "text_embedding_llama_impl.hpp"

/* =========================================================================
 * Main embed function (dispatches to backend)
 * ========================================================================= */
lembed_status_t lembed_text_embedding_embed(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        lembed_embeddings_t* result) {
    if (!ctx || !texts || num_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (ctx->quantization == LEMBED_QUANTIZATION_DYNAMIC) {
        if (batch_size > 0 && batch_size < num_texts) {
            lembed::detail::set_error(
                "Dynamic quantization cannot be used with batching smaller than total texts");
            return LEMBED_ERROR_BATCH_SIZE;
        }
        batch_size = num_texts;
    }

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        int dim = ctx->dim;
        result->dim = dim;
        result->num_embeddings = num_texts;
        result->data = (float*)malloc((size_t)num_texts * dim * sizeof(float));
        if (!result->data) return LEMBED_ERROR_OUT_OF_MEMORY;

        auto t_start = std::chrono::high_resolution_clock::now();

        lembed_status_t s;
        if (ctx->backend_type == LEMBED_BACKEND_LLAMACPP) {
            lembed__llama_embed_batch(ctx, texts, num_texts, result->data);
            ctx->texts_embedded += num_texts;
            ctx->batches_run++;
            s = LEMBED_OK;
        } else {
            s = lembed__onnx_embed(ctx, texts, num_texts, batch_size, result->data);
            if (s == LEMBED_OK) {
                ctx->texts_embedded += num_texts;
                ctx->batches_run += lembed::detail::batch_count(num_texts, batch_size);
            }
        }

        if (s == LEMBED_OK) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            ctx->total_latency_ms += elapsed_ms;
            ctx->stats_calls++;
        }
        return s;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (result->data) { free(result->data); result->data = NULL; }
        return (ctx->backend_type == LEMBED_BACKEND_LLAMACPP)
            ? LEMBED_ERROR_LLAMA : LEMBED_ERROR_ONNX_RUNTIME;
    }
}

/* =========================================================================
 * Embed stream (dispatches to backend)
 * ========================================================================= */
lembed_status_t lembed_text_embedding_embed_stream(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        void (*callback)(const float* embedding, int dim, void* userdata),
        void* userdata) {
    if (!ctx || !texts || num_texts <= 0 || !callback)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        if (ctx->backend_type == LEMBED_BACKEND_LLAMACPP) {
            lembed__llama_embed_stream(ctx, texts, num_texts, callback, userdata);
            ctx->texts_embedded += num_texts;
            ctx->batches_run++;
        } else {
            lembed__onnx_embed_stream(ctx, texts, num_texts, batch_size, callback, userdata);
            ctx->texts_embedded += num_texts;
            ctx->batches_run += lembed::detail::batch_count(num_texts, batch_size);
        }
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return (ctx->backend_type == LEMBED_BACKEND_LLAMACPP)
            ? LEMBED_ERROR_LLAMA : LEMBED_ERROR_ONNX_RUNTIME;
    }
}

/* =========================================================================
 * Create from local path (directory or .gguf file)
 * ========================================================================= */
lembed_status_t lembed_text_embedding_create_from_path(
        const char* dir_path,
        const lembed_text_options_t* options,
        lembed_text_embedding_t** out) {
    if (!dir_path || !dir_path[0] || !options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    std::string path_str(dir_path);
    bool is_gguf = (path_str.size() > 5 && path_str.compare(path_str.size() - 5, 5, ".gguf") == 0);

    if (is_gguf) {
        return lembed_text_embedding_create_from_gguf_path(dir_path, options, out);
    }

    /* ONNX backend from local directory */
    try {
        std::string onnx_path = std::string(dir_path) + "/model.onnx";
        std::string onnx_data = lembed::detail::read_file_to_string(onnx_path);
        std::string tok_path = std::string(dir_path) + "/tokenizer.json";
        std::string tok_data = lembed::detail::read_file_to_string(tok_path);

        std::string config_path = std::string(dir_path) + "/config.json";
        bool has_config = lembed::detail::file_exists(config_path);
        std::string config_data;
        if (has_config) config_data = lembed::detail::read_file_to_string(config_path);

        lembed_user_defined_model_t udm;
        memset(&udm, 0, sizeof(udm));
        udm.onnx_data = (const unsigned char*)onnx_data.data();
        udm.onnx_data_size = onnx_data.size();
        udm.tokenizer_json = (const unsigned char*)tok_data.data();
        udm.tokenizer_json_size = tok_data.size();

        int dim = 0, max_length = 0;
        lembed_pooling_t pooling = LEMBED_POOLING_MEAN;

        if (has_config) {
            lembed::detail::parse_config_json(config_data, &dim, &max_length);
            pooling = lembed::detail::infer_pooling_from_path(dir_path);
        }

        if (dim == 0) dim = options->dim;
        if (max_length == 0) max_length = options->max_length;
        if (options->pooling == LEMBED_POOLING_CLS) pooling = LEMBED_POOLING_CLS;

        if (dim == 0) {
            lembed::detail::set_error("Cannot determine model dimension");
            return LEMBED_ERROR_INVALID_ARGUMENT;
        }

        udm.dim = dim;
        udm.pooling = pooling;
        udm.max_length = max_length;

        lembed_text_embedding_t* ctx = nullptr;
        lembed_status_t s = lembed_text_embedding_create_custom(
            &udm, options->provider, options->num_threads, &ctx);
        if (s != LEMBED_OK) return s;

        ctx->model_name_str = std::filesystem::path(dir_path).filename().string();
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size : LEMBED_DEFAULT_BATCH_SIZE;
        lembed__text_update_desc(ctx);
        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_IO;
    }
}

/* =========================================================================
 * Introspection
 * ========================================================================= */
int lembed_text_embedding_dim(const lembed_text_embedding_t* ctx) {
    return ctx ? ctx->dim : 0;
}

const lembed_model_desc_t* lembed_text_embedding_desc(const lembed_text_embedding_t* ctx) {
    return ctx ? &ctx->desc : nullptr;
}

const lembed_model_desc_v2_t* lembed_text_embedding_desc_v2(const lembed_text_embedding_t* ctx) {
    return ctx ? &ctx->desc_v2 : nullptr;
}

const char* lembed_text_embedding_model_name(const lembed_text_embedding_t* ctx) {
    return ctx ? ctx->model_name_str.c_str() : nullptr;
}

int lembed_text_embedding_max_length(const lembed_text_embedding_t* ctx) {
    return ctx ? ctx->max_length : 0;
}

void lembed_text_embedding_stats(const lembed_text_embedding_t* ctx, lembed_stats_t* out) {
    if (!out) return;
    if (!ctx) { memset(out, 0, sizeof(*out)); return; }
    out->texts_embedded = ctx->texts_embedded;
    out->batches_run = ctx->batches_run;
    out->avg_latency_ms = ctx->stats_calls > 0
        ? ctx->total_latency_ms / (double)ctx->stats_calls
        : 0.0;
}

void lembed_text_embedding_stats_v2(const lembed_text_embedding_t* ctx, lembed_stats_v2_t* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!ctx) return;
    out->base.texts_embedded = ctx->texts_embedded;
    out->base.batches_run = ctx->batches_run;
    out->base.avg_latency_ms = ctx->stats_calls > 0
        ? ctx->total_latency_ms / (double)ctx->stats_calls
        : 0.0;
    out->cache_hits = ctx->cache_hits;
    out->cache_misses = ctx->cache_misses;
    out->cache_size = ctx->cache ? ctx->cache->capacity() : 0;
}

void lembed_text_embedding_free(lembed_text_embedding_t* ctx) {
    delete ctx;
}

lembed_worker_config_t lembed_detect_optimal_workers(void) {
    return lembed::detail::detect_optimal_workers();
}

int lembed_recommended_workers_for_model(const char* model_path) {
    return lembed::detail::recommended_workers_for_model(model_path);
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_IMPL_HPP */
