/*
 * libembedding - reranker.h
 * Cross-encoder reranker C API (ONNX + llama.cpp backends)
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_RERANKER_H
#define LIBEMBEDDING_RERANKER_H

#include "types.h"
#include "llamacpp_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

lembed_reranker_options_t lembed_reranker_options_default(void);

/* Auto-routing create: ONNX registry for HuggingFace IDs, GGUF for .gguf paths */
lembed_status_t lembed_reranker_create(
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Load from local directory (auto-detects ONNX vs GGUF) or direct .gguf path */
lembed_status_t lembed_reranker_create_from_path(
    const char* path,
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Explicit GGUF loading */
lembed_status_t lembed_reranker_create_from_gguf_path(
    const char* gguf_path,
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Load GGUF model from HuggingFace (llama.cpp backend) */
lembed_status_t lembed_reranker_create_from_gguf_model(
    const char* repo,
    const char* filename,
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Rerank documents against a query.
 * Results are sorted by score descending. */
lembed_status_t lembed_reranker_rerank(
    lembed_reranker_t* ctx,
    const char* query,
    const char* const* documents,
    int num_documents,
    int batch_size,
    lembed_rerank_results_t* result);

/* Introspection */
const lembed_model_desc_t* lembed_reranker_desc(const lembed_reranker_t* ctx);
const char* lembed_reranker_model_name(const lembed_reranker_t* ctx);
int lembed_reranker_max_length(const lembed_reranker_t* ctx);

/* Runtime statistics */
void lembed_reranker_stats(const lembed_reranker_t* ctx, lembed_stats_t* out);

void lembed_reranker_free(lembed_reranker_t* ctx);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_RERANKER_IMPL
#define LIBEMBEDDING_RERANKER_IMPL

#include "model_registry.h"
#include "downloader.h"
#include "detail/model_loader_impl.hpp"
#include "detail/onnx_session_impl.hpp"
#include "detail/tokenizer_impl.hpp"
#include "detail/batch.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

struct lembed_reranker {
    lembed_backend_t backend_type;

    /* ONNX backend */
    struct OnnxBackend {
        lembed::detail::OnnxSession session;
        lembed::detail::TokenizerWrapper tokenizer;
    } onnx;

    /* llama.cpp backend */
    struct LlamaBackend {
        lembed::detail::LlamaSession session;
    } llama;

    int max_length;

    /* Runtime metadata for introspection */
    std::string model_name_str;
    int num_threads;
    int batch_size;
    lembed_execution_provider_t provider;
    int device_id;
    lembed_model_desc_t desc;

    /* Stats counters */
    uint64_t texts_embedded = 0;
    uint64_t batches_run = 0;
    double   total_latency_ms = 0.0;
    int      stats_calls = 0;
};

#include "detail/reranker_llama_impl.hpp"

#ifdef __cplusplus
extern "C" {
#endif

static bool lembed__path_ends_with_gguf(const char* path) {
    if (!path) return false;
    std::string p = path;
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);
    return p.size() > 4 && p.substr(p.size() - 4) == ".gguf";
}

static lembed_status_t lembed__reranker_create_onnx_impl(
        const lembed_reranker_options_t* options,
        lembed_reranker_t** out) {
    try {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_reranker_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        char* model_dir_cstr = nullptr;
        s = lembed_ensure_reranker_model(options->model, options->cache_dir,
                                         options->show_download_progress,
                                         options->offline, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        auto* ctx = new lembed_reranker();
        ctx->backend_type = LEMBED_BACKEND_ONNX;
        ctx->max_length = (options->max_length > 0) ? options->max_length : info.max_tokens;
        ctx->model_name_str = info.model_name;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->onnx.session.load_from_file(onnx_path.c_str(),
                                    options->num_threads,
                                    (int)options->provider);

        std::string tok_path = model_dir + "/tokenizer.json";
        ctx->onnx.tokenizer.load_from_file(tok_path, ctx->max_length);

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
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_reranker_create(
        const lembed_reranker_options_t* options,
        lembed_reranker_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    lembed_backend_t backend = (lembed_backend_t)options->backend;

    /* Validate backend enum range */
    if (backend < LEMBED_BACKEND_ONNX || backend > LEMBED_BACKEND_AUTO) {
        lembed::detail::set_error("Invalid backend value in reranker options");
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }

    /* If backend is explicitly ONNX, or AUTO and model is an ONNX enum, use ONNX path */
    if (backend == LEMBED_BACKEND_ONNX || backend == LEMBED_BACKEND_AUTO) {
        lembed_status_t s = lembed__reranker_create_onnx_impl(options, out);
        if (s == LEMBED_OK || backend == LEMBED_BACKEND_ONNX) return s;
        /* If AUTO and ONNX failed with MODEL_NOT_FOUND, fall through to GGUF */
        if (s != LEMBED_ERROR_MODEL_NOT_FOUND) return s;
    }

    /* llama.cpp backend (explicit or AUTO fallback) */
    if (backend == LEMBED_BACKEND_LLAMACPP || backend == LEMBED_BACKEND_AUTO) {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_reranker_model_info(options->model, &info);
        if (s == LEMBED_OK) {
            /* Try to find a matching GGUF model by name or code */
            const lembed_gguf_model_info_t* gguf_info = lembed_find_gguf_model(info.model_name);
            if (!gguf_info) gguf_info = lembed_find_gguf_model(info.model_code);
            if (gguf_info && gguf_info->gguf_url && gguf_info->gguf_url[0]) {
                return lembed_reranker_create_from_gguf_path(gguf_info->gguf_url, options, out);
            }
        }
        if (backend == LEMBED_BACKEND_LLAMACPP) {
            lembed::detail::set_error("No GGUF model found for reranker and llama.cpp backend requested");
            return LEMBED_ERROR_MODEL_NOT_FOUND;
        }
    }

    return LEMBED_ERROR_MODEL_NOT_FOUND;
}

lembed_status_t lembed_reranker_create_from_path(
        const char* path,
        const lembed_reranker_options_t* options,
        lembed_reranker_t** out) {
    if (!path || !path[0] || !options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    if (lembed__path_ends_with_gguf(path)) {
        return lembed_reranker_create_from_gguf_path(path, options, out);
    }

    try {
        std::string onnx_data = lembed::detail::read_file_to_string(
            std::string(path) + "/model.onnx");
        std::string tok_data = lembed::detail::read_file_to_string(
            std::string(path) + "/tokenizer.json");

        std::string config_path = std::string(path) + "/config.json";
        int dim = 0, max_len = 0;
        if (lembed::detail::file_exists(config_path)) {
            std::string config_data = lembed::detail::read_file_to_string(config_path);
            lembed::detail::parse_config_json(config_data, &dim, &max_len);
        }

        int max_length = (max_len > 0) ? max_len :
                         (options->max_length > 0) ? options->max_length
                                                     : LEMBED_DEFAULT_MAX_LENGTH;

        auto* ctx = new lembed_reranker();
        ctx->backend_type = LEMBED_BACKEND_ONNX;
        ctx->max_length = max_length;
        ctx->model_name_str = path;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        ctx->onnx.session.load_from_memory(
            (const void*)onnx_data.data(), onnx_data.size(),
            options->num_threads, (int)options->provider);

        ctx->onnx.tokenizer.load_from_blob(tok_data, ctx->max_length);

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
        return LEMBED_ERROR_IO;
    }
}

lembed_status_t lembed_reranker_rerank(
        lembed_reranker_t* ctx,
        const char* query,
        const char* const* documents,
        int num_documents,
        int batch_size,
        lembed_rerank_results_t* result) {
    if (!ctx || !query || !documents || num_documents <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<float> all_scores(num_documents);

        if (ctx->backend_type == LEMBED_BACKEND_LLAMACPP) {
            lembed__llama_rerank(ctx, query, documents, num_documents,
                                 batch_size, all_scores);
        } else {
            int num_batches = lembed::detail::batch_count(num_documents, batch_size);
            for (int bi = 0; bi < num_batches; bi++) {
                auto range = lembed::detail::get_batch(bi, num_documents, batch_size);
                int bsz = range.end - range.start;

                std::vector<std::string> pair_texts;
                pair_texts.reserve(bsz);
                for (int i = range.start; i < range.end; i++) {
                    pair_texts.push_back(std::string(query) + " [SEP] " + documents[i]);
                }

                auto enc = ctx->onnx.tokenizer.encode_batch(pair_texts);
                int seq_len = enc.seq_length;

                std::vector<int64_t> ids_flat(bsz * seq_len);
                std::vector<int64_t> mask_flat(bsz * seq_len);
                std::vector<int64_t> type_flat(bsz * seq_len, 0);

                for (int i = 0; i < bsz; i++) {
                    for (int j = 0; j < seq_len; j++) {
                        ids_flat[i * seq_len + j] = enc.input_ids[i][j];
                        mask_flat[i * seq_len + j] = enc.attention_mask[i][j];
                    }
                }

                auto outputs = ctx->onnx.session.run(
                    ids_flat.data(), mask_flat.data(), type_flat.data(),
                    bsz, seq_len);

                auto& output = outputs[0];
                for (int i = 0; i < bsz; i++) {
                    all_scores[range.start + i] = output.data[i * ((int)output.shape.back())];
                }
            }
        }

        result->count = num_documents;
        result->items = (lembed_rerank_result_t*)malloc(
            num_documents * sizeof(lembed_rerank_result_t));
        if (!result->items) return LEMBED_ERROR_OUT_OF_MEMORY;

        for (int i = 0; i < num_documents; i++) {
            result->items[i].index = i;
            result->items[i].score = all_scores[i];
        }

        std::sort(result->items, result->items + num_documents,
                  [](const lembed_rerank_result_t& a, const lembed_rerank_result_t& b) {
                      return a.score > b.score;
                  });

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ctx->texts_embedded += num_documents;
        ctx->batches_run += lembed::detail::batch_count(num_documents, batch_size);
        ctx->total_latency_ms += elapsed_ms;
        ctx->stats_calls++;

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

const lembed_model_desc_t* lembed_reranker_desc(const lembed_reranker_t* ctx) {
    return ctx ? &ctx->desc : nullptr;
}

const char* lembed_reranker_model_name(const lembed_reranker_t* ctx) {
    return ctx ? ctx->model_name_str.c_str() : nullptr;
}

int lembed_reranker_max_length(const lembed_reranker_t* ctx) {
    return ctx ? ctx->max_length : 0;
}

void lembed_reranker_stats(const lembed_reranker_t* ctx, lembed_stats_t* out) {
    if (!out) return;
    if (!ctx) { memset(out, 0, sizeof(*out)); return; }
    out->texts_embedded = ctx->texts_embedded;
    out->batches_run = ctx->batches_run;
    out->avg_latency_ms = ctx->stats_calls > 0
        ? ctx->total_latency_ms / (double)ctx->stats_calls
        : 0.0;
}

void lembed_reranker_free(lembed_reranker_t* ctx) {
    delete ctx;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_RERANKER_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_RERANKER_H */





