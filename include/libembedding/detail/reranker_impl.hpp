/*
 * libembedding - detail/reranker_impl.hpp
 * Reranker C++ implementation (ONNX + llama.cpp backends)
 * Include-only header (included from reranker.h when LIBEMBEDDING_IMPLEMENTATION is defined)
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_RERANKER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_RERANKER_IMPL_HPP

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "This header must be included only when LIBEMBEDDING_IMPLEMENTATION is defined"
#endif

#include "../model_registry.h"
#include "../downloader.h"
#include "model_loader_impl.hpp"
#include "onnx_session_impl.hpp"
#include "tokenizer_impl.hpp"
#include "batch.hpp"
#include "embedding_cache_impl.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

/**
 * @brief Reranker context struct (ONNX + llama.cpp backends).
 *
 * Contains ONNX backend (session + tokenizer), llama.cpp backend (session),
 * quantization config, runtime metadata, stats counters, and optional LRU cache
 * for query+document scoring.
 */
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

    lembed_quantization_t quantization;
    int max_length;

    /* Runtime metadata for introspection */
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

    /* Embedding cache (for re-ranked query results) */
    lembed::detail::LRUCache* cache = nullptr;
};

#include "reranker_llama_impl.hpp"

static bool lembed__path_ends_with_gguf(const char* path) {
    if (!path) return false;
    std::string p = path;
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);
    return p.size() > 4 && p.substr(p.size() - 4) == ".gguf";
}

static lembed_status_t lembed__reranker_create_onnx_impl(
        const lembed_reranker_options_t* options,
        lembed_quantization_t quantization,
        int quantization_from_options,
        lembed_reranker_t** out) {
    lembed_reranker_t* ctx = nullptr;
    char* model_dir_cstr = nullptr;
    lembed_status_t s = LEMBED_OK;
    std::string model_dir;

    try {
        lembed_model_info_t info;
        s = lembed_get_reranker_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        s = lembed_ensure_reranker_model(options->model, options->cache_dir,
                                         options->show_download_progress,
                                         options->offline, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        model_dir = model_dir_cstr;
        lembed_free_string(model_dir_cstr);
        model_dir_cstr = nullptr;

        ctx = new lembed_reranker();
        ctx->backend_type = LEMBED_BACKEND_ONNX;
        ctx->quantization = quantization_from_options
            ? quantization
            : (lembed_quantization_t)info.quantization;
        ctx->max_length = (options->max_length > 0) ? options->max_length : info.max_tokens;
        ctx->model_name_str = info.model_name;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        if (options->cache_size > 0) {
            ctx->cache = new lembed::detail::LRUCache((size_t)options->cache_size, 0);
        }

        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->onnx.session.load_from_file(onnx_path.c_str(),
                                         options->num_threads,
                                         (int)options->provider,
                                         (int)ctx->quantization);

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
        ctx->desc_v2.base = ctx->desc;
        ctx->desc_v2.quantization = (int)ctx->quantization;
        ctx->desc_v2.cache_size = ctx->cache ? (int)ctx->cache->capacity() : 0;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (ctx) {
            delete ctx->cache;
            delete ctx;
        }
        if (model_dir_cstr) lembed_free_string(model_dir_cstr);
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
        lembed_status_t s = lembed__reranker_create_onnx_impl(
            options, LEMBED_QUANTIZATION_NONE, 0, out);
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

lembed_status_t lembed_reranker_create_v2(
        const lembed_reranker_options_v2_t* options,
        lembed_reranker_t** out) {
    if (!options) return LEMBED_ERROR_INVALID_ARGUMENT;
    lembed_reranker_options_t base = options->base;
    if (base.backend == LEMBED_BACKEND_LLAMACPP) {
        return lembed_reranker_create(&base, out);
    }
    lembed_status_t s = lembed__reranker_create_onnx_impl(
        &base, (lembed_quantization_t)options->quantization, 1, out);
    if (s != LEMBED_OK || base.backend != LEMBED_BACKEND_AUTO) return s;
    return lembed_reranker_create(&base, out);
}

lembed_status_t lembed_reranker_create_from_path(
        const char* path,
        const lembed_reranker_options_t* options,
        lembed_reranker_t** out) {
    if (!path || !path[0] || !options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    if (lembed__path_ends_with_gguf(path)) {
        return lembed_reranker_create_from_gguf_path(path, options, out);
    }

    lembed_reranker_t* ctx = nullptr;

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

        ctx = new lembed_reranker();
        ctx->backend_type = LEMBED_BACKEND_ONNX;
        ctx->quantization = LEMBED_QUANTIZATION_NONE;
        ctx->max_length = max_length;
        ctx->model_name_str = path;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        if (options->cache_size > 0) {
            ctx->cache = new lembed::detail::LRUCache((size_t)options->cache_size, 0);
        }

        ctx->onnx.session.load_from_memory(
            (const void*)onnx_data.data(), onnx_data.size(),
            options->num_threads, (int)options->provider,
            (int)ctx->quantization);

        ctx->onnx.tokenizer.load_from_blob(tok_data, ctx->max_length);

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = 0;
        ctx->desc.max_length = ctx->max_length;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;
        ctx->desc_v2.base = ctx->desc;
        ctx->desc_v2.quantization = (int)ctx->quantization;
        ctx->desc_v2.cache_size = ctx->cache ? (int)ctx->cache->capacity() : 0;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (ctx) {
            delete ctx->cache;
            delete ctx;
        }
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

        std::vector<float> all_scores(num_documents, 0.0f);
        std::vector<int> miss_indices;
        miss_indices.reserve(num_documents);

        if (ctx->cache) {
            for (int i = 0; i < num_documents; i++) {
                std::string key = std::string(query) + "\n" + documents[i];
                std::vector<float> cached;
                if (ctx->cache->get_copy(key, cached) && cached.size() == 1) {
                    all_scores[i] = cached[0];
                    ctx->cache_hits++;
                } else {
                    miss_indices.push_back(i);
                    ctx->cache_misses++;
                }
            }
        } else {
            miss_indices.reserve(num_documents);
            for (int i = 0; i < num_documents; i++) {
                miss_indices.push_back(i);
            }
            ctx->cache_misses += num_documents;
        }

        if (!miss_indices.empty()) {
            if (ctx->backend_type == LEMBED_BACKEND_LLAMACPP) {
                std::vector<const char*> miss_docs;
                miss_docs.reserve(miss_indices.size());
                for (int idx : miss_indices) miss_docs.push_back(documents[idx]);
                std::vector<float> miss_scores(miss_indices.size());
                lembed__llama_rerank(ctx, query, miss_docs.data(),
                                     (int)miss_docs.size(), batch_size, miss_scores);
                for (size_t i = 0; i < miss_indices.size(); i++) {
                    all_scores[miss_indices[i]] = miss_scores[i];
                    if (ctx->cache) {
                        ctx->cache->put(std::string(query) + "\n" + documents[miss_indices[i]],
                                        &miss_scores[i], 1);
                    }
                }
            } else {
                int num_batches = lembed::detail::batch_count((int)miss_indices.size(), batch_size);
                for (int bi = 0; bi < num_batches; bi++) {
                    auto range = lembed::detail::get_batch(
                        bi, (int)miss_indices.size(), batch_size);
                    int bsz = range.end - range.start;

                    std::vector<std::string> pair_texts;
                    pair_texts.reserve(bsz);
                    for (int i = range.start; i < range.end; i++) {
                        pair_texts.push_back(std::string(query) + " [SEP] " +
                                             documents[miss_indices[i]]);
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
                        int doc_idx = miss_indices[range.start + i];
                        all_scores[doc_idx] = output.data[i * ((int)output.shape.back())];
                        if (ctx->cache) {
                            ctx->cache->put(std::string(query) + "\n" + documents[doc_idx],
                                            &all_scores[doc_idx], 1);
                        }
                    }
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

const lembed_model_desc_v2_t* lembed_reranker_desc_v2(const lembed_reranker_t* ctx) {
    return ctx ? &ctx->desc_v2 : nullptr;
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

void lembed_reranker_stats_v2(const lembed_reranker_t* ctx, lembed_stats_v2_t* out) {
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

void lembed_reranker_free(lembed_reranker_t* ctx) {
    if (!ctx) return;
    if (ctx->cache) {
        delete ctx->cache;
        ctx->cache = nullptr;
    }
    delete ctx;
}

#endif /* LIBEMBEDDING_DETAIL_RERANKER_IMPL_HPP */
