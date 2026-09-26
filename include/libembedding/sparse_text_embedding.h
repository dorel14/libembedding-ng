/*
 * libembedding - sparse_text_embedding.h
 * Sparse text embedding C API (SPLADE, BGE-M3)
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_H
#define LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_H

#include "types.h"
#include "autotuner.h"

#ifdef __cplusplus
extern "C" {
#endif

lembed_sparse_options_t lembed_sparse_options_default(void);


lembed_status_t lembed_sparse_text_embedding_create(
    const lembed_sparse_options_t* options,
    lembed_sparse_embedding_ctx_t** out);

lembed_status_t lembed_sparse_text_embedding_embed(
    lembed_sparse_embedding_ctx_t* ctx,
    const char* const* texts,
    int num_texts,
    int batch_size,
    const lembed_sparse_options_t* sparse_opts,
    lembed_sparse_embeddings_t* result);

/* Introspection
 * NOTE: desc/model_name return memory owned by the context: the pointers are
 * dangling after lembed_sparse_text_embedding_free(). Copy the fields you need. */
const lembed_model_desc_t* lembed_sparse_text_embedding_desc(const lembed_sparse_embedding_ctx_t* ctx);
const char* lembed_sparse_text_embedding_model_name(const lembed_sparse_embedding_ctx_t* ctx);
int lembed_sparse_text_embedding_max_length(const lembed_sparse_embedding_ctx_t* ctx);

/* Runtime statistics */
void lembed_sparse_text_embedding_stats(const lembed_sparse_embedding_ctx_t* ctx, lembed_stats_t* out);
/* Versioned stats (cache fields present for API parity with the dense/rerank
 * contexts). Sparse contexts have no embedding cache, so cache_hits,
 * cache_misses and cache_size are always 0. */
void lembed_sparse_text_embedding_stats_v2(const lembed_sparse_embedding_ctx_t* ctx, lembed_stats_v2_t* out);

void lembed_sparse_text_embedding_free(lembed_sparse_embedding_ctx_t* ctx);

/* Find optimal sparse configuration for a model and corpus.
 * The ONNX session is created once and reused for every candidate: top_k and
 * min_weight are post-processing filters, so reloading the model per candidate
 * would be pure waste. storage_format is not yet honoured by the C embed path
 * and is therefore not benchmarked.
 * texts: array of text samples for benchmarking (must be non-empty)
 * n_texts: number of texts (must be > 0)
 * result: output configuration with optimal top_k, min_weight, storage_format
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_sparse_best_config(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_sparse_tuning_result_t* result);

/* Load from local directory */
lembed_status_t lembed_sparse_text_embedding_create_from_path(
    const char* dir_path,
    const lembed_sparse_options_t* options,
    lembed_sparse_embedding_ctx_t** out);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_IMPL
#define LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_IMPL

#include "model_registry.h"
#include "downloader.h"
#include "detail/model_loader_impl.hpp"
#include "detail/onnx_session_impl.hpp"
#include "detail/tokenizer_impl.hpp"
#include "detail/sparse_postprocess.hpp"
#include "detail/batch.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>


struct lembed_sparse_embedding {
    lembed::detail::OnnxSession session;
    lembed::detail::TokenizerWrapper tokenizer;
    lembed_sparse_model_t model;
    int max_length;

    /* Runtime metadata for introspection */
    std::string model_name_str;
    int num_threads;
    int batch_size;
    lembed_execution_provider_t provider;
    int device_id;
    lembed_model_desc_t desc;
    int top_k;
    float min_weight;

    /* Stats counters */
    uint64_t texts_embedded = 0;
    uint64_t batches_run = 0;
    double   total_latency_ms = 0.0;
    int      stats_calls = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

lembed_status_t lembed_sparse_text_embedding_create(
        const lembed_sparse_options_t* options,
        lembed_sparse_embedding_ctx_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_sparse_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        char* model_dir_cstr = nullptr;
        s = lembed_ensure_sparse_model(options->model, options->cache_dir,
                                       options->show_download_progress,
                                       options->offline, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        auto* ctx = new lembed_sparse_embedding();
        ctx->model = options->model;
        ctx->max_length = (options->max_length > 0) ? options->max_length : info.max_tokens;
        ctx->model_name_str = info.model_name;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->session.load_from_file(onnx_path.c_str(),
                                    options->num_threads,
                                    (int)options->provider);

        std::string tok_path = model_dir + "/tokenizer.json";
        ctx->tokenizer.load_from_file(tok_path, ctx->max_length);

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = 0;
        ctx->desc.max_length = ctx->max_length;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;
        ctx->top_k = options->top_k;
        ctx->min_weight = options->min_weight;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_sparse_text_embedding_create_from_path(
        const char* dir_path,
        const lembed_sparse_options_t* options,
        lembed_sparse_embedding_ctx_t** out) {
    if (!dir_path || !dir_path[0] || !options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        /* Read files from directory */
        std::string onnx_data = lembed::detail::read_file_to_string(
            std::string(dir_path) + "/model.onnx");
        std::string tok_data = lembed::detail::read_file_to_string(
            std::string(dir_path) + "/tokenizer.json");

        /* Read config.json (optional) */
        std::string config_path = std::string(dir_path) + "/config.json";
        bool has_config = lembed::detail::file_exists(config_path);
        int dim = 0, max_len = 0;
        if (has_config) {
            std::string config_data = lembed::detail::read_file_to_string(config_path);
            lembed::detail::parse_config_json(config_data, &dim, &max_len);
        }

        int max_length = (max_len > 0) ? max_len :
                         (options->max_length > 0) ? options->max_length
                                                     : LEMBED_DEFAULT_MAX_LENGTH;

        auto* ctx = new lembed_sparse_embedding();
        ctx->model = LEMBED_SPARSE_MODEL_COUNT; /* sentinel: local model */
        ctx->max_length = max_length;
        ctx->model_name_str = dir_path;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        /* Load ONNX from memory */
        ctx->session.load_from_memory(
            (const void*)onnx_data.data(), onnx_data.size(),
            options->num_threads, (int)options->provider);

        /* Load tokenizer from memory */
        ctx->tokenizer.load_from_blob(tok_data, ctx->max_length);

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = 0;
        ctx->desc.max_length = ctx->max_length;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;
        ctx->top_k = options->top_k;
        ctx->min_weight = options->min_weight;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_IO;
    }
}

lembed_status_t lembed_sparse_text_embedding_embed(
        lembed_sparse_embedding_ctx_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        const lembed_sparse_options_t* sparse_opts,
        lembed_sparse_embeddings_t* result) {
    if (!ctx || !texts || num_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        result->count = num_texts;
        result->items = (lembed_sparse_embedding_t*)calloc(
            num_texts, sizeof(lembed_sparse_embedding_t));
        if (!result->items) return LEMBED_ERROR_OUT_OF_MEMORY;

        auto t_start = std::chrono::high_resolution_clock::now();
        int out_offset = 0;
        int num_batches = lembed::detail::batch_count(num_texts, batch_size);

        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_texts, batch_size);
            int bsz = range.end - range.start;

            std::vector<std::string> batch_texts;
            for (int i = range.start; i < range.end; i++)
                batch_texts.push_back(texts[i]);

            auto enc = ctx->tokenizer.encode_batch(batch_texts);
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

            auto outputs = ctx->session.run(
                ids_flat.data(), mask_flat.data(), type_flat.data(),
                bsz, seq_len);

            /* SPLADE: output is [batch, seq_len, vocab_size] */
            auto& output = outputs[0];
            int vocab_size = (output.shape.size() >= 3)
                ? (int)output.shape[2]
                : (int)output.shape[1];

            auto sparse_results = lembed::detail::splade_postprocess(
                output.data.data(), mask_flat.data(),
                bsz, seq_len, vocab_size);

            /* Resolve filtering parameters.
             * An explicit sparse_opts always wins, including top_k == 0
             * which means "keep every term". The context-level values are
             * only used when the caller passes NULL. */
            const int top_k = sparse_opts ? sparse_opts->top_k : ctx->top_k;
            const float min_weight = sparse_opts ? sparse_opts->min_weight
                                                 : ctx->min_weight;

            /* Copy to C output */
            for (int i = 0; i < bsz; i++) {
                auto& sr = sparse_results[i];

                /* Apply min_weight pruning */
                if (min_weight > 0.0f) {
                    std::vector<int32_t> filt_idx;
                    std::vector<float> filt_val;
                    filt_idx.reserve(sr.indices.size());
                    filt_val.reserve(sr.values.size());
                    for (size_t j = 0; j < sr.indices.size(); j++) {
                        if (sr.values[j] >= min_weight) {
                            filt_idx.push_back(sr.indices[j]);
                            filt_val.push_back(sr.values[j]);
                        }
                    }
                    sr.indices = std::move(filt_idx);
                    sr.values = std::move(filt_val);
                }

                /* Apply top_k selection */
                if (top_k > 0 && (int)sr.indices.size() > top_k) {
                    std::vector<std::pair<float, int32_t>> val_idx;
                    val_idx.reserve(sr.indices.size());
                    for (size_t j = 0; j < sr.indices.size(); j++) {
                        val_idx.emplace_back(sr.values[j], sr.indices[j]);
                    }
                    std::partial_sort(
                        val_idx.begin(),
                        val_idx.begin() + top_k,
                        val_idx.end(),
                        [](const auto& a, const auto& b) {
                            return a.first > b.first;
                        });
                    std::vector<int32_t> filt_idx(top_k);
                    std::vector<float> filt_val(top_k);
                    for (int k = 0; k < top_k; k++) {
                        filt_idx[k] = val_idx[k].second;
                        filt_val[k] = val_idx[k].first;
                    }
                    sr.indices = std::move(filt_idx);
                    sr.values = std::move(filt_val);
                }

                int idx = out_offset + i;
                result->items[idx].length = 0;
                result->items[idx].indices = nullptr;
                result->items[idx].values = nullptr;
                if (sr.indices.empty()) continue;

                result->items[idx].indices = (int32_t*)malloc(
                    sr.indices.size() * sizeof(int32_t));
                result->items[idx].values = (float*)malloc(
                    sr.values.size() * sizeof(float));
                if (result->items[idx].indices && result->items[idx].values) {
                    result->items[idx].length = (int)sr.indices.size();
                    std::memcpy(result->items[idx].indices, sr.indices.data(),
                               sr.indices.size() * sizeof(int32_t));
                    std::memcpy(result->items[idx].values, sr.values.data(),
                               sr.values.size() * sizeof(float));
                } else {
                    /* Allocation failed: never publish a length that does not
                     * match an allocated buffer. */
                    free(result->items[idx].indices);
                    free(result->items[idx].values);
                    result->items[idx].indices = nullptr;
                    result->items[idx].values = nullptr;
                }
            }
            out_offset += bsz;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ctx->texts_embedded += num_texts;
        ctx->batches_run += num_batches;
        ctx->total_latency_ms += elapsed_ms;
        ctx->stats_calls++;

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        lembed_sparse_embeddings_free(result);
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

const lembed_model_desc_t* lembed_sparse_text_embedding_desc(const lembed_sparse_embedding_ctx_t* ctx) {
    return ctx ? &ctx->desc : nullptr;
}

const char* lembed_sparse_text_embedding_model_name(const lembed_sparse_embedding_ctx_t* ctx) {
    return ctx ? ctx->model_name_str.c_str() : nullptr;
}

int lembed_sparse_text_embedding_max_length(const lembed_sparse_embedding_ctx_t* ctx) {
    return ctx ? ctx->max_length : 0;
}

void lembed_sparse_text_embedding_stats(const lembed_sparse_embedding_ctx_t* ctx, lembed_stats_t* out) {
    if (!out) return;
    if (!ctx) { memset(out, 0, sizeof(*out)); return; }
    out->texts_embedded = ctx->texts_embedded;
    out->batches_run = ctx->batches_run;
    out->avg_latency_ms = ctx->stats_calls > 0
        ? ctx->total_latency_ms / (double)ctx->stats_calls
        : 0.0;
}

void lembed_sparse_text_embedding_free(lembed_sparse_embedding_ctx_t* ctx) {
    delete ctx;
}

void lembed_sparse_text_embedding_stats_v2(const lembed_sparse_embedding_ctx_t* ctx, lembed_stats_v2_t* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!ctx) return;
    out->base.texts_embedded = ctx->texts_embedded;
    out->base.batches_run = ctx->batches_run;
    out->base.avg_latency_ms = ctx->stats_calls > 0
        ? ctx->total_latency_ms / (double)ctx->stats_calls
        : 0.0;
}

/* Find optimal sparse configuration by benchmarking variants.
 *
 * The ONNX session is created ONCE and reused for every candidate
 * configuration: top_k / min_weight are post-processing filters, so
 * reloading the model per candidate would be pure waste (24 model
 * loads for the default grid). storage_format is not yet honoured by
 * the C embed path, so it is not benchmarked here.
 */
lembed_status_t lembed_sparse_best_config(
        const char* model_name,
        const char* const* texts,
        int n_texts,
        lembed_sparse_tuning_result_t* result) {
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    memset(result, 0, sizeof(*result));

    /* Resolve model: registry entry or local directory */
    int model_idx = 0;
    bool is_local = false;
    std::string model_path;
    if (lembed::detail::file_exists(model_name)) {
        is_local = true;
        model_path = model_name;
    } else {
        int idx = lembed_find_sparse_model_by_code(model_name);
        if (idx < 0) return LEMBED_ERROR_MODEL_NOT_FOUND;
        model_idx = idx;
    }

    /* Single shared session for all candidates.
     * Uses <thread> rather than autotune_cache.hpp so this public header
     * does not drag in <windows.h> (NOMINMAX / min-max macro clashes). */
    unsigned hw = std::thread::hardware_concurrency();
    int cores = (hw == 0) ? 1 : (int)hw;
    if (cores > 4) cores = 4;

    lembed_sparse_options_t base = lembed_sparse_options_default();
    base.model = static_cast<lembed_sparse_model_t>(model_idx);
    base.num_threads = cores;
    base.batch_size = 32;
    base.show_download_progress = 0;
    base.offline = 1;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    lembed_status_t s;
    if (is_local)
        s = lembed_sparse_text_embedding_create_from_path(model_path.c_str(), &base, &ctx);
    else
        s = lembed_sparse_text_embedding_create(&base, &ctx);
    if (s != LEMBED_OK) return s;

    int best_top_k = 0;
    float best_min_weight = 0.0f;
    double best_throughput = 0.0;
    int configs_tested = 0;

    const int top_k_options[]    = {16, 32, 64, 128};
    const float min_weight_options[] = {0.0f, 0.01f, 0.05f};

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int tk : top_k_options) {
        for (float mw : min_weight_options) {
            lembed_sparse_options_t probe = base;
            probe.top_k = tk;
            probe.min_weight = mw;

            /* Warmup pass (allocator / first-inference cost) */
            lembed_sparse_embeddings_t warm = {0};
            if (lembed_sparse_text_embedding_embed(ctx, texts, n_texts, base.batch_size,
                                                   &probe, &warm) != LEMBED_OK)
                continue;
            lembed_sparse_embeddings_free(&warm);

            auto t0 = std::chrono::high_resolution_clock::now();
            lembed_sparse_embeddings_t emb = {0};
            s = lembed_sparse_text_embedding_embed(ctx, texts, n_texts, base.batch_size,
                                                   &probe, &emb);
            auto t1 = std::chrono::high_resolution_clock::now();

            if (s == LEMBED_OK && emb.count > 0) {
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                double tp = (ms > 0.0) ? (n_texts / (ms / 1000.0)) : 0.0;
                if (tp > best_throughput) {
                    best_throughput = tp;
                    best_top_k = tk;
                    best_min_weight = mw;
                }
                configs_tested++;
            }
            lembed_sparse_embeddings_free(&emb);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    lembed_sparse_text_embedding_free(ctx);

    if (configs_tested == 0 || best_throughput <= 0.0) {
        memset(result, 0, sizeof(*result));
        return LEMBED_ERROR_ONNX_RUNTIME;
    }

    result->top_k = best_top_k;
    result->min_weight = best_min_weight;
    result->storage_format = 0;
    result->threads = cores;
    result->batch_size = base.batch_size;
    result->throughput_docs_sec = best_throughput;
    result->latency_ms = (n_texts > 0) ? (total_ms / n_texts) : 0.0;
    result->memory_mb = 0.0;

    return LEMBED_OK;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_H */




