/*
 * libembedding - detail/autotune_bench_sparse.hpp
 * Sparse embedding auto-tuner
 *
 * Auteur: David Orel
 * Version: 1.9.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP

#include "libembedding/autotuner.h"
#include "libembedding/sparse_text_embedding.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Benchmark a single sparse configuration */
inline lembed_sparse_tuning_result_t bench_sparse_config(
    const char* model_name,
    int top_k,
    float min_weight,
    int storage_format,
    int threads,
    int batch_size,
    const std::vector<std::string>& texts,
    int warmup_iters,
    int bench_iters)
{
    lembed_sparse_tuning_result_t res = {0};
    res.top_k = top_k;
    res.min_weight = min_weight;
    res.storage_format = storage_format;
    res.threads = threads;
    res.batch_size = batch_size;

    /* Create sparse model */
    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.top_k = top_k;
    opts.min_weight = min_weight;
    opts.storage_format = storage_format;
    opts.show_download_progress = 0;

    /* Resolve model */
    int model_idx = lembed_find_sparse_model_by_code(model_name);
    if (model_idx < 0) {
        /* Unknown model: do not silently benchmark a different one. */
        res.latency_ms = 999999;
        return res;
    }
    opts.model = static_cast<lembed_sparse_model_t>(model_idx);

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    lembed_status_t s = lembed_sparse_text_embedding_create(&opts, &ctx);
    if (s != LEMBED_OK) {
        res.latency_ms = 999999;
        return res;
    }

    /* Prepare texts */
    std::vector<const char*> c_texts(texts.size());
    for (size_t i = 0; i < texts.size(); i++) {
        c_texts[i] = texts[i].c_str();
    }

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        lembed_sparse_embeddings_t result = {0};
        if (lembed_sparse_text_embedding_embed(ctx, c_texts.data(), (int)texts.size(), batch_size, &opts, &result) == LEMBED_OK)
            lembed_sparse_embeddings_free(&result);
    }

    /* Benchmark */
    std::vector<double> times;
    times.reserve(bench_iters > 0 ? (size_t)bench_iters : 0);
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        lembed_sparse_embeddings_t result = {0};
        lembed_status_t es = lembed_sparse_text_embedding_embed(ctx, c_texts.data(), (int)texts.size(), batch_size, &opts, &result);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (es != LEMBED_OK) continue;
        lembed_sparse_embeddings_free(&result);
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    lembed_sparse_text_embedding_free(ctx);

    /* Every iteration failed: report the sentinel so this configuration is
     * never mistaken for the best one. */
    if (times.empty()) {
        res.latency_ms = 999999;
        return res;
    }

    /* Compute stats */
    std::sort(times.begin(), times.end());
    double p50 = percentile_sorted(times, 0.50);

    res.latency_ms = p50;
    res.throughput_docs_sec = (p50 > 0) ? (1000.0 / p50) * texts.size() : 0;

    return res;
}

/* Main sparse auto-tune function */
extern "C" lembed_status_t lembed_sparse_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_sparse_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    int threads = 4;
    int batch_size = 256;
    int n_docs = 20;
    int warmup = 2;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

    /* Generate synthetic texts */
    std::vector<std::string> texts;
    for (int i = 0; i < n_docs; i++) {
        texts.push_back("Machine learning is a branch of artificial intelligence that enables systems to learn from data.");
    }

    /* Configurations to test */
    int top_k_options[] = {32, 64, 128};
    float min_weight_options[] = {0.0f, 0.01f, 0.05f};
    int storage_options[] = {0, 1}; /* dict, CSR */

    lembed_sparse_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total = (int)(sizeof(top_k_options) / sizeof(int) *
                      sizeof(min_weight_options) / sizeof(float) *
                      sizeof(storage_options) / sizeof(int));

    fprintf(stderr, "sparse_autotune: testing %d configurations (mode=%s)...\n",
            total, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL");

    for (int t : top_k_options) {
        for (float w : min_weight_options) {
            for (int s : storage_options) {
                auto r = bench_sparse_config(model_name, t, w, s, threads, batch_size, texts, warmup, bench_iters);

                if (r.latency_ms >= 999999) continue;
                if (r.throughput_docs_sec > best.throughput_docs_sec) {
                    best = r;
                }
            }
        }
    }

    /* No configuration could be benchmarked: do not report the sentinel as a
     * tuning result. */
    if (best.latency_ms >= 999999) {
        *result = {0};
        return LEMBED_ERROR_ONNX_RUNTIME;
    }

    fprintf(stderr, "sparse_autotune: best config: top_k=%d min_weight=%.2f storage=%d (%.1f docs/s)\n",
            best.top_k, best.min_weight, best.storage_format, best.throughput_docs_sec);

    *result = best;
    return LEMBED_OK;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP */




