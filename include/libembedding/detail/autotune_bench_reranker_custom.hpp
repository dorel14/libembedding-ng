/*
 * libembedding - detail/autotune_bench_reranker_custom.hpp
 * Reranker auto-tuner with custom corpus
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_CUSTOM_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_CUSTOM_HPP

#include "libembedding/autotuner.h"
#include "libembedding/reranker.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"
#include "autotune_bench_reranker.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace lembed { namespace detail {

/* Benchmark a single reranker configuration with custom documents */
inline lembed_reranker_tuning_result_t bench_reranker_config_custom(
    const char* model_name,
    int threads,
    int batch_size,
    int max_tokens,
    const std::vector<std::string>& docs,
    const char* query,
    int warmup_iters,
    int bench_iters)
{
    lembed_reranker_tuning_result_t res = {0};
    res.threads = threads;
    res.batch_size = batch_size;
    res.max_tokens = max_tokens;

    /* Create reranker */
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.max_length = max_tokens;
    opts.show_download_progress = 0;

    /* Resolve model by name or code */
    int model_idx = -1;
    {
        const lembed_model_info_t* models = nullptr;
        int count = 0;
        lembed_list_reranker_models(&models, &count);
        for (int i = 0; i < count; i++) {
            std::string name = models[i].model_name;
            std::string code = models[i].model_code;
            if (model_name == name || model_name == code) {
                model_idx = i;
                break;
            }
        }
    }
    if (model_idx < 0) {
        /* Unknown model: do not silently benchmark a different model */
        res.latency_ms = 999999;
        return res;
    }
    opts.model = static_cast<lembed_reranker_model_t>(model_idx);

    lembed_reranker_t* ctx = nullptr;
    lembed_status_t s = lembed_reranker_create(&opts, &ctx);
    if (s != LEMBED_OK) {
        res.latency_ms = 999999;
        return res;
    }

    /* Build C string array */
    std::vector<const char*> c_docs;
    c_docs.reserve(docs.size());
    for (const auto& d : docs) c_docs.push_back(d.c_str());
    if (c_docs.empty()) {
        lembed_reranker_free(ctx);
        res.latency_ms = 999999;
        return res;
    }

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        lembed_rerank_results_t result = {0};
        if (lembed_reranker_rerank(ctx, query, c_docs.data(), (int)c_docs.size(),
                                    batch_size, &result) == LEMBED_OK) {
            lembed_rerank_results_free(&result);
        }
    }

    /* Benchmark */
    std::vector<double> times;
    times.reserve(bench_iters);
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        lembed_rerank_results_t result = {0};
        lembed_status_t rs = lembed_reranker_rerank(ctx, query, c_docs.data(),
                                                    (int)c_docs.size(),
                                                    batch_size, &result);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (rs != LEMBED_OK) continue;
        lembed_rerank_results_free(&result);
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    lembed_reranker_free(ctx);

    if (times.empty()) {
        res.latency_ms = 999999;
        return res;
    }

    /* Compute stats */
    std::sort(times.begin(), times.end());
    double p50 = times[times.size() / 2];
    double p95 = times[(size_t)(0.95 * (double)times.size())];
    if ((size_t)(0.95 * (double)times.size()) >= times.size()) p95 = times.back();

    res.latency_ms = p50;
    res.p95_latency_ms = p95;
    res.throughput_docs_sec = (p50 > 0) ? (1000.0 / p50) * (double)c_docs.size() : 0;
    res.memory_mb = 0;

    return res;
}

/* Main reranker auto-tune with custom corpus implementation */
inline lembed_status_t lembed_reranker_autotune_custom_run(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    int cores = cpu_logical_cores();
    int warmup = 1;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

    /* Build documents and query from corpus.
     * The last text is used as the query and is excluded from the
     * document list so the query is not scored against itself. */
    std::vector<std::string> docs;
    docs.reserve(n_texts);
    for (int i = 0; i < n_texts; i++)
        docs.push_back(texts[i]);

    std::string query_storage;
    const char* query;
    std::vector<std::string> bench_docs;

    if (docs.size() >= 2) {
        query_storage = docs.back();
        query = query_storage.c_str();
        bench_docs.assign(docs.begin(), docs.end() - 1);
    } else if (docs.size() == 1) {
        /* Single text: duplicate it as its own document so the
         * benchmark still has at least one (query, document) pair. */
        query_storage = docs[0];
        query = query_storage.c_str();
        bench_docs.push_back(docs[0]);
    } else {
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }

    /* Configurations to test */
    std::vector<int> threads_vec, batch_vec, tokens_vec;
    if (mode == LEMBED_AUTOTUNE_QUICK) {
        threads_vec = {1, 4, 8};
        batch_vec = {4, 16};
        tokens_vec = {64, 256};
    } else {
        threads_vec = {1, 2, 4, 8};
        batch_vec = {4, 8, 16};
        tokens_vec = {32, 64, 128, 256};
    }

    /* Filter threads > cores */
    std::vector<int> valid_threads;
    for (int t : threads_vec) {
        if (t <= cores) valid_threads.push_back(t);
    }

    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    for (int t : valid_threads) {
        for (int b : batch_vec) {
            for (int k : tokens_vec) {
                auto r = bench_reranker_config_custom(
                    model_name, t, b, k, bench_docs, query, warmup, bench_iters);

                double score = score_reranker_config(r, objective);
                double best_score = score_reranker_config(best, objective);

                if (score < best_score) {
                    best = r;
                }
            }
        }
    }

    fprintf(stderr, "reranker_autotune(custom): best config: threads=%d batch=%d tokens=%d (P50=%.1fms, P95=%.1fms)\n",
            best.threads, best.batch_size, best.max_tokens, best.latency_ms, best.p95_latency_ms);

    if (best.latency_ms >= 999999) {
        return LEMBED_ERROR_ONNX_RUNTIME;
    }

    *result = best;
    return LEMBED_OK;
}

/* C API: reranker autotune with custom corpus */
inline lembed_status_t lembed_reranker_autotune_custom_impl(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        return lembed_reranker_autotune_custom_run(
            model_name, texts, n_texts, mode, objective, result);
    } catch (const std::exception& e) {
        fprintf(stderr, "reranker_autotune_custom: exception: %s\n", e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_CUSTOM_HPP */
