/*
 * libembedding - detail/autotune_bench_text_custom.hpp
 * Text embedding auto-tuner with custom corpus
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_CUSTOM_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_CUSTOM_HPP

#include "libembedding/autotuner.h"
#include "libembedding/text_embedding.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"
#include "autotune_bench_text.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace lembed { namespace detail {

/* Auto-tune with custom corpus (user-provided texts) */
inline lembed_status_t autotune_text_custom_impl(
        lembed_text_model_t model,
        const std::vector<std::string>& corpus,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {

    const char* model_code = "";
    lembed_model_info_t info;
    if (lembed_get_text_model_info(model, &info) == LEMBED_OK) {
        model_code = info.model_code;
    }

    int cores = cpu_logical_cores();
    int n_samples = (int)corpus.size();

    if (n_samples == 0) {
        if (result) memset(result, 0, sizeof(*result));
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }

    /* Configurations to test */
    struct Config { int workers; int threads; int batch; };
    std::vector<Config> configs;

    if (mode == LEMBED_AUTOTUNE_QUICK) {
        int worker_opts[] = {1, std::min(4, cores), std::min(8, cores)};
        for (int w : worker_opts) {
            if (w > cores) continue;
            configs.push_back({w, 1, 32});
        }
    } else {
        int worker_opts[] = {1, 2, 4, 8};
        int thread_opts[] = {1, 2, 4};
        int batch_opts[] = {16, 32, 64, 128, 256};

        for (int w : worker_opts) {
            if (w > cores) continue;
            for (int t : thread_opts) {
                if (w * t > cores) continue;
                for (int b : batch_opts) {
                    configs.push_back({w, t, b});
                }
            }
        }
    }

    double best_score = -1e18;
    lembed_tuning_result_t best = {1, 1, 64, 0, 0, 0};

    if (configs.empty()) {
        /* Single-core host in QUICK mode: fall back to one safe config */
        configs.push_back({1, 1, 32});
    }

    for (const auto& cfg : configs) {
        double latency = 0.0;
        double throughput = benchmark_text_config(corpus, cfg.workers, cfg.threads, cfg.batch, latency);
        if (throughput <= 0.0) {
            /* Configuration failed to run (model load error, etc.) */
            fprintf(stderr, "  autotune(custom): workers=%d threads=%d batch=%d -> FAILED\n",
                    cfg.workers, cfg.threads, cfg.batch);
            continue;
        }
        /* Workers dominate memory footprint; this is a relative estimate
         * used only to rank configurations, not an absolute RSS reading. */
        double memory = cfg.workers * 100.0;
        double score = score_text_config(throughput, latency, memory);

        fprintf(stderr, "  autotune(custom): workers=%d threads=%d batch=%d -> %.1f docs/s\n",
                cfg.workers, cfg.threads, cfg.batch, throughput);

        if (score > best_score) {
            best_score = score;
            best.workers = cfg.workers;
            best.threads = cfg.threads;
            best.batch_size = cfg.batch;
            best.throughput_docs_sec = throughput;
            best.latency_ms = latency;
            best.memory_mb = memory;
        }
    }

    if (best_score <= -1e18) {
        /* No configuration could be benchmarked */
        if (result) memset(result, 0, sizeof(*result));
        return LEMBED_ERROR_ONNX_RUNTIME;
    }

    if (result) *result = best;

    /* Write to cache if we have a model code */
    if (model_code && model_code[0]) {
        std::ofstream of(get_cache_path(model_code));
        if (of.is_open()) {
            of << "{\n";
            of << "  \"workers\": " << best.workers << ",\n";
            of << "  \"threads\": " << best.threads << ",\n";
            of << "  \"batch_size\": " << best.batch_size << ",\n";
            of << "  \"throughput_docs_sec\": " << best.throughput_docs_sec << ",\n";
            of << "  \"latency_ms\": " << best.latency_ms << ",\n";
            of << "  \"memory_mb\": " << best.memory_mb << "\n";
            of << "}\n";
        }
    }

    return LEMBED_OK;
}

/* C API: autotune with custom corpus */
inline lembed_status_t lembed_autotune_custom_impl(
        const char* model_name,
        const char* const* texts,
        int n_texts,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    int idx = lembed_find_text_model_by_code(model_name);
    if (idx < 0) return LEMBED_ERROR_MODEL_NOT_FOUND;

    std::vector<std::string> corpus;
    corpus.reserve(n_texts);
    for (int i = 0; i < n_texts; i++)
        corpus.push_back(texts[i]);

    return autotune_text_custom_impl((lembed_text_model_t)idx, corpus, mode, result);
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_CUSTOM_HPP */
