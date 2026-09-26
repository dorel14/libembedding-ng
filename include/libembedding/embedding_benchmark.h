/*
 * libembedding - embedding_benchmark.h
 * Model selection autotuner by objective
 *
 * Pipeline: Hard constraints -> Pareto frontier -> Objective scoring -> Selection
 *
 * Uses unified types from unified_benchmark.h
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_EMBEDDING_BENCHMARK_H
#define LIBEMBEDDING_EMBEDDING_BENCHMARK_H

#include "types.h"
#include "autotuner.h"
#include "unified_benchmark.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Predefined scoring profiles (weights normalized to sum to 1.0) */
static inline lembed_benchmark_weights_t lembed_benchmark_profile_weights(lembed_objective_t obj) {
    lembed_benchmark_weights_t w;
    switch (obj) {
        case LEMBED_OBJECTIVE_THROUGHPUT:
            w.quality_weight = 0.20f; w.throughput_weight = 0.70f; w.cost_weight = 0.10f;
            break;
        case LEMBED_OBJECTIVE_LATENCY:
            w.quality_weight = 0.70f; w.throughput_weight = 0.20f; w.cost_weight = 0.10f;
            break;
        case LEMBED_OBJECTIVE_MEMORY:
            w.quality_weight = 0.40f; w.throughput_weight = 0.10f; w.cost_weight = 0.50f;
            break;
        default:
            w.quality_weight = 0.50f; w.throughput_weight = 0.30f; w.cost_weight = 0.20f;
            break;
    }
    return w;
}

static inline lembed_benchmark_weights_t lembed_benchmark_custom_weights(float q, float t, float c) {
    float sum = q + t + c;
    if (sum <= 0.0f) sum = 1.0f;
    lembed_benchmark_weights_t w;
    w.quality_weight = q / sum; w.throughput_weight = t / sum; w.cost_weight = c / sum;
    return w;
}

/* Select best model from directory using constraints + Pareto + scoring.
 * model_dir: directory with .gguf files (NULL = default cache)
 * objective: optimization profile
 * constraints: hard limits (NULL = none)
 * custom_weights: override profile (NULL = use profile)
 * result: output best model + config
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_benchmark_select_model(
    const char* model_dir,
    lembed_objective_t objective,
    const lembed_benchmark_constraints_t* constraints,
    const lembed_benchmark_weights_t* custom_weights,
    lembed_benchmark_result_t* result);

/* Auto-detect optimal session count for a model.
 * optimal_sessions: output count (1-8)
 * best_throughput: output throughput
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_benchmark_detect_sessions(
    const char* model_path,
    int max_sessions,
    int* optimal_sessions,
    float* best_throughput);

/* Get default GGUF cache directory (static). */
const char* lembed_benchmark_default_cache_dir(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_EMBEDDING_BENCHMARK_H */

/* ---- Implementation ---- */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(LIBEMBEDDING_EMBEDDING_BENCHMARK_IMPL)
#define LIBEMBEDDING_EMBEDDING_BENCHMARK_IMPL
#include "detail/embedding_benchmark_impl.hpp"
#endif

