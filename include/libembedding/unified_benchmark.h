/*
 * libembedding - unified_benchmark.h
 * Unified Backend Benchmark: ONNX vs llama.cpp
 *
 * Same corpus, same metrics, same protocol for both backends.
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_UNIFIED_BENCHMARK_H
#define LIBEMBEDDING_UNIFIED_BENCHMARK_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Test categories */
typedef enum {
    LEMBED_CORPUS_SHORT = 0,     /* < 20 tokens */
    LEMBED_CORPUS_MEDIUM,        /* 20-80 tokens */
    LEMBED_CORPUS_LONG,          /* 80-200 tokens */
    LEMBED_CORPUS_VERY_LONG,     /* 200+ tokens */
    LEMBED_CORPUS_MIXED,         /* All lengths */
    LEMBED_CORPUS_MULTILINGUAL,  /* Multi-language */
    LEMBED_CORPUS_EDGE_CASES,    /* Edge cases */
} lembed_corpus_type_t;

/* Scoring weights (normalized to sum to 1.0) */
typedef struct {
    float quality_weight;
    float throughput_weight;
    float cost_weight;
} lembed_benchmark_weights_t;

/* Hard constraints for elimination (use 0 to disable) */
typedef struct {
    float quality_min;       /* Minimum retrieval recall@3 (0-1) */
    float throughput_min;    /* Minimum docs/sec */
    float memory_max_mb;     /* Maximum model file size in MB */
} lembed_benchmark_constraints_t;

/* Per-backend configuration */
typedef struct {
    char        backend[32];      /* "onnx" or "llama.cpp" */
    int         num_threads;      /* ONNX: threads, llama.cpp: threads/session */
    int         batch_size;       /* ONNX: batch size, llama.cpp: sessions */
    int         workers;          /* ONNX: workers */
} lembed_backend_config_t;

/* Metrics for one run */
typedef struct {
    float       throughput_docs_sec;
    float       latency_p50_ms;
    float       latency_p95_ms;
    float       load_time_ms;
    float       peak_memory_mb;
    int         dim;
    int         num_texts;
    int         num_errors;
} lembed_benchmark_metrics_t;

/* Result for one model × backend */
typedef struct {
    char        model_name[128];
    char        model_path[512];
    char        backend[32];
    lembed_backend_config_t config;
    lembed_benchmark_metrics_t metrics;
    float       score;            /* Composite score */
    float       quality_score;    /* Retrieval quality (0-1) */
    float       file_size_mb;     /* Model file size */
    int         pareto_rank;       /* 1 = Pareto optimal */
} lembed_benchmark_result_t;

/* Run unified benchmark for a model on a specific backend.
 * model_path: path to model file (.onnx or .gguf)
 * backend: "onnx" or "llama.cpp"
 * corpus_type: which corpus to use
 * config: backend-specific configuration
 * result: output metrics
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_benchmark_run(
    const char* model_path,
    const char* backend,
    lembed_corpus_type_t corpus_type,
    const lembed_backend_config_t* config,
    lembed_benchmark_result_t* result);

/* Run comparison: same model on both backends (if available).
 * onnx_path: path to ONNX model (NULL if not available)
 * gguf_path: path to GGUF model (NULL if not available)
 * corpus_type: which corpus to use
 * results: output array of 2 results [onnx, llama]
 * returns: number of results filled (0-2) */
int lembed_benchmark_compare(
    const char* onnx_path,
    const char* gguf_path,
    lembed_corpus_type_t corpus_type,
    lembed_benchmark_result_t* results);

/* Get test corpus (deduplicated from plan).
 * type: corpus category
 * out_texts: output array of const char* (static, do not free)
 * out_count: output number of texts
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_benchmark_get_corpus(
    lembed_corpus_type_t type,
    const char* const** out_texts,
    int* out_count);

/* Auto-tune a backend for a model.
 * model_path: path to model
 * backend: "onnx" or "llama.cpp"
 * objective: optimization objective
 * result: output best config + metrics
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_benchmark_autotune(
    const char* model_path,
    const char* backend,
    lembed_objective_t objective,
    lembed_benchmark_result_t* result);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_UNIFIED_BENCHMARK_H */

/* ---- Implementation ---- */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(LIBEMBEDDING_UNIFIED_BENCHMARK_IMPL)
#define LIBEMBEDDING_UNIFIED_BENCHMARK_IMPL
#include "detail/unified_benchmark_impl.hpp"
#endif




