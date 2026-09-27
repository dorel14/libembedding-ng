/*
 * libembedding - autotuner.h
 * Auto-tuning for optimal embedding performance
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_AUTOTUNER_H
#define LIBEMBEDDING_AUTOTUNER_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tuning result */
typedef struct {
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
} lembed_tuning_result_t;

/* Autotune modes */
typedef enum {
    LEMBED_AUTOTUNE_QUICK = 0,  /* 5-15s */
    LEMBED_AUTOTUNE_FULL        /* 30-120s */
} lembed_autotune_mode_t;

/* Run auto-tuning for a model
 * Returns optimal configuration for current hardware.
 * mode: QUICK (5-15s) or FULL (30-120s)
 * result: output configuration
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result);

/* Run auto-tuning with custom corpus
 * texts: array of text samples
 * n_texts: number of texts
 * avg_tokens: average token count (for calibration) */
lembed_status_t lembed_autotune_custom(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_tuning_result_t* result);

/* Clear autotune cache for a model (or all if model_name=NULL) */
void lembed_autotune_clear_cache(const char* model_name);

/* Auto model selection result */
typedef struct {
    const char* model_code;
    const char* model_name;
    int dim;
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
    double score;
} lembed_model_selection_t;

/* Auto-select best model for hardware and use case
 * use_case: "speed", "quality", or "balanced" (default)
 * Returns optimal model + configuration */
lembed_status_t lembed_auto_select_model(
    const char* use_case,
    lembed_model_selection_t* result);

/* =========================================================================
 * Reranker Auto-Tuner
 * ========================================================================= */

/* Reranker tuning result */
typedef struct {
    int threads;
    int batch_size;
    int max_tokens;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
    double p95_latency_ms;
} lembed_reranker_tuning_result_t;

/* Optimization objectives */
typedef enum {
    LEMBED_OBJECTIVE_LATENCY = 0,    /* minimize latency */
    LEMBED_OBJECTIVE_THROUGHPUT = 1, /* maximize throughput */
    LEMBED_OBJECTIVE_BALANCED = 2,   /* balance latency/throughput */
    LEMBED_OBJECTIVE_MEMORY = 3,     /* minimize memory */
} lembed_objective_t;

/* Reranker profiles for auto-config */
typedef enum {
    LEMBED_PROFILE_INTERACTIVE = 0,  /* latency priority: <100ms */
    LEMBED_PROFILE_BALANCED = 1,     /* balance: ~300ms */
    LEMBED_PROFILE_QUALITY = 2,      /* quality priority: best NDCG */
} lembed_reranker_profile_t;

/* Run auto-tuning for a reranker model
 * Optimizes threads × batch_size × max_tokens for the given objective.
 * mode: QUICK (5-15s) or FULL (30-120s)
 * objective: what to optimize for
 * result: output configuration
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_reranker_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result);

/* Run reranker auto-tuning with constraints
 * min_tokens: minimum acceptable max_tokens (quality constraint)
 * max_latency_ms: maximum acceptable latency
 * Only considers configurations that satisfy both constraints */
lembed_status_t lembed_reranker_autotune_constrained(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    int min_tokens,
    double max_latency_ms,
    lembed_reranker_tuning_result_t* result);

/* Run reranker auto-tuning with custom corpus
 * texts: array of text samples
 * n_texts: number of texts
 * mode: QUICK or FULL
 * objective: what to optimize for
 * result: output configuration */
lembed_status_t lembed_reranker_autotune_custom(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result);

/* Auto-configure reranker for a target latency budget
 * target_latency_ms: maximum acceptable latency (e.g., 500ms)
 * objective: what to optimize for within the budget
 * result: output configuration that fits within budget */
lembed_status_t lembed_reranker_auto_config(
    const char* model_name,
    double target_latency_ms,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result);

/* Auto-configure reranker using a profile
 * profile: INTERACTIVE (fast), BALANCED, or QUALITY (best ranking)
 * result: output configuration for the profile */
lembed_status_t lembed_reranker_auto_config_profile(
    const char* model_name,
    lembed_reranker_profile_t profile,
    lembed_reranker_tuning_result_t* result);

/* Clear reranker autotune cache for a model (or all if model_name=NULL) */
void lembed_reranker_autotune_clear_cache(const char* model_name);

/* =========================================================================
 * Unified Auto-Tuner (framework for all task types)
 * ========================================================================= */

/* Task types for unified auto-tuner */
typedef enum {
    LEMBED_TASK_EMBEDDING = 0,   /* Text embedding */
    LEMBED_TASK_RERANKING = 1,   /* Document reranking */
    LEMBED_TASK_IMAGE = 2,       /* Image embedding (future) */
    LEMBED_TASK_SPARSE = 3,      /* Sparse embedding (future) */
} lembed_task_t;

/* Unified tuning result (union of all task-specific results) */
typedef struct {
    lembed_task_t task;
    int threads;
    int batch_size;
    int workers;         /* embedding only */
    int max_tokens;      /* reranker only */
    int top_k;           /* sparse only */
    float min_weight;    /* sparse only */
    int storage_format;  /* sparse only */
    double throughput_docs_sec;
    double latency_ms;
    double p95_latency_ms;
    double memory_mb;
} lembed_unified_tuning_result_t;

/* Sparse auto-tuner result */
typedef struct {
    int top_k;
    float min_weight;
    int storage_format;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
} lembed_sparse_tuning_result_t;

/* Run auto-tuning for sparse embedding
 * Optimizes top_k x min_weight x storage_format for throughput */
lembed_status_t lembed_sparse_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_sparse_tuning_result_t* result);

/* Unified auto-tune entry point
 * Routes to the appropriate implementation based on task type.
 * This is the recommended API for all auto-tuning needs.
 *
 * task: EMBEDDING, RERANKING, IMAGE, or SPARSE
 * model_name: model identifier (e.g. "BAAI/bge-small-en-v1.5")
 * mode: QUICK (5-15s) or FULL (30-120s)
 * result: output configuration
 */
lembed_status_t lembed_autotune_unified(
    lembed_task_t task,
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_unified_tuning_result_t* result);

/* Unified auto-config with latency budget */
lembed_status_t lembed_autotune_unified_config(
    lembed_task_t task,
    const char* model_name,
    double target_latency_ms,
    lembed_unified_tuning_result_t* result);

/* Clear unified autotune cache */
void lembed_autotune_unified_clear_cache(lembed_task_t task, const char* model_name);

/* =========================================================================
 * Image Auto-Tuner
 * ========================================================================= */

/* Image tuning result */
typedef struct {
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
} lembed_image_tuning_result_t;

/* Run auto-tuning for an image embedding model
 * Optimizes threads × batch_size for throughput */
lembed_status_t lembed_image_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_image_tuning_result_t* result);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_AUTOTUNER_H */




