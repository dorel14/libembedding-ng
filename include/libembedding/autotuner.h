/*
 * libembedding - autotuner.h
 * Auto-tuning for optimal embedding performance
 *
 * Auteur: David Orel
 * Version: 1.12.1
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
    LEMBED_AUTOTUNE_QUICK = 0,  /* small grid, 1 warmup + 3 timed calls per point */
    LEMBED_AUTOTUNE_FULL        /* larger grid, 2 warmups + 10 timed calls per point */
} lembed_autotune_mode_t;

/* Run auto-tuning for a model
 * Returns optimal configuration for current hardware.
 * mode: QUICK or FULL (duration depends on model and CPU: seconds to minutes)
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
 * Quantization Auto-Selection
 *
 * Picks the quantization variant that is actually fastest on this machine,
 * instead of asking the caller to choose between FP32, static INT8, dynamic INT8
 * and FP16 without knowing which one wins.
 *
 * Only the variants the registry really ships are considered: a model with no
 * STATIC sibling is never compared against one. FP32 is the baseline, and it is
 * kept unless another variant is measurably faster.
 *
 * lembed_quantization_choice_t is declared in types.h.
 * ========================================================================= */

/* Choose the best quantization variant for a text model.
 *
 * On the first call for a given (model, machine, library version) the available
 * variants are benchmarked and the decision is cached; later calls only read the
 * cache. Variants whose weights are absent are skipped, never downloaded.
 *
 * num_docs: corpus size for the benchmark. Clamped to [16, 256]: the lower
 *          bound is what makes two variants comparable, the upper one keeps the
 *          first load from stalling. 0 selects the default (16).
 * dry_run:  1 = measure and report without writing the cache.
 *
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_quantization_auto_select(
    lembed_text_model_t model,
    int num_threads,
    int batch_size,
    int num_docs,
    int dry_run,
    lembed_quantization_choice_t* result);

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
 * max_tokens truncates documents (speed vs ranking quality): set a floor with
 * lembed_reranker_autotune_constrained() or lembed_autotune_ex().
 * mode: QUICK or FULL
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
    LEMBED_TASK_IMAGE = 2,       /* Image embedding */
    LEMBED_TASK_SPARSE = 3,      /* Sparse embedding */
} lembed_task_t;

/* Unified tuning result (union of all task-specific results) */
typedef struct {
    lembed_task_t task;
    int threads;
    int batch_size;
    int workers;         /* embedding only */
    int max_tokens;      /* reranker only */
    int top_k;           /* sparse only: model default, reported, not tuned */
    float min_weight;    /* sparse only: model default, reported, not tuned */
    int storage_format;  /* sparse only: model default, reported, not tuned */
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
 * Optimizes threads x batch_size for throughput. top_k, min_weight and
 * storage_format change the output vectors, so they are not tuned: the result
 * reports the model defaults. */
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
 * mode: QUICK or FULL
 * result: output configuration
 * Facade over lembed_autotune_ex() with the task's default options.
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

/* Clear the autotune cache of one task for a model (or every model if NULL).
 * Covers all four tasks. */
void lembed_autotune_unified_clear_cache(lembed_task_t task, const char* model_name);

/* Options of lembed_autotune_ex(). Always start from
 * lembed_autotune_options_default(task), then override fields. */
typedef struct {
    lembed_task_t task;
    lembed_autotune_mode_t mode;     /* QUICK or FULL */
    lembed_objective_t objective;    /* see lembed_autotune_options_default() */

    /* Optional custom workload (NULL / 0 = built-in synthetic workload).
     * EMBEDDING, SPARSE: the texts to embed.
     * RERANKING: the last text is the query, the others are the documents.
     * IMAGE: not supported (LEMBED_ERROR_INVALID_ARGUMENT). */
    const char* const* texts;
    int n_texts;

    /* p95 budget in ms for one tuning call (0 = none). A call is one request
     * over the whole workload: the documents of a reranking query, or the
     * full text/image batch. When no configuration fits, the best
     * unconstrained one is returned: compare p95_latency_ms to the budget. */
    double max_latency_ms;

    /* RERANKING only: smallest acceptable max_tokens (0 = none). */
    int min_tokens;
} lembed_autotune_options_t;

/* Defaults for a task: QUICK mode, no budget, synthetic workload, and the
 * objective each task has always optimised: THROUGHPUT for EMBEDDING, SPARSE
 * and IMAGE (batch workloads), BALANCED for RERANKING (request workload). */
lembed_autotune_options_t lembed_autotune_options_default(lembed_task_t task);

/* Auto-tune any task with full control over objective, workload and
 * constraints. Results are cached per (model, machine, runtime, workload,
 * objective, mode, constraints); a repeated call returns in milliseconds.
 *
 * Search: every grid point is measured (warmup, then P50/P95 over timed
 * calls); points clearly slower than the best so far stop early; costs within
 * 5% of the best are treated as a tie and resolved toward the preferred point
 * (fewer sessions, less truncation, smaller batch), so the choice does not flip
 * on measurement noise.
 *
 * Availability: like every auto-tuner entry point, this is compiled into the
 * shared library (python/csrc/libembedding_shlib.cpp), not into the
 * header-only LIBEMBEDDING_IMPLEMENTATION build.
 *
 * model_name: registry repo or canonical name.
 * options:    NULL = lembed_autotune_options_default(LEMBED_TASK_EMBEDDING).
 * result:     filled on LEMBED_OK; latency_ms/p95_latency_ms are per call,
 *             except EMBEDDING where they are amortised per text (as in
 *             lembed_tuning_result_t).
 * Returns LEMBED_ERROR_MODEL_NOT_FOUND for an unknown model,
 * LEMBED_ERROR_INVALID_ARGUMENT for inconsistent options, and
 * LEMBED_ERROR_ONNX_RUNTIME when no configuration could run. */
lembed_status_t lembed_autotune_ex(
    const char* model_name,
    const lembed_autotune_options_t* options,
    lembed_unified_tuning_result_t* result);

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

/* ---- Implementation ---- */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(LIBEMBEDDING_QUANTIZATION_AUTO_ENTRY)
#define LIBEMBEDDING_QUANTIZATION_AUTO_ENTRY
#include "detail/quantization_auto_entry.hpp"
#endif




