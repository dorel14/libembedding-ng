/*
 * libembedding - autotune_cache.h
 * Caching for autotuning results with full fingerprint
 *
 * Cache key = hash of hardware + software + model
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_AUTOTUNE_CACHE_H
#define LIBEMBEDDING_AUTOTUNE_CACHE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hardware fingerprint for tuning cache */
typedef struct {
    char        cpu_name[128];
    int         physical_cores;
    int         logical_cores;
    char        os_name[64];
    int         ram_mb;
    char        features[256];     /* e.g. "AVX2,AVX512" */
} lembed_cache_hardware_info_t;

/* Software fingerprint for tuning cache */
typedef struct {
     char        libembedding[32];  /* e.g. "1.4.0" */
    char        llama_cpp[32];     /* e.g. "b5434" */
} lembed_cache_software_info_t;

/* Model fingerprint for tuning cache */
typedef struct {
    char        model_id[128];     /* e.g. "Snowflake-XS-Q4" */
    char        quantization[16];  /* e.g. "Q4_K_M" */
    int         dim;               /* embedding dimension */
    int         file_size_bytes;   /* for change detection */
} lembed_cache_model_info_t;

/* Single configuration result */
typedef struct {
    int         num_sessions;
    int         num_threads;
    int         batch_size;
    float       throughput_docs_sec;
    float       latency_p50_ms;
    float       latency_p95_ms;
} lembed_tune_config_result_t;

/* Cached tuning entry (full) */
typedef struct {
    int         cache_schema_version; /* current = 1 */
    lembed_cache_hardware_info_t hardware;
    lembed_cache_software_info_t software;
    lembed_cache_model_info_t model;
    char        backend[32];       /* "llama.cpp" or "onnx" */
    /* All measured configurations */
    int         num_configs;
    lembed_tune_config_result_t configs[16];
    /* Selected best */
    int         best_idx;           /* index into configs[] */
} lembed_tune_cache_entry_t;

#define LEMBED_TUNE_CACHE_SCHEMA_VERSION 1

/* Load cached result.
 * Returns LEMBED_OK on hit, LEMBED_ERROR_CACHE_MISS on miss. */
lembed_status_t lembed_tune_cache_load(
    const lembed_cache_hardware_info_t* hw,
    const lembed_cache_software_info_t* sw,
    const lembed_cache_model_info_t* model,
    const char* backend,
    lembed_tune_cache_entry_t* entry);

/* Save result to cache (includes all measured configs). */
lembed_status_t lembed_tune_cache_save(const lembed_tune_cache_entry_t* entry);

/* Clear all cache. */
lembed_status_t lembed_tune_cache_clear(void);

/* Auto-detect hardware for cache. */
lembed_status_t lembed_cache_detect_hardware(lembed_cache_hardware_info_t* hw);

/* Get software info for cache. */
lembed_status_t lembed_cache_detect_software(lembed_cache_software_info_t* sw);

/* Generate cache key from fingerprints. */
void lembed_tune_cache_key(const lembed_cache_hardware_info_t* hw,
                           const lembed_cache_software_info_t* sw,
                           const lembed_cache_model_info_t* model,
                           const char* backend,
                           char* key_out);

/* Cache file path (static). */
const char* lembed_tune_cache_path(void);

/* Add a config result to entry. */
void lembed_tune_cache_add_config(lembed_tune_cache_entry_t* entry,
                                   const lembed_tune_config_result_t* config);

/* Set best config index. */
void lembed_tune_cache_set_best(lembed_tune_cache_entry_t* entry, int idx);

/* Clear all cache. */
lembed_status_t lembed_tune_cache_clear(void);

/* Cache file path (static). */
const char* lembed_tune_cache_path(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_AUTOTUNE_CACHE_H */

/* ---- Implementation ---- */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(LIBEMBEDDING_AUTOTUNE_CACHE_IMPL)
#define LIBEMBEDDING_AUTOTUNE_CACHE_IMPL
#include "detail/autotune_cache_impl.hpp"
#endif





