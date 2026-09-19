/*
 * libembedding - model_selector.h
 * Automatic model selection based on hardware and use case
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_MODEL_SELECTOR_H
#define LIBEMBEDDING_MODEL_SELECTOR_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Use case for model selection */
typedef enum {
    LEMBED_USE_CASE_SPEED = 0,    /* Prioritize throughput */
    LEMBED_USE_CASE_QUALITY,      /* Prioritize accuracy */
    LEMBED_USE_CASE_BALANCED      /* Balance speed and quality */
} lembed_use_case_t;

/* Model candidate info */
typedef struct {
    char model_name[256];      /* e.g. "BAAI/bge-small-en-v1.5" */
    int dim;                   /* embedding dimension */
    int max_length;            /* max token length */
    int pooling;               /* LEMBED_POOLING_CLS or MEAN */
    int estimated_ram_mb;      /* estimated RAM usage */
    double estimated_throughput; /* estimated docs/sec */
} lembed_model_candidate_t;

/* Select the best model for given hardware and use case */
int lembed_model_select(
    int logical_cores,
    int ram_mb,
    lembed_use_case_t use_case,
    lembed_model_candidate_t* out_selected
);

/* =========================================================================
 * DEPRECATED: Use lembed_cache_hardware_info_t / lembed_cache_detect_hardware()
 * from autotune_cache.h instead. This legacy struct is kept for backward
 * compatibility but may be removed in a future major release.
 * ========================================================================= */
typedef struct {
    char cpu_model[256];
    int physical_cores;
    int logical_cores;
    int ram_mb;
} lembed_hardware_info_t;

int lembed_detect_hardware(lembed_hardware_info_t* out_info);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#include "autotune_cache.h"

int lembed_detect_hardware(lembed_hardware_info_t* out_info) {
    if (!out_info) return 0;
    lembed_cache_hardware_info_t hw = {0};
    if (lembed_cache_detect_hardware(&hw) != LEMBED_OK) return 0;
    memset(out_info, 0, sizeof(*out_info));
    memcpy(out_info->cpu_model, hw.cpu_name, sizeof(out_info->cpu_model) - 1);
    out_info->physical_cores = hw.physical_cores;
    out_info->logical_cores = hw.logical_cores;
    out_info->ram_mb = hw.ram_mb;
    return hw.logical_cores;
}
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_MODEL_SELECTOR_H */
