/*
 * libembedding - gguf_registry.h
 * Recommended GGUF models for llama.cpp backend
 *
 * Unlike ONNX models (loaded by enum), GGUF models are loaded by file path.
 * This registry provides metadata for known-good GGUF models: download URLs,
 * dimensions, quality scores, and recommended use cases.
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_REGISTRY_H
#define LIBEMBEDDING_GGUF_REGISTRY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF model metadata */
typedef struct {
    const char* name;           /* Human-readable name (e.g., "Snowflake-XS-Q4") */
    const char* gguf_url;       /* Download URL for the .gguf file */
    const char* model_code;     /* Original model code (e.g., "snowflake/snowflake-arctic-embed-xs") */
    const char* description;    /* Short description */
    int         dim;            /* Embedding dimension */
    int         params_m;       /* Parameter count in millions */
    int         file_size_mb;   /* Approximate file size in MB */
    float       quality_mteb;   /* MTEB Retrieval Score (NDCG@10), 0 if unknown */
    int         recommended_sessions; /* Recommended session count for pooling */
} lembed_gguf_model_info_t;

/* List all recommended GGUF models (returns pointer to static data, do not free) */
lembed_status_t lembed_list_gguf_models(const lembed_gguf_model_info_t** out, int* count);

/* Find a GGUF model by name (case-insensitive partial match) */
const lembed_gguf_model_info_t* lembed_find_gguf_model(const char* name);

/* Get the default recommended GGUF model (best quality/throughput tradeoff) */
const lembed_gguf_model_info_t* lembed_default_gguf_model(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_GGUF_REGISTRY_H */

/* ---- Implementation ---- */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(LIBEMBEDDING_GGUF_REGISTRY_IMPL)
#define LIBEMBEDDING_GGUF_REGISTRY_IMPL
#include "detail/gguf_registry_impl.hpp"
#endif

