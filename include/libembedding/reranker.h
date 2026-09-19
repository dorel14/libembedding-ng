/*
 * libembedding - reranker.h
 * Cross-encoder reranker C API (ONNX + llama.cpp backends)
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_RERANKER_H
#define LIBEMBEDDING_RERANKER_H

#include "types.h"
#include "llamacpp_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

lembed_reranker_options_t lembed_reranker_options_default(void);

/* Auto-routing create: ONNX registry for HuggingFace IDs, GGUF for .gguf paths */
lembed_status_t lembed_reranker_create(
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Versioned create preserving the v1.4.0 options ABI. */
lembed_status_t lembed_reranker_create_v2(
    const lembed_reranker_options_v2_t* options,
    lembed_reranker_t** out);

/* Load from local directory (auto-detects ONNX vs GGUF) or direct .gguf path */
lembed_status_t lembed_reranker_create_from_path(
    const char* path,
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Explicit GGUF loading */
lembed_status_t lembed_reranker_create_from_gguf_path(
    const char* gguf_path,
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Load GGUF model from HuggingFace (llama.cpp backend) */
lembed_status_t lembed_reranker_create_from_gguf_model(
    const char* repo,
    const char* filename,
    const lembed_reranker_options_t* options,
    lembed_reranker_t** out);

/* Rerank documents against a query.
 * Results are sorted by score descending. */
lembed_status_t lembed_reranker_rerank(
    lembed_reranker_t* ctx,
    const char* query,
    const char* const* documents,
    int num_documents,
    int batch_size,
    lembed_rerank_results_t* result);

/* Introspection */
const lembed_model_desc_t* lembed_reranker_desc(const lembed_reranker_t* ctx);
const lembed_model_desc_v2_t* lembed_reranker_desc_v2(const lembed_reranker_t* ctx);
const char* lembed_reranker_model_name(const lembed_reranker_t* ctx);
int lembed_reranker_max_length(const lembed_reranker_t* ctx);

/* Runtime statistics */
void lembed_reranker_stats(const lembed_reranker_t* ctx, lembed_stats_t* out);
void lembed_reranker_stats_v2(const lembed_reranker_t* ctx, lembed_stats_v2_t* out);

void lembed_reranker_free(lembed_reranker_t* ctx);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#include "detail/reranker_impl.hpp"
#endif

#endif /* LIBEMBEDDING_RERANKER_H */
