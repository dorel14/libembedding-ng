/*
 * libembedding - text_embedding.h
 * Dense text embedding C API (declarations only)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_TEXT_EMBEDDING_H
#define LIBEMBEDDING_TEXT_EMBEDDING_H

#include "types.h"
#include "llamacpp_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default-initialize text options */
lembed_text_options_t lembed_text_options_default(void);

/* Create a text embedding context (downloads model if needed) */
lembed_status_t lembed_text_embedding_create(
    const lembed_text_options_t* options,
    lembed_text_embedding_t** out);

/* Create from user-defined model (bring your own ONNX) */
lembed_status_t lembed_text_embedding_create_custom(
    const lembed_user_defined_model_t* model,
    lembed_execution_provider_t provider,
    int num_threads,
    lembed_text_embedding_t** out);

/* Embed texts. Result is a flat float array [num_texts * dim].
 * batch_size=0 means use the context's configured batch_size. */
lembed_status_t lembed_text_embedding_embed(
    lembed_text_embedding_t* ctx,
    const char* const* texts,
    int num_texts,
    int batch_size,
    lembed_embeddings_t* result);

/* Embed texts as a stream. */
lembed_status_t lembed_text_embedding_embed_stream(
    lembed_text_embedding_t* ctx,
    const char* const* texts,
    int num_texts,
    int batch_size,
    void (*callback)(const float* embedding, int dim, void* userdata),
    void* userdata);

/* Get embedding dimension */
int lembed_text_embedding_dim(const lembed_text_embedding_t* ctx);

/* Introspection */
const lembed_model_desc_t* lembed_text_embedding_desc(const lembed_text_embedding_t* ctx);
const char* lembed_text_embedding_model_name(const lembed_text_embedding_t* ctx);
int lembed_text_embedding_max_length(const lembed_text_embedding_t* ctx);

/* Runtime statistics */
void lembed_text_embedding_stats(const lembed_text_embedding_t* ctx, lembed_stats_t* out);

/* Destroy context */
void lembed_text_embedding_free(lembed_text_embedding_t* ctx);

/* Load from local directory */
lembed_status_t lembed_text_embedding_create_from_path(
    const char* dir_path,
    const lembed_text_options_t* options,
    lembed_text_embedding_t** out);

/* Load from GGUF file (llama.cpp backend) */
lembed_status_t lembed_text_embedding_create_from_gguf_path(
    const char* gguf_path,
    const lembed_text_options_t* options,
    lembed_text_embedding_t** out);

/* Load from GGUF model on HuggingFace (llama.cpp backend) */
lembed_status_t lembed_text_embedding_create_from_gguf_model(
    const char* repo,
    const char* filename,
    const lembed_text_options_t* options,
    lembed_text_embedding_t** out);

/* Worker auto-tuning for llama.cpp backend */
#include <libembedding/worker_autotune.h>

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#include "detail/model_loader_impl.hpp"
#include "detail/text_embedding_impl.hpp"
#endif

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_H */
