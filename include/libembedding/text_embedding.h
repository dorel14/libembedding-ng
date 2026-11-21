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

/* Versioned create preserving the v1.4.0 options ABI. */
lembed_status_t lembed_text_embedding_create_v2(
    const lembed_text_options_v2_t* options,
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

/**
 * @brief Matryoshka truncation of a dense result, in place.
 *
 * Keeps the first @p new_dim components of every row of @p emb, compacts the
 * rows so that @c emb->data becomes a [num_embeddings * new_dim] array, and
 * sets @c emb->dim to @p new_dim. Only meaningful for models trained with
 * Matryoshka Representation Learning (e.g. nomic-embed-text-v1.5,
 * mxbai-embed-large-v1, embeddinggemma-300m); on other models the prefix of a
 * vector carries no quality guarantee.
 *
 * The buffer is neither reallocated nor freed: @c emb->data keeps its address,
 * lembed_embeddings_free() still releases a library-allocated result, and a
 * caller-owned buffer is accepted as well.
 *
 * @param emb          Dense result (from lembed_text_embedding_embed,
 *                     lembed_image_embedding_embed or the caller). Modified
 *                     in place.
 * @param new_dim      Target dimension, 1 <= new_dim <= emb->dim. Equal to
 *                     emb->dim is valid (only renormalises, if asked).
 * @param renormalize  Non-zero: L2-renormalise each truncated row (an all-zero
 *                     row stays zero). Zero: plain prefix, values unchanged.
 * @return LEMBED_OK, or LEMBED_ERROR_INVALID_ARGUMENT when @p emb is NULL,
 *         emb->dim <= 0, emb->num_embeddings < 0, emb->data is NULL with rows,
 *         or @p new_dim is outside [1, emb->dim]. The detail is available from
 *         lembed_last_error(); @p emb is left untouched on error.
 */
lembed_status_t lembed_embeddings_truncate(
    lembed_embeddings_t* emb,
    int new_dim,
    int renormalize);

/* Introspection
 * NOTE: the desc/desc_v2 pointers reference memory owned by the context.
 * They stay valid until the next create_* call on the same context and are
 * dangling as soon as lembed_text_embedding_free() is called: copy the fields
 * you need, never store the pointer. model_name() returns an internal
 * std::string buffer with the same lifetime. */
const lembed_model_desc_t* lembed_text_embedding_desc(const lembed_text_embedding_t* ctx);
const lembed_model_desc_v2_t* lembed_text_embedding_desc_v2(const lembed_text_embedding_t* ctx);
const char* lembed_text_embedding_model_name(const lembed_text_embedding_t* ctx);
int lembed_text_embedding_max_length(const lembed_text_embedding_t* ctx);

/**
 * @brief Runtime that actually serves this context.
 *
 * "onnxruntime" (ONNX model) or "llama.cpp" (GGUF model). The runtime follows
 * the model format, never a parameter.
 *
 * @param ctx Text embedding context.
 * @return Static string, never freed; NULL if @p ctx is NULL.
 */
const char* lembed_text_embedding_backend_name(const lembed_text_embedding_t* ctx);

/**
 * @brief Execution provider actually running the model.
 *
 * ONNX Runtime: "CPU", "CUDA", "CoreML", "DirectML" or "TensorRT" (spelling of
 * lembed_execution_provider_name()). A provider other than CPU that cannot be
 * attached makes the create function fail with LEMBED_ERROR_UNSUPPORTED, so
 * this never reports a provider that is not in use. llama.cpp: "CPU", or "GPU"
 * when llama_n_gpu_layers != 0 and ggml has a GPU device (the shipped builds
 * have none).
 *
 * @param ctx Text embedding context.
 * @return Static string, never freed; NULL if @p ctx is NULL.
 */
const char* lembed_text_embedding_effective_provider(const lembed_text_embedding_t* ctx);

/* Runtime statistics */
void lembed_text_embedding_stats(const lembed_text_embedding_t* ctx, lembed_stats_t* out);
void lembed_text_embedding_stats_v2(const lembed_text_embedding_t* ctx, lembed_stats_v2_t* out);

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
