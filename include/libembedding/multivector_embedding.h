/*
 * libembedding - multivector_embedding.h
 * Multi-vector (late interaction, ColBERT) embedding C API, ONNX Runtime only.
 *
 * One vector per token instead of one per text. Scoring is done with
 * lembed_maxsim() / lembed_maxsim_batch() from similarity.h. This API embeds;
 * it does not index, sort or search.
 *
 * Auteur: David Orel
 * Version: 1.16.7
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_MULTIVECTOR_EMBEDDING_H
#define LIBEMBEDDING_MULTIVECTOR_EMBEDDING_H

#include <stdint.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle. */
typedef struct lembed_multivector lembed_multivector_t;

/* Which side of the retrieval the texts are: it selects the marker token
 * ([Q] / [D]) inserted after [CLS], the length limit, and whether the query is
 * padded with [MASK] (query expansion) or the punctuation of a document is
 * dropped. */
typedef enum {
    LEMBED_MV_DOCUMENT = 0,
    LEMBED_MV_QUERY = 1
} lembed_mv_mode_t;

typedef struct {
    /* Local directory holding the ONNX file (model.onnx, vespa_colbert.onnx or
     * onnx/model.onnx), tokenizer.json and, optionally, onnx_config.json (or
     * colbert_config.json). A path ending in ".gguf" is refused with
     * LEMBED_ERROR_UNSUPPORTED.
     *
     * When no such directory exists and the string is a registry entry (the
     * HuggingFace repo or the registry name, e.g.
     * "answerdotai/answerai-colbert-small-v1"), the model is downloaded into the
     * cache (cache_dir below) and loaded from there. */
    const char* model_dir;
    lembed_execution_provider_t provider;
    int device_id;
    int num_threads;            /* 0 = auto */
    int batch_size;             /* 0 = 32 */
    int max_tokens_per_batch;   /* batch x padded length cap; 0 = library default (512) */
    int query_maxlen;           /* > 0 overrides the model's query length (default 32); max 8192 */
    int doc_maxlen;             /* > 0 overrides the model's document length (default 300); max 8192 */
    int normalize;              /* 1 = L2-normalise every vector (idempotent, default) */
    int mask_punctuation;       /* documents only: 1 = drop skiplist tokens, 0 = keep,
                                 * -1 = follow the model (default) */
    int query_expansion;        /* 1 = pad queries with [MASK] up to the query length and keep
                                 * those vectors, 0 = no padding, -1 = on (default). The model's
                                 * do_query_expansion flag is not consulted: see the docs. */
    int attend_to_expansion;    /* with expansion: 1 = [MASK] pads are attended, 0 = masked out,
                                 * -1 = follow the model (default) */
    /* Registry models only (ignored for a local directory). */
    const char* cache_dir;      /* NULL = default cache */
    int show_download_progress; /* 1 = progress bar on stderr */
    int offline;                /* 1 = cache only, never download */
} lembed_multivector_options_t;

/* Ragged result: the vectors of document i are
 * data[offsets[i] * dim .. offsets[i + 1] * dim), row-major, `dim` floats each.
 * offsets has num_docs + 1 entries, offsets[0] == 0, offsets[num_docs] ==
 * total_vectors. Both buffers are owned by the caller (lembed_multivectors_free). */
typedef struct {
    float*   data;
    int64_t* offsets;
    int      num_docs;
    int      dim;
    int64_t  total_vectors;
} lembed_multivectors_t;

/* Defaults: CPU, auto threads, normalize = 1, query expansion on, everything else
 * "follow the model" (onnx_config.json, then the documented defaults). */
lembed_multivector_options_t lembed_multivector_options_default(void);

/* Loads the model from options->model_dir.
 *
 * Errors: LEMBED_ERROR_INVALID_ARGUMENT (NULL / empty arguments),
 * LEMBED_ERROR_UNSUPPORTED (.gguf path, unavailable provider, graph without the
 * projection the config declares), LEMBED_ERROR_IO (missing directory or file,
 * invalid onnx_config.json, marker that the vocabulary cannot resolve),
 * LEMBED_ERROR_DOWNLOAD (registry model that cannot be fetched). The
 * detail is in lembed_last_error(). */
lembed_status_t lembed_multivector_create(const lembed_multivector_options_t* options,
                                          lembed_multivector_t** out);

/* Embeds `n` texts. On success `out` is filled and must be released with
 * lembed_multivectors_free(); on failure it is zeroed. Thread-safe per handle
 * (calls are serialised). */
lembed_status_t lembed_multivector_embed(lembed_multivector_t* mv,
                                         const char* const* texts, int n,
                                         lembed_mv_mode_t mode,
                                         lembed_multivectors_t* out);

/* Idempotent: frees both buffers and zeroes the struct. NULL is accepted. */
void lembed_multivectors_free(lembed_multivectors_t* result);

/* Vector dimension (0 if mv is NULL). */
int lembed_multivector_dim(const lembed_multivector_t* mv);

/* Effective configuration. The pointer (and desc->name) is owned by the handle:
 * dangling after lembed_multivector_free(). desc->pooling is meaningless here
 * (0); desc->max_length is the document length, desc->dimension the vector size. */
const lembed_model_desc_t* lembed_multivector_desc(const lembed_multivector_t* mv);

/* "onnxruntime"; NULL if mv is NULL. Static string. */
const char* lembed_multivector_backend_name(const lembed_multivector_t* mv);

/* Usage counters (texts_embedded counts texts of both modes). */
void lembed_multivector_stats(const lembed_multivector_t* mv, lembed_stats_t* out);

void lembed_multivector_free(lembed_multivector_t* mv);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#include "detail/multivector_embedding_impl.hpp"
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_MULTIVECTOR_EMBEDDING_H */
