/*
 * libembedding - C/C++ Embedding Library (header-only on Linux/macOS, shared lib/DLL on Windows)
 * types.h - Core types, opaque handles, enums, output structures
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_TYPES_H
#define LIBEMBEDDING_TYPES_H

#include "config.h"
#include "error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Execution Provider
 * ========================================================================= */

typedef enum {
    LEMBED_PROVIDER_CPU = 0,
    LEMBED_PROVIDER_CUDA,
    LEMBED_PROVIDER_COREML,
    LEMBED_PROVIDER_DIRECTML,
    LEMBED_PROVIDER_TENSORRT,
    LEMBED_PROVIDER_LLAMACPP,
} lembed_execution_provider_t;

/* =========================================================================
 * Backend Type (auto-detected or explicit)
 * ========================================================================= */

typedef enum {
    LEMBED_BACKEND_ONNX = 0,
    LEMBED_BACKEND_LLAMACPP,
    LEMBED_BACKEND_AUTO,       /* Auto-detect based on model path/name */
} lembed_backend_t;

/* =========================================================================
 * Pooling Strategy
 * ========================================================================= */

typedef enum {
    LEMBED_POOLING_CLS = 0,
    LEMBED_POOLING_MEAN,
} lembed_pooling_t;

/* =========================================================================
 * Quantization Mode
 * ========================================================================= */

typedef enum {
    LEMBED_QUANTIZATION_NONE = 0,
    LEMBED_QUANTIZATION_STATIC,
    LEMBED_QUANTIZATION_DYNAMIC,
} lembed_quantization_t;

/* =========================================================================
 * Text Embedding Models (~40 models, matching fastembed-rs)
 * ========================================================================= */

typedef enum {
    LEMBED_TEXT_ALL_MINILM_L6_V2 = 0,
    LEMBED_TEXT_ALL_MINILM_L6_V2_Q,
    LEMBED_TEXT_ALL_MINILM_L12_V2,
    LEMBED_TEXT_ALL_MINILM_L12_V2_Q,
    LEMBED_TEXT_ALL_MPNET_BASE_V2,
    LEMBED_TEXT_BGE_BASE_EN_V15,
    LEMBED_TEXT_BGE_BASE_EN_V15_Q,
    LEMBED_TEXT_BGE_LARGE_EN_V15,
    LEMBED_TEXT_BGE_LARGE_EN_V15_Q,
    LEMBED_TEXT_BGE_SMALL_EN_V15,        /* default */
    LEMBED_TEXT_BGE_SMALL_EN_V15_Q,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V1,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V15,
    LEMBED_TEXT_NOMIC_EMBED_TEXT_V15_Q,
    LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2,
    LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2_Q,
    LEMBED_TEXT_PARAPHRASE_ML_MPNET_BASE_V2,
    LEMBED_TEXT_BGE_SMALL_ZH_V15,
    LEMBED_TEXT_BGE_LARGE_ZH_V15,
    LEMBED_TEXT_BGE_M3,
    LEMBED_TEXT_MODERNBERT_EMBED_LARGE,
    LEMBED_TEXT_MULTILINGUAL_E5_SMALL,
    LEMBED_TEXT_MULTILINGUAL_E5_BASE,
    LEMBED_TEXT_MULTILINGUAL_E5_LARGE,
    LEMBED_TEXT_MXBAI_EMBED_LARGE_V1,
    LEMBED_TEXT_MXBAI_EMBED_LARGE_V1_Q,
    LEMBED_TEXT_GTE_BASE_EN_V15,
    LEMBED_TEXT_GTE_BASE_EN_V15_Q,
    LEMBED_TEXT_GTE_LARGE_EN_V15,
    LEMBED_TEXT_GTE_LARGE_EN_V15_Q,
    LEMBED_TEXT_CLIP_VIT_B32,
    LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_CODE,
    LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_EN,
    LEMBED_TEXT_EMBEDDING_GEMMA_300M,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG_Q,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L,
    LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L_Q,
    LEMBED_TEXT_MODEL_COUNT,
} lembed_text_model_t;

#define LEMBED_TEXT_MODEL_DEFAULT LEMBED_TEXT_BGE_SMALL_EN_V15

/* =========================================================================
 * Sparse Embedding Models
 * ========================================================================= */

typedef enum {
    LEMBED_SPARSE_SPLADE_PP_V1 = 0,     /* default */
    LEMBED_SPARSE_BGE_M3,
    LEMBED_SPARSE_MODEL_COUNT,
} lembed_sparse_model_t;

#define LEMBED_SPARSE_MODEL_DEFAULT LEMBED_SPARSE_SPLADE_PP_V1

/* =========================================================================
 * Image Embedding Models
 * ========================================================================= */

typedef enum {
    LEMBED_IMAGE_CLIP_VIT_B32 = 0,       /* default */
    LEMBED_IMAGE_RESNET50,
    LEMBED_IMAGE_UNICOM_VIT_B16,
    LEMBED_IMAGE_UNICOM_VIT_B32,
    LEMBED_IMAGE_NOMIC_EMBED_VISION_V15,
    LEMBED_IMAGE_CLIP_VIT_B32_QUANTIZED,  /* INT8 quantized, 4x smaller */
    LEMBED_IMAGE_MODEL_COUNT,
} lembed_image_model_t;

#define LEMBED_IMAGE_MODEL_DEFAULT LEMBED_IMAGE_CLIP_VIT_B32

/* =========================================================================
 * Reranker Models
 * ========================================================================= */

typedef enum {
    LEMBED_RERANKER_BGE_BASE = 0,        /* default */
    LEMBED_RERANKER_BGE_V2_M3,
    LEMBED_RERANKER_JINA_V1_TURBO_EN,
    LEMBED_RERANKER_JINA_V2_BASE_MULTILINGUAL,
    LEMBED_RERANKER_JINA_V1_TURBO_EN_QUANTIZED,  /* INT8 quantized, 4x smaller */
    LEMBED_RERANKER_MODEL_COUNT,
} lembed_reranker_model_t;

#define LEMBED_RERANKER_MODEL_DEFAULT LEMBED_RERANKER_JINA_V1_TURBO_EN_QUANTIZED

/* =========================================================================
 * Model Info (returned by registry queries)
 * ========================================================================= */

typedef struct {
    const char* model_name;       /* Human-readable name, e.g. "BAAI/bge-small-en-v1.5" */
    const char* model_code;       /* HuggingFace repo, e.g. "Xenova/bge-small-en-v1.5" */
    const char* model_file;       /* ONNX file path within repo */
    const char* description;
    int         dim;              /* Embedding dimension (0 for sparse/reranker) */
    int         max_tokens;       /* Default max token length */
    int         pooling;          /* lembed_pooling_t */
    int         quantization;     /* lembed_quantization_t */
} lembed_model_info_t;

/* =========================================================================
 * Model Descriptor (runtime introspection)
 * Returned by lembed_*_desc() to expose the effective configuration
 * of an already-created embedding context.
 * ========================================================================= */
typedef struct {
    const char*                name;          /* model name or local path */
    int                        dimension;       /* embedding dimension */
    int                        max_length;      /* effective max token length */
    int                        pooling;         /* lembed_pooling_t */
    int                        num_threads;     /* threads configured */
    int                        batch_size;      /* batch_size configured */
    lembed_execution_provider_t provider;       /* execution provider in use */
    int                        device_id;       /* device id in use */
} lembed_model_desc_t;

/* =========================================================================
 * Runtime Statistics
 * Returned by lembed_*_stats() to expose usage counters for a context.
 * ========================================================================= */
typedef struct {
    uint64_t texts_embedded;    /* total texts processed since context creation */
    uint64_t batches_run;       /* total ONNX inference batches executed */
    double   avg_latency_ms;    /* average wall-clock time per embed() call (ms) */
} lembed_stats_t;

/* =========================================================================
 * Opaque Handles
 * ========================================================================= */

typedef struct lembed_text_embedding    lembed_text_embedding_t;
typedef struct lembed_sparse_embedding  lembed_sparse_embedding_ctx_t;
typedef struct lembed_image_embedding   lembed_image_embedding_t;
typedef struct lembed_reranker          lembed_reranker_t;

/* =========================================================================
 * Output Structures
 * ========================================================================= */

/* Dense embedding result (flat array: num_embeddings * dim) */
typedef struct {
    float*  data;
    int     num_embeddings;
    int     dim;
} lembed_embeddings_t;

/* Sparse embedding for a single text */
typedef struct {
    int32_t* indices;
    float*   values;
    int      length;
} lembed_sparse_embedding_t;

/* Batch of sparse embeddings */
typedef struct {
    lembed_sparse_embedding_t* items;
    int count;
} lembed_sparse_embeddings_t;

/* Rerank result for a single document */
typedef struct {
    int   index;
    float score;
} lembed_rerank_result_t;

/* Batch of rerank results */
typedef struct {
    lembed_rerank_result_t* items;
    int count;
} lembed_rerank_results_t;

/* =========================================================================
 * Init Options
 * ========================================================================= */

typedef struct {
    lembed_text_model_t         model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;    /* NULL = default */
    int                         max_length;   /* 0 = model default */
    int                         num_threads;  /* 0 = auto */
    int                         show_download_progress;
    int                         batch_size;   /* 0 = default (256) */
    int                         offline;      /* 1 = skip downloads, use cache only */
    int                         pooling;      /* lembed_pooling_t, for local models without config.json */
    int                         dim;          /* embedding dim, for local models without config.json */
    /* llama.cpp specific options */
    int                         llama_n_ctx;  /* context size (0 = model default) */
    int                         llama_n_gpu_layers; /* GPU layers for llama.cpp (-1 = all, 0 = CPU) */
    int                         llama_verbose; /* 1 = enable llama.cpp logging */
    int                         llama_n_batch; /* max tokens per llama_encode() (0 = model default) */
    int                         auto_workers;   /* 1 = auto-detect optimal workers/sessions */
    int                         cache_size;     /* 0 = disabled, >0 = LRU cache capacity */
    /* Backend selection */
    lembed_backend_t            backend;       /* ONNX, LLAMACPP, or AUTO */
    /* ONNX batching strategy */
    int                         batch_strategy; /* lembed_batch_strategy_t (ONNX only) */
} lembed_text_options_t;

/* Batching strategy for ONNX backend */
typedef enum {
    LEMBED_BATCH_SEQUENTIAL = 0,    /* One text at a time (no batching) */
    LEMBED_BATCH_FIXED = 1,         /* Fixed-size batches (naive, may pad heavily) */
    LEMBED_BATCH_LENGTH_BUCKET = 2, /* Sort by length, then batch (minimizes padding) */
} lembed_batch_strategy_t;

typedef struct {
    lembed_sparse_model_t       model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    int                         top_k;        /* max terms to keep (0 = all) */
    float                       min_weight;   /* pruning threshold */
    int                         storage_format; /* 0=dict, 1=CSR, 2=numpy */
} lembed_sparse_options_t;

typedef struct {
    lembed_image_model_t        model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    int                         dim;          /* for local models without config.json */
} lembed_image_options_t;

typedef struct {
    lembed_reranker_model_t     model;
    lembed_execution_provider_t provider;
    int                         device_id;
    const char*                 cache_dir;
    int                         max_length;
    int                         num_threads;
    int                         show_download_progress;
    int                         batch_size;
    int                         offline;
    /* Backend selection */
    lembed_backend_t            backend;       /* ONNX, LLAMACPP, or AUTO */
    /* llama.cpp specific options */
    int                         llama_n_ctx;  /* context size (0 = model default) */
    int                         llama_n_gpu_layers; /* GPU layers for llama.cpp (-1 = all, 0 = CPU) */
    int                         llama_verbose; /* 1 = enable llama.cpp logging */
    int                         llama_n_batch; /* max tokens per llama_encode() (0 = model default) */
    int                         auto_workers;   /* 1 = auto-detect optimal workers/sessions */
    int                         cache_size;     /* 0 = disabled, >0 = LRU cache capacity */
} lembed_reranker_options_t;

/* User-defined model (bring-your-own ONNX) */
typedef struct {
    const unsigned char* onnx_data;
    size_t               onnx_data_size;
    const unsigned char* tokenizer_json;
    size_t               tokenizer_json_size;
    const unsigned char* config_json;       /* optional, may be NULL */
    size_t               config_json_size;
    lembed_pooling_t     pooling;
    int                  dim;
    int                  max_length;
} lembed_user_defined_model_t;

/* =========================================================================
 * Memory Free Functions (forward declarations)
 * ========================================================================= */

void lembed_embeddings_free(lembed_embeddings_t* result);
void lembed_sparse_embeddings_free(lembed_sparse_embeddings_t* result);
void lembed_rerank_results_free(lembed_rerank_results_t* result);

#ifdef __cplusplus
}
#endif

/* ---- Implementation of defaults and free functions ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_TYPES_IMPL
#define LIBEMBEDDING_TYPES_IMPL

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

lembed_text_options_t lembed_text_options_default(void) {
    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_TEXT_MODEL_DEFAULT;
    opts.provider = LEMBED_PROVIDER_CPU;
    opts.show_download_progress = 1;
    opts.batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    opts.pooling = LEMBED_POOLING_MEAN;
    opts.llama_n_gpu_layers = 0;
    opts.batch_strategy = LEMBED_BATCH_LENGTH_BUCKET;
    opts.backend = LEMBED_BACKEND_AUTO;  /* Auto-detect backend by default */
    opts.auto_workers = 0;                /* Manual worker count by default */
    opts.cache_size = 0;                  /* No cache by default */
    return opts;
}

lembed_sparse_options_t lembed_sparse_options_default(void) {
    lembed_sparse_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_SPARSE_MODEL_DEFAULT;
    opts.provider = LEMBED_PROVIDER_CPU;
    opts.show_download_progress = 1;
    opts.batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    return opts;
}

lembed_image_options_t lembed_image_options_default(void) {
    lembed_image_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_IMAGE_MODEL_DEFAULT;
    opts.provider = LEMBED_PROVIDER_CPU;
    opts.show_download_progress = 1;
    opts.batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    return opts;
}

lembed_reranker_options_t lembed_reranker_options_default(void) {
    lembed_reranker_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_RERANKER_MODEL_DEFAULT;
    opts.provider = LEMBED_PROVIDER_CPU;
    opts.show_download_progress = 1;
    opts.batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    opts.backend = LEMBED_BACKEND_AUTO;
    opts.llama_n_gpu_layers = 0;
    opts.auto_workers = 0;
    opts.cache_size = 0;
    return opts;
}

void lembed_embeddings_free(lembed_embeddings_t* result) {
    if (result) {
        free(result->data);
        result->data = NULL;
        result->num_embeddings = 0;
        result->dim = 0;
    }
}

void lembed_sparse_embeddings_free(lembed_sparse_embeddings_t* result) {
    if (result) {
        for (int i = 0; i < result->count; i++) {
            free(result->items[i].indices);
            free(result->items[i].values);
        }
        free(result->items);
        result->items = NULL;
        result->count = 0;
    }
}

void lembed_rerank_results_free(lembed_rerank_results_t* result) {
    if (result) {
        free(result->items);
        result->items = NULL;
        result->count = 0;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_TYPES_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_TYPES_H */




