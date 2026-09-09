/*
 * libembedding - C Header-Only Embedding Library
 * model_registry.h - Model registry query API
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_MODEL_REGISTRY_H
#define LIBEMBEDDING_MODEL_REGISTRY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Query model info by enum value */
lembed_status_t lembed_get_text_model_info(lembed_text_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_sparse_model_info(lembed_sparse_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_image_model_info(lembed_image_model_t model, lembed_model_info_t* out);
lembed_status_t lembed_get_reranker_model_info(lembed_reranker_model_t model, lembed_model_info_t* out);

/* List all supported models (returns pointer to static data, do not free) */
lembed_status_t lembed_list_text_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_sparse_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_image_models(const lembed_model_info_t** out, int* count);
lembed_status_t lembed_list_reranker_models(const lembed_model_info_t** out, int* count);

/* Lookup model enum from HuggingFace model code string, returns -1 if not found */
int lembed_find_text_model_by_code(const char* model_code);
int lembed_find_sparse_model_by_code(const char* model_code);
int lembed_find_reranker_model_by_code(const char* model_code);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_MODEL_REGISTRY_IMPL
#define LIBEMBEDDING_MODEL_REGISTRY_IMPL

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Text Model Registry Data
 * ========================================================================= */

static const lembed_model_info_t lembed__text_models[] = {
    /* LEMBED_TEXT_ALL_MINILM_L6_V2 */
    { "sentence-transformers/all-MiniLM-L6-v2", "Qdrant/all-MiniLM-L6-v2-onnx",
      "model.onnx", "Sentence Transformer model, MiniLM-L6-v2",
      384, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_ALL_MINILM_L6_V2_Q */
    { "sentence-transformers/all-MiniLM-L6-v2", "Xenova/all-MiniLM-L6-v2",
      "onnx/model_quantized.onnx", "Quantized Sentence Transformer, MiniLM-L6-v2",
      384, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_ALL_MINILM_L12_V2 */
    { "sentence-transformers/all-MiniLM-L12-v2", "Xenova/all-MiniLM-L12-v2",
      "onnx/model.onnx", "Sentence Transformer model, MiniLM-L12-v2",
      384, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_ALL_MINILM_L12_V2_Q */
    { "sentence-transformers/all-MiniLM-L12-v2", "Xenova/all-MiniLM-L12-v2",
      "onnx/model_quantized.onnx", "Quantized Sentence Transformer, MiniLM-L12-v2",
      384, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_ALL_MPNET_BASE_V2 */
    { "sentence-transformers/all-mpnet-base-v2", "Xenova/all-mpnet-base-v2",
      "onnx/model.onnx", "Sentence Transformer model, mpnet-base-v2",
      768, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_BGE_BASE_EN_V15 */
    { "BAAI/bge-base-en-v1.5", "Xenova/bge-base-en-v1.5",
      "onnx/model.onnx", "v1.5 release of the base English model",
      768, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_BGE_BASE_EN_V15_Q */
    { "BAAI/bge-base-en-v1.5", "Qdrant/bge-base-en-v1.5-onnx-Q",
      "model_optimized.onnx", "Quantized v1.5 base English model",
      768, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_STATIC },
    /* LEMBED_TEXT_BGE_LARGE_EN_V15 */
    { "BAAI/bge-large-en-v1.5", "Xenova/bge-large-en-v1.5",
      "onnx/model.onnx", "v1.5 release of the large English model",
      1024, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_BGE_LARGE_EN_V15_Q */
    { "BAAI/bge-large-en-v1.5", "Qdrant/bge-large-en-v1.5-onnx-Q",
      "model_optimized.onnx", "Quantized v1.5 large English model",
      1024, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_STATIC },
    /* LEMBED_TEXT_BGE_SMALL_EN_V15 (default) */
    { "BAAI/bge-small-en-v1.5", "Xenova/bge-small-en-v1.5",
      "onnx/model.onnx", "v1.5 release of the fast and default English model",
      384, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_BGE_SMALL_EN_V15_Q */
    { "BAAI/bge-small-en-v1.5", "Qdrant/bge-small-en-v1.5-onnx-Q",
      "model_optimized.onnx", "Quantized v1.5 fast and default English model",
      384, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_STATIC },
    /* LEMBED_TEXT_NOMIC_EMBED_TEXT_V1 */
    { "nomic-ai/nomic-embed-text-v1", "nomic-ai/nomic-embed-text-v1",
      "onnx/model.onnx", "8192 context length English model",
      768, 8192, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_NOMIC_EMBED_TEXT_V15 */
    { "nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5",
      "onnx/model.onnx", "v1.5 8192 context length English model",
      768, 8192, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_NOMIC_EMBED_TEXT_V15_Q */
    { "nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5",
      "onnx/model_quantized.onnx", "Quantized v1.5 8192 context length English model",
      768, 8192, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2 */
    { "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
      "onnx/model.onnx", "Multi-lingual model",
      384, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2_Q */
    { "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
      "model_optimized.onnx", "Quantized multi-lingual model",
      384, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_STATIC },
    /* LEMBED_TEXT_PARAPHRASE_ML_MPNET_BASE_V2 */
    { "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
      "Xenova/paraphrase-multilingual-mpnet-base-v2",
      "onnx/model.onnx", "Sentence-transformers model for clustering or semantic search",
      768, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_BGE_SMALL_ZH_V15 */
    { "BAAI/bge-small-zh-v1.5", "Xenova/bge-small-zh-v1.5",
      "onnx/model.onnx", "v1.5 release of the small Chinese model",
      512, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_BGE_LARGE_ZH_V15 */
    { "BAAI/bge-large-zh-v1.5", "Xenova/bge-large-zh-v1.5",
      "onnx/model.onnx", "v1.5 release of the large Chinese model",
      1024, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_BGE_M3 */
    { "BAAI/bge-m3", "BAAI/bge-m3",
      "onnx/model.onnx", "Multilingual M3, 8192 context, 100+ languages",
      1024, 8192, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_MODERNBERT_EMBED_LARGE */
    { "lightonai/modernbert-embed-large", "lightonai/modernbert-embed-large",
      "onnx/model.onnx", "Large model of ModernBert Text Embeddings",
      1024, 8192, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_MULTILINGUAL_E5_SMALL */
    { "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-small",
      "onnx/model.onnx", "Small multilingual E5 Text Embeddings",
      384, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_MULTILINGUAL_E5_BASE */
    { "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-base",
      "onnx/model.onnx", "Base multilingual E5 Text Embeddings",
      768, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_MULTILINGUAL_E5_LARGE */
    { "intfloat/multilingual-e5-large", "Qdrant/multilingual-e5-large-onnx",
      "model.onnx", "Large multilingual E5 Text Embeddings",
      1024, 512, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_MXBAI_EMBED_LARGE_V1 */
    { "mixedbread-ai/mxbai-embed-large-v1", "mixedbread-ai/mxbai-embed-large-v1",
      "onnx/model.onnx", "Large English embedding model from MixedBread.ai",
      1024, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_MXBAI_EMBED_LARGE_V1_Q */
    { "mixedbread-ai/mxbai-embed-large-v1", "mixedbread-ai/mxbai-embed-large-v1",
      "onnx/model_quantized.onnx", "Quantized large English model from MixedBread.ai",
      1024, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_GTE_BASE_EN_V15 */
    { "Alibaba-NLP/gte-base-en-v1.5", "Alibaba-NLP/gte-base-en-v1.5",
      "onnx/model.onnx", "Base English embedding model from Alibaba",
      768, 8192, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_GTE_BASE_EN_V15_Q */
    { "Alibaba-NLP/gte-base-en-v1.5", "Alibaba-NLP/gte-base-en-v1.5",
      "onnx/model_quantized.onnx", "Quantized base English model from Alibaba",
      768, 8192, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_GTE_LARGE_EN_V15 */
    { "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-large-en-v1.5",
      "onnx/model.onnx", "Large English embedding model from Alibaba",
      1024, 8192, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_GTE_LARGE_EN_V15_Q */
    { "Alibaba-NLP/gte-large-en-v1.5", "Alibaba-NLP/gte-large-en-v1.5",
      "onnx/model_quantized.onnx", "Quantized large English model from Alibaba",
      1024, 8192, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_CLIP_VIT_B32 */
    { "openai/clip-vit-base-patch32", "Qdrant/clip-ViT-B-32-text",
      "model.onnx", "CLIP text encoder based on ViT-B/32",
      512, 77, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_CODE */
    { "jinaai/jina-embeddings-v2-base-code", "jinaai/jina-embeddings-v2-base-code",
      "onnx/model.onnx", "Jina embeddings v2 base code",
      768, 8192, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_JINA_EMBEDDINGS_V2_BASE_EN */
    { "jinaai/jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en",
      "model.onnx", "Jina embeddings v2 base English",
      768, 8192, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_EMBEDDING_GEMMA_300M */
    { "google/embeddinggemma-300m", "onnx-community/embeddinggemma-300m-ONNX",
      "onnx/model.onnx", "EmbeddingGemma 300M parameter model from Google",
      768, 2048, LEMBED_POOLING_MEAN, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS */
    { "snowflake/snowflake-arctic-embed-xs", "snowflake/snowflake-arctic-embed-xs",
      "onnx/model.onnx", "Snowflake Arctic embed model, xs",
      384, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS_Q */
    { "snowflake/snowflake-arctic-embed-xs", "snowflake/snowflake-arctic-embed-xs",
      "onnx/model_quantized.onnx", "Quantized Snowflake Arctic embed, xs",
      384, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S */
    { "snowflake/snowflake-arctic-embed-s", "snowflake/snowflake-arctic-embed-s",
      "onnx/model.onnx", "Snowflake Arctic embed model, small",
      384, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_S_Q */
    { "snowflake/snowflake-arctic-embed-s", "snowflake/snowflake-arctic-embed-s",
      "onnx/model_quantized.onnx", "Quantized Snowflake Arctic embed, small",
      384, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M */
    { "snowflake/snowflake-arctic-embed-m", "Snowflake/snowflake-arctic-embed-m",
      "onnx/model.onnx", "Snowflake Arctic embed model, medium",
      768, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_Q */
    { "snowflake/snowflake-arctic-embed-m", "Snowflake/snowflake-arctic-embed-m",
      "onnx/model_quantized.onnx", "Quantized Snowflake Arctic embed, medium",
      768, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG */
    { "snowflake/snowflake-arctic-embed-m-long", "snowflake/snowflake-arctic-embed-m-long",
      "onnx/model.onnx", "Snowflake Arctic embed, medium with 2048 context",
      768, 2048, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M_LONG_Q */
    { "snowflake/snowflake-arctic-embed-m-long", "snowflake/snowflake-arctic-embed-m-long",
      "onnx/model_quantized.onnx", "Quantized Snowflake Arctic embed, medium 2048 ctx",
      768, 2048, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L */
    { "snowflake/snowflake-arctic-embed-l", "snowflake/snowflake-arctic-embed-l",
      "onnx/model.onnx", "Snowflake Arctic embed model, large",
      1024, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L_Q */
    { "snowflake/snowflake-arctic-embed-l", "snowflake/snowflake-arctic-embed-l",
      "onnx/model_quantized.onnx", "Quantized Snowflake Arctic embed, large",
      1024, 512, LEMBED_POOLING_CLS, LEMBED_QUANTIZATION_DYNAMIC },
};

/* =========================================================================
 * Sparse Model Registry Data
 * ========================================================================= */

static const lembed_model_info_t lembed__sparse_models[] = {
    /* LEMBED_SPARSE_SPLADE_PP_V1 */
    { "prithivida/Splade_PP_en_v1", "Qdrant/Splade_PP_en_v1",
      "model.onnx", "SPLADE++ sparse vector model, v1",
      0, 512, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_SPARSE_BGE_M3 */
    { "BAAI/bge-m3", "BAAI/bge-m3",
      "onnx/model.onnx", "BGE-M3 sparse embedding, 8192 context, 100+ languages",
      0, 8192, 0, LEMBED_QUANTIZATION_NONE },
};

/* =========================================================================
 * Image Model Registry Data
 * ========================================================================= */

static const lembed_model_info_t lembed__image_models[] = {
    /* LEMBED_IMAGE_CLIP_VIT_B32 */
    { "openai/clip-vit-base-patch32", "Qdrant/clip-ViT-B-32-vision",
      "model.onnx", "CLIP vision encoder based on ViT-B/32",
      512, 0, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_IMAGE_RESNET50 */
    { "microsoft/resnet-50", "Qdrant/resnet50-onnx",
      "model.onnx", "ResNet-50 image embedding model",
      2048, 0, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_IMAGE_UNICOM_VIT_B16 */
    { "open-metric-learning/unicom-vit-b-16", "Qdrant/Unicom-ViT-B-16",
      "model.onnx", "Unicom ViT-B-16 from open-metric-learning",
      768, 0, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_IMAGE_UNICOM_VIT_B32 */
    { "open-metric-learning/unicom-vit-b-32", "Qdrant/Unicom-ViT-B-32",
      "model.onnx", "Unicom ViT-B-32 from open-metric-learning",
      512, 0, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_IMAGE_NOMIC_EMBED_VISION_V15 */
    { "nomic-ai/nomic-embed-vision-v1.5", "nomic-ai/nomic-embed-vision-v1.5",
      "onnx/model.onnx", "Nomic embed vision v1.5",
      768, 0, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_IMAGE_CLIP_VIT_B32_QUANTIZED */
    { "openai/clip-vit-base-patch32-quantized", "Xenova/clip-vit-base-patch32",
      "onnx/vision_model_int8.onnx", "CLIP vision encoder INT8 quantized",
      512, 0, 0, LEMBED_QUANTIZATION_STATIC },
};

/* =========================================================================
 * Reranker Model Registry Data
 * ========================================================================= */

static const lembed_model_info_t lembed__reranker_models[] = {
    /* LEMBED_RERANKER_BGE_BASE */
    { "BAAI/bge-reranker-base", "BAAI/bge-reranker-base",
      "onnx/model.onnx", "Reranker model for English and Chinese",
      0, 512, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_RERANKER_BGE_V2_M3 */
    { "BAAI/bge-reranker-v2-m3", "rozgo/bge-reranker-v2-m3",
      "model.onnx", "Multilingual reranker model",
      0, 512, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_RERANKER_JINA_V1_TURBO_EN */
    { "jinaai/jina-reranker-v1-turbo-en", "jinaai/jina-reranker-v1-turbo-en",
      "onnx/model.onnx", "Jina reranker v1 turbo English",
      0, 8192, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_RERANKER_JINA_V2_BASE_MULTILINGUAL */
    { "jinaai/jina-reranker-v2-base-multilingual", "jinaai/jina-reranker-v2-base-multilingual",
      "onnx/model.onnx", "Jina reranker v2 base multilingual",
      0, 8192, 0, LEMBED_QUANTIZATION_NONE },
    /* LEMBED_RERANKER_JINA_V1_TURBO_EN_QUANTIZED */
    { "jinaai/jina-reranker-v1-turbo-en-quantized", "jinaai/jina-reranker-v1-turbo-en",
      "onnx/model_quantized.onnx", "Jina reranker v1 turbo English (INT8 quantized)",
      0, 8192, 0, LEMBED_QUANTIZATION_STATIC },
};

/* =========================================================================
 * Additional files lookup (models that need extra files beyond the ONNX model)
 * ========================================================================= */

typedef struct {
    int         model_enum;
    int         model_type;  /* 0=text, 1=sparse, 2=image, 3=reranker */
    const char* files[4];    /* NULL-terminated list */
} lembed__additional_files_entry_t;

#define LEMBED__MODEL_TYPE_TEXT     0
#define LEMBED__MODEL_TYPE_SPARSE   1
#ifndef LIBEMBEDDING_NO_IMAGE
#define LEMBED__MODEL_TYPE_IMAGE    2
#endif
#define LEMBED__MODEL_TYPE_RERANKER 3

static const lembed__additional_files_entry_t lembed__additional_files[] = {
    { LEMBED_TEXT_BGE_M3,            LEMBED__MODEL_TYPE_TEXT,     { "onnx/model.onnx_data", "onnx/Constant_7_attr__value", NULL } },
    { LEMBED_TEXT_MULTILINGUAL_E5_LARGE, LEMBED__MODEL_TYPE_TEXT, { "model.onnx_data", NULL } },
    { LEMBED_TEXT_EMBEDDING_GEMMA_300M,  LEMBED__MODEL_TYPE_TEXT, { "onnx/model.onnx_data", NULL } },
    { LEMBED_SPARSE_BGE_M3,         LEMBED__MODEL_TYPE_SPARSE,   { "onnx/model.onnx_data", "onnx/Constant_7_attr__value", NULL } },
    { LEMBED_RERANKER_BGE_V2_M3,    LEMBED__MODEL_TYPE_RERANKER, { "model.onnx.data", NULL } },
};

static const int lembed__additional_files_count =
    (int)(sizeof(lembed__additional_files) / sizeof(lembed__additional_files[0]));

/* =========================================================================
 * Query Functions
 * ========================================================================= */

lembed_status_t lembed_get_text_model_info(lembed_text_model_t model, lembed_model_info_t* out) {
    if ((int)model < 0 || (int)model >= LEMBED_TEXT_MODEL_COUNT || !out)
        return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__text_models[(int)model];
    return LEMBED_OK;
}

lembed_status_t lembed_get_sparse_model_info(lembed_sparse_model_t model, lembed_model_info_t* out) {
    if ((int)model < 0 || (int)model >= LEMBED_SPARSE_MODEL_COUNT || !out)
        return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__sparse_models[(int)model];
    return LEMBED_OK;
}

lembed_status_t lembed_get_image_model_info(lembed_image_model_t model, lembed_model_info_t* out) {
    if ((int)model < 0 || (int)model >= LEMBED_IMAGE_MODEL_COUNT || !out)
        return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__image_models[(int)model];
    return LEMBED_OK;
}

lembed_status_t lembed_get_reranker_model_info(lembed_reranker_model_t model, lembed_model_info_t* out) {
    if ((int)model < 0 || (int)model >= LEMBED_RERANKER_MODEL_COUNT || !out)
        return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__reranker_models[(int)model];
    return LEMBED_OK;
}

lembed_status_t lembed_list_text_models(const lembed_model_info_t** out, int* count) {
    if (!out || !count) return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__text_models;
    *count = LEMBED_TEXT_MODEL_COUNT;
    return LEMBED_OK;
}

lembed_status_t lembed_list_sparse_models(const lembed_model_info_t** out, int* count) {
    if (!out || !count) return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__sparse_models;
    *count = LEMBED_SPARSE_MODEL_COUNT;
    return LEMBED_OK;
}

lembed_status_t lembed_list_image_models(const lembed_model_info_t** out, int* count) {
    if (!out || !count) return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__image_models;
    *count = LEMBED_IMAGE_MODEL_COUNT;
    return LEMBED_OK;
}

lembed_status_t lembed_list_reranker_models(const lembed_model_info_t** out, int* count) {
    if (!out || !count) return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__reranker_models;
    *count = LEMBED_RERANKER_MODEL_COUNT;
    return LEMBED_OK;
}

int lembed_find_text_model_by_code(const char* model_code) {
    if (!model_code) return -1;
    for (int i = 0; i < LEMBED_TEXT_MODEL_COUNT; i++) {
        if (strcmp(lembed__text_models[i].model_code, model_code) == 0) return i;
    }
    return -1;
}

int lembed_find_sparse_model_by_code(const char* model_code) {
    if (!model_code) return -1;
    for (int i = 0; i < LEMBED_SPARSE_MODEL_COUNT; i++) {
        if (strcmp(lembed__sparse_models[i].model_code, model_code) == 0) return i;
    }
    return -1;
}

int lembed_find_reranker_model_by_code(const char* model_code) {
    if (!model_code) return -1;
    for (int i = 0; i < LEMBED_RERANKER_MODEL_COUNT; i++) {
        if (strcmp(lembed__reranker_models[i].model_code, model_code) == 0) return i;
    }
    return -1;
}

int lembed_find_image_model_by_code(const char* model_code) {
    if (!model_code) return -1;
    for (int i = 0; i < LEMBED_IMAGE_MODEL_COUNT; i++) {
        if (strcmp(lembed__image_models[i].model_code, model_code) == 0) return i;
    }
    return -1;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_MODEL_REGISTRY_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_MODEL_REGISTRY_H */




