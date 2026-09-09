/*
 * libembedding - detail/gguf_registry_impl.hpp
 * GGUF model registry implementation
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_REGISTRY_IMPL_HPP
#define LIBEMBEDDING_GGUF_REGISTRY_IMPL_HPP

#include "../gguf_registry.h"
#include <cstring>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GGUF Model Registry Data
 *
 * Curated list of recommended GGUF models for the llama.cpp backend.
 * Models are ordered by recommendation priority (best first).
 * ========================================================================= */
static const lembed_gguf_model_info_t lembed__gguf_models[] = {
    /* Snowflake Arctic Embed XS — best quality/throughput tradeoff */
    { "Snowflake-XS-Q4",
      "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-xs-gguf/resolve/main/snowflake-arctic-embed-xs--Q4_K_M.GGUF",
      "snowflake/snowflake-arctic-embed-xs",
      "Best overall: fast (6 layers) with excellent retrieval quality (MTEB 50.2)",
      384, 22, 20, 50.2f, 4 },

    /* MiniLM-L6-v2 — fastest small model */
    { "MiniLM-L6-Q4",
      "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf",
      "sentence-transformers/all-MiniLM-L6-v2",
      "Fastest small model (6 layers), good for throughput-critical apps",
      384, 22, 20, 41.9f, 4 },

    /* MiniLM-L6-v2 Q8 — slightly better quality than Q4 */
    { "MiniLM-L6-Q8",
      "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf",
      "sentence-transformers/all-MiniLM-L6-v2",
      "Fast with better quality than Q4 (8-bit quantization)",
      384, 22, 24, 41.9f, 4 },

    /* Snowflake Arctic Embed S — higher quality, slower */
    { "Snowflake-S-Q4",
      "https://huggingface.co/ChristianAzinn/snowflake-arctic-embed-s-gguf/resolve/main/snowflake-arctic-embed-s--Q4_K_M.GGUF",
      "snowflake/snowflake-arctic-embed-s",
      "Higher quality (MTEB 52.0) for quality-critical applications",
      384, 33, 28, 52.0f, 4 },

    /* E5-small-v2 — multilingual support */
    { "E5-small-Q4",
      "https://huggingface.co/ChristianAzinn/e5-small-v2-gguf/resolve/main/e5-small-v2.Q4_K_M.gguf",
      "intfloat/e5-small-v2",
      "Multilingual embedding model, good cross-lingual retrieval",
      384, 33, 28, 46.0f, 4 },

    /* GIST-small — quality-focused */
    { "GIST-small-Q4",
      "https://huggingface.co/ChristianAzinn/gist-small-embedding-v0-gguf/resolve/main/gist-small-embedding-v0.Q4_K_M.gguf",
      "avsolatorio/GIST-small-Embedding-v0",
      "Quality-focused small model, good for semantic search",
      384, 33, 28, 48.0f, 4 },

    /* =========================================================================
     * Reranker GGUF models (llama.cpp backend)
     * ========================================================================= */

    /* default: BGE-Reranker-v2-M3 — multilingual cross-encoder reranker GGUF */
    { "BGE-Reranker-v2-M3",
      "https://huggingface.co/smarttasks/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf",
      "BAAI/bge-reranker-v2-m3",
      "Default multilingual reranker GGUF (Q4_K_M), 438 MB, validated by SmartTasks",
      0, 560, 438, 0.0f, 4 },

    /* fast: BGE-Reranker-Base — lightweight cross-encoder reranker GGUF */
    { "BGE-Reranker-Base",
      "https://huggingface.co/sinjab/bge-reranker-base-Q4_K_M-GGUF/resolve/main/bge-reranker-base-Q4_K_M.gguf",
      "BAAI/bge-reranker-base",
      "Fast lightweight reranker GGUF (Q4_K_M), 219 MB, English/Chinese",
      0, 110, 219, 0.0f, 4 },

    /* quality: BGE-Reranker-Large — high-accuracy cross-encoder reranker GGUF */
    { "BGE-Reranker-Large",
      "https://huggingface.co/sinjab/bge-reranker-large-Q4_K_M-GGUF/resolve/main/bge-reranker-large-Q4_K_M.gguf",
      "BAAI/bge-reranker-large",
      "High-accuracy reranker GGUF (Q4_K_M), 407 MB, English/Chinese",
      0, 560, 407, 0.0f, 4 },

    /* qwen: Qwen3-Reranker-0.6B — Qwen3 reranker GGUF */
    { "Qwen3-Reranker-0.6B",
      "https://huggingface.co/sinjab/Qwen3-Reranker-0.6B-Q4_K_M-GGUF/resolve/main/Qwen3-Reranker-0.6B-Q4_K_M.gguf",
      "Qwen/Qwen3-Reranker-0.6B",
      "Qwen3 reranker GGUF (Q4_K_M), ~396 MB, modern architecture",
      0, 600, 396, 0.0f, 4 },

    /* edge: Jina-Reranker-v1-Turbo-En — tiny edge reranker GGUF */
    { "Jina-Reranker-v1-Turbo-En",
      "https://huggingface.co/sinjab/jina-reranker-v1-turbo-en-Q8_0-GGUF/resolve/main/jina-reranker-v1-turbo-en-Q8_0.gguf",
      "jinaai/jina-reranker-v1-turbo-en",
      "Edge tiny reranker GGUF (Q8_0), ~36 MB, 6-layer JinaBERT, 8192 ctx",
      0, 38, 36, 0.0f, 4 },
};

static const int lembed__gguf_model_count = sizeof(lembed__gguf_models) / sizeof(lembed__gguf_models[0]);

#ifdef __cplusplus
}
#endif

/* =========================================================================
 * C API Implementation
 * ========================================================================= */
#ifdef LIBEMBEDDING_IMPLEMENTATION

lembed_status_t lembed_list_gguf_models(const lembed_gguf_model_info_t** out, int* count) {
    if (!out || !count) return LEMBED_ERROR_INVALID_ARGUMENT;
    *out = lembed__gguf_models;
    *count = lembed__gguf_model_count;
    return LEMBED_OK;
}

const lembed_gguf_model_info_t* lembed_find_gguf_model(const char* name) {
    if (!name) return nullptr;
    std::string query(name);
    /* Case-insensitive partial match */
    for (auto& c : query) c = (char)tolower(c);
    for (int i = 0; i < lembed__gguf_model_count; i++) {
        std::string candidate(lembed__gguf_models[i].name);
        for (auto& c : candidate) c = (char)tolower(c);
        if (candidate.find(query) != std::string::npos)
            return &lembed__gguf_models[i];
    }
    return nullptr;
}

const lembed_gguf_model_info_t* lembed_default_gguf_model(void) {
    /* Return the first model (best quality/throughput tradeoff) */
    return lembed__gguf_model_count > 0 ? &lembed__gguf_models[0] : nullptr;
}

#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_GGUF_REGISTRY_IMPL_HPP */



