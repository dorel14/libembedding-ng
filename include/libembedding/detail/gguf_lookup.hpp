/*
 * libembedding - detail/gguf_lookup.hpp
 * Shared lookup of the GGUF registry entry matching an ONNX registry model
 * (used by the text and reranker llama.cpp fallback paths).
 *
 * Auteur: David Orel
 * Version: 1.15.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_GGUF_LOOKUP_HPP
#define LIBEMBEDDING_DETAIL_GGUF_LOOKUP_HPP

#include "../gguf_registry.h"
#include <cctype>

/* Case-insensitive string equality (NULL never matches). */
static inline bool lembed__gguf_ieq(const char* a, const char* b) {
    if (!a || !b) return false;
    for (; *a && *b; ++a, ++b) {
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
    }
    return *a == *b;
}

/* GGUF registry entry for a registry model, or NULL.
 *
 * The GGUF registry records the original HuggingFace model of each file in
 * model_code: an exact (case-insensitive) match on the model's name or code is
 * the reliable key. lembed_find_gguf_model() matches the GGUF *display* name by
 * substring and is only the last resort. `dim` (0 = unknown, as for rerankers)
 * is checked against the entry's dimension when both are known, so a partial
 * match never silently changes the embedding dimension. */
static inline const lembed_gguf_model_info_t* lembed__gguf_for_model(
        const char* model_name, const char* model_code, int dim) {
    const lembed_gguf_model_info_t* gguf = nullptr;
    const lembed_gguf_model_info_t* list = nullptr;
    int n = 0;
    if (lembed_list_gguf_models(&list, &n) == LEMBED_OK && list) {
        for (int i = 0; i < n && !gguf; i++) {
            const char* code = list[i].model_code;
            if (!code) continue;
            if (lembed__gguf_ieq(code, model_name) || lembed__gguf_ieq(code, model_code)) {
                gguf = &list[i];
            }
        }
    }
    if (!gguf) gguf = lembed_find_gguf_model(model_name);
    if (!gguf) gguf = lembed_find_gguf_model(model_code);
    if (gguf && gguf->dim > 0 && dim > 0 && gguf->dim != dim) gguf = nullptr;
    return gguf;
}

#endif /* LIBEMBEDDING_DETAIL_GGUF_LOOKUP_HPP */
