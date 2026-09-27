/*
 * libembedding - embedding_mode.h
 * Embedding quality/speed modes
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_EMBEDDING_MODE_H
#define LIBEMBEDDING_EMBEDDING_MODE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LEMBED_MODE_FAST = 0,
    LEMBED_MODE_BALANCED = 1,
    LEMBED_MODE_QUALITY = 2,
} lembed_embedding_mode_t;

const char* lembed_mode_to_string(lembed_embedding_mode_t mode);
lembed_text_model_t lembed_recommended_model_for_mode(lembed_embedding_mode_t mode);

#ifdef LIBEMBEDDING_IMPLEMENTATION
#if defined(__cplusplus)
extern "C++" {
#endif
#include "detail/embedding_mode_impl.hpp"
#if defined(__cplusplus)
}
#endif

const char* lembed_mode_to_string(lembed_embedding_mode_t mode) {
    return lembed::detail::mode_to_string(mode);
}

lembed_text_model_t lembed_recommended_model_for_mode(lembed_embedding_mode_t mode) {
    return lembed::detail::recommended_model_for_mode(mode);
}
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_EMBEDDING_MODE_H */
