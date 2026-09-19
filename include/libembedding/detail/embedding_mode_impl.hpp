/*
 * libembedding - detail/embedding_mode_impl.hpp
 * Embedding mode implementation
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_EMBEDDING_MODE_IMPL_HPP
#define LIBEMBEDDING_DETAIL_EMBEDDING_MODE_IMPL_HPP

#include <cstring>
#include "../embedding_mode.h"
#include "../model_registry.h"

namespace lembed { namespace detail {

inline const char* mode_to_string(lembed_embedding_mode_t mode) {
    switch (mode) {
        case LEMBED_MODE_FAST:     return "fast";
        case LEMBED_MODE_BALANCED: return "balanced";
        case LEMBED_MODE_QUALITY:  return "quality";
        default:                   return "unknown";
    }
}

inline lembed_text_model_t recommended_model_for_mode(lembed_embedding_mode_t mode) {
    switch (mode) {
        case LEMBED_MODE_FAST:
            return LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2_Q;
        case LEMBED_MODE_BALANCED:
            return LEMBED_TEXT_BGE_SMALL_EN_V15;
        case LEMBED_MODE_QUALITY:
            return LEMBED_TEXT_BGE_BASE_EN_V15;
        default:
            return LEMBED_TEXT_MODEL_COUNT;
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_EMBEDDING_MODE_IMPL_HPP */

