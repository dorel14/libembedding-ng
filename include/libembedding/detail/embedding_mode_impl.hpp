/*
 * libembedding - detail/embedding_mode_impl.hpp
 * Embedding mode implementation
 *
 * Auteur: David Orel
 * Version: 1.16.3
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_EMBEDDING_MODE_IMPL_HPP
#define LIBEMBEDDING_DETAIL_EMBEDDING_MODE_IMPL_HPP

#include <cstring>
#include "../embedding_mode.h"
#include "../model_registry.h"
#include "enum_check.hpp"

namespace lembed { namespace detail {

/* These two return a sentinel rather than a status. The switch is on the raw
 * value so an out-of-range `mode` from a C caller reaches `default` without
 * an enum-typed read (see enum_check.hpp). */
inline const char* mode_to_string(lembed_embedding_mode_t mode) {
    switch (enum_raw(mode)) {
        case LEMBED_MODE_FAST:     return "fast";
        case LEMBED_MODE_BALANCED: return "balanced";
        case LEMBED_MODE_QUALITY:  return "quality";
        default:                   return "unknown";
    }
}

inline lembed_text_model_t recommended_model_for_mode(lembed_embedding_mode_t mode) {
    switch (enum_raw(mode)) {
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

