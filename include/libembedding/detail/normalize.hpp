/*
 * libembedding - detail/normalize.hpp
 * L2 normalization for embedding vectors
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_NORMALIZE_HPP
#define LIBEMBEDDING_DETAIL_NORMALIZE_HPP

#include <cmath>
#include <cstddef>

namespace lembed { namespace detail {

/* L2-normalize each row in a flat [count x dim] matrix in-place */
inline void l2_normalize(float* vectors, int count, int dim) {
    for (int i = 0; i < count; i++) {
        float* row = vectors + i * dim;
        float norm_sq = 0.0f;
        for (int j = 0; j < dim; j++) {
            norm_sq += row[j] * row[j];
        }
        float norm = std::sqrt(norm_sq);
        float inv = (norm > 1e-12f) ? (1.0f / norm) : 0.0f;
        for (int j = 0; j < dim; j++) {
            row[j] *= inv;
        }
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_NORMALIZE_HPP */




