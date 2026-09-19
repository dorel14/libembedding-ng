/*
 * libembedding - similarity.h
 * Native similarity functions: cosine, dot product, euclidean distance
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_SIMILARITY_H
#define LIBEMBEDDING_SIMILARITY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compute cosine similarity between two vectors.
 * cosine = dot(a,b) / (||a|| * ||b||)
 * Returns 0.0 if either vector has zero magnitude. */
float lembed_cosine_similarity(const float* a, const float* b, int dim);

/* Compute dot product of two vectors. */
float lembed_dot_product(const float* a, const float* b, int dim);

/* Compute euclidean (L2) distance between two vectors. */
float lembed_euclidean_distance(const float* a, const float* b, int dim);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_SIMILARITY_IMPL
#define LIBEMBEDDING_SIMILARITY_IMPL

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

float lembed_cosine_similarity(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0) return 0.0f;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

float lembed_dot_product(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0) return 0.0f;
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

float lembed_euclidean_distance(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_SIMILARITY_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_SIMILARITY_H */




