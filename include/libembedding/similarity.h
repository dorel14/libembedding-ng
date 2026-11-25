/*
 * libembedding - similarity.h
 * Native similarity functions: cosine, dot product, euclidean distance
 *
 * Auteur: David Orel
 * Version: 1.16.5
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

/**
 * MaxSim late-interaction score (ColBERT): the sum, over the nq query vectors,
 * of the maximum dot product with any of the nd document vectors.
 *
 * Vectors are row-major, `dim` floats each. L2-normalised inputs give the
 * cosine MaxSim. nq == 0 or nd == 0 gives *out_score = 0. No state,
 * re-entrant. Portable C kernel (no intrinsics) that auto-vectorises; one
 * temporary copy of the query is allocated and freed within the call (a
 * scalar loop with the same results is used if that allocation fails).
 * Deterministic accumulation order.
 *
 * @param q          nq x dim query matrix (may be NULL when nq == 0)
 * @param nq         number of query vectors (>= 0)
 * @param d          nd x dim document matrix (may be NULL when nd == 0)
 * @param nd         number of document vectors (>= 0)
 * @param dim        vector dimension (> 0)
 * @param out_score  receives the score (must not be NULL)
 * @return LEMBED_OK, or LEMBED_ERROR_INVALID_ARGUMENT (NULL out pointer, NULL
 *         q/d with a non-zero count, negative count, dim <= 0); details via
 *         lembed_last_error().
 */
lembed_status_t lembed_maxsim(const float* q, int nq, const float* d, int nd,
                              int dim, float* out_score);

/**
 * MaxSim of one query against several documents: one score per document, in
 * input order (no sorting, no top-K, no index).
 *
 * Document i is rows offsets[i] .. offsets[i+1]-1 of `docs`; `offsets` has
 * num_docs + 1 entries, offsets[0] == 0, non-decreasing.
 *
 * @param q           nq x dim query matrix (may be NULL when nq == 0)
 * @param nq          number of query vectors (>= 0)
 * @param docs        concatenated document rows (may be NULL when offsets[num_docs] == 0)
 * @param offsets     num_docs + 1 row offsets
 * @param num_docs    number of documents (>= 0)
 * @param dim         vector dimension (> 0)
 * @param out_scores  caller-allocated, num_docs floats
 * @return LEMBED_OK, or LEMBED_ERROR_INVALID_ARGUMENT (NULL out pointer or
 *         offsets, negative count, dim <= 0, offsets[0] != 0, decreasing
 *         offsets, a document longer than INT_MAX rows, NULL q/docs with a
 *         non-zero count).
 */
lembed_status_t lembed_maxsim_batch(const float* q, int nq, const float* docs,
                                    const int64_t* offsets, int num_docs,
                                    int dim, float* out_scores);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_SIMILARITY_IMPL
#define LIBEMBEDDING_SIMILARITY_IMPL

#include <math.h>
#include <stdlib.h>

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

static lembed_status_t lembed__maxsim_invalid(const char* msg) {
#ifdef __cplusplus
    lembed::detail::set_error(msg);
#else
    lembed__set_error(msg);
#endif
    return LEMBED_ERROR_INVALID_ARGUMENT;
}

/*
 * MaxSim kernel. The query is repacked once per call in tiles of
 * LEMBED__MAXSIM_TILE rows, transposed (tile[k][lane]). For one document row the
 * dot products of a whole tile are then built together:
 *     acc[lane] += tile[k][lane] * d[k]      (k = 0 .. dim - 1)
 * Each lane keeps its own running sum, so there is no horizontal reduction and
 * the loop over lanes vectorises without fast-math (4 AVX2 registers of
 * accumulators for 32 lanes). Every lane sums its terms in the same order as a
 * plain scalar dot product, so results are deterministic and match the scalar
 * loop up to float contraction (FMA). Padding lanes of a partial tile are zero
 * and ignored.
 */
#define LEMBED__MAXSIM_TILE 32

/* Packed query size in floats, or 0 if it does not fit in size_t. */
static size_t lembed__maxsim_packed_floats(int nq, int dim) {
    size_t tiles = ((size_t)nq + LEMBED__MAXSIM_TILE - 1) / LEMBED__MAXSIM_TILE;
    size_t row = (size_t)dim * LEMBED__MAXSIM_TILE;
    if (tiles > ((size_t)-1) / sizeof(float) / row) return 0;
    return tiles * row;
}

/* Transposed, zero-padded copy of the query (nq >= 1); NULL on allocation failure. */
static float* lembed__maxsim_pack(const float* q, int nq, int dim) {
    size_t n = lembed__maxsim_packed_floats(nq, dim);
    float* packed = n ? (float*)malloc(n * sizeof(float)) : NULL;
    if (!packed) return NULL;
    for (int i0 = 0; i0 < nq; i0 += LEMBED__MAXSIM_TILE) {
        int m = nq - i0 < LEMBED__MAXSIM_TILE ? nq - i0 : LEMBED__MAXSIM_TILE;
        float* tile = packed + (size_t)(i0 / LEMBED__MAXSIM_TILE) * (size_t)dim * LEMBED__MAXSIM_TILE;
        for (int k = 0; k < dim; k++) {
            for (int l = 0; l < LEMBED__MAXSIM_TILE; l++) {
                tile[(size_t)k * LEMBED__MAXSIM_TILE + l] =
                    l < m ? q[(size_t)(i0 + l) * (size_t)dim + (size_t)k] : 0.0f;
            }
        }
    }
    return packed;
}

/* Dot products of one document row against every lane of a packed tile. */
static void lembed__maxsim_tile_dots(const float* tile, const float* dj, int dim, float* acc) {
    for (int l = 0; l < LEMBED__MAXSIM_TILE; l++) acc[l] = 0.0f;
    for (int k = 0; k < dim; k++) {
        const float dv = dj[k];
        const float* qk = tile + (size_t)k * LEMBED__MAXSIM_TILE;
        for (int l = 0; l < LEMBED__MAXSIM_TILE; l++) acc[l] += qk[l] * dv;
    }
}

/* Score of one document (nd >= 1 rows) against a packed query (nq >= 1 rows); inputs already validated. */
static float lembed__maxsim_core(const float* packed, int nq, const float* d, int nd, int dim) {
    float total = 0.0f;
    for (int i0 = 0; i0 < nq; i0 += LEMBED__MAXSIM_TILE) {
        int m = nq - i0 < LEMBED__MAXSIM_TILE ? nq - i0 : LEMBED__MAXSIM_TILE;
        const float* tile = packed + (size_t)(i0 / LEMBED__MAXSIM_TILE) * (size_t)dim * LEMBED__MAXSIM_TILE;
        float best[LEMBED__MAXSIM_TILE];
        float acc[LEMBED__MAXSIM_TILE];
        lembed__maxsim_tile_dots(tile, d, dim, best);
        for (int j = 1; j < nd; j++) {
            lembed__maxsim_tile_dots(tile, d + (size_t)j * (size_t)dim, dim, acc);
            for (int l = 0; l < LEMBED__MAXSIM_TILE; l++) best[l] = acc[l] > best[l] ? acc[l] : best[l];
        }
        for (int l = 0; l < m; l++) total += best[l];
    }
    return total;
}

/* Allocation-free fallback with the same results, used when the packed copy
 * cannot be allocated. */
static float lembed__maxsim_core_plain(const float* q, int nq, const float* d, int nd, int dim) {
    float total = 0.0f;
    for (int i = 0; i < nq; i++) {
        const float* qi = q + (size_t)i * (size_t)dim;
        float best = 0.0f;
        for (int j = 0; j < nd; j++) {
            const float* dj = d + (size_t)j * (size_t)dim;
            float dot = 0.0f;
            for (int k = 0; k < dim; k++) dot += qi[k] * dj[k];
            if (j == 0 || dot > best) best = dot;
        }
        total += best;
    }
    return total;
}

lembed_status_t lembed_maxsim(const float* q, int nq, const float* d, int nd,
                              int dim, float* out_score) {
    if (!out_score) return lembed__maxsim_invalid("lembed_maxsim: out_score is NULL");
    if (nq < 0 || nd < 0) return lembed__maxsim_invalid("lembed_maxsim: negative vector count");
    if (dim <= 0) return lembed__maxsim_invalid("lembed_maxsim: dim must be > 0");
    if ((nq > 0 && !q) || (nd > 0 && !d))
        return lembed__maxsim_invalid("lembed_maxsim: NULL matrix with a non-zero count");
    if (nq == 0 || nd == 0) {
        *out_score = 0.0f;
        return LEMBED_OK;
    }
    float* packed = lembed__maxsim_pack(q, nq, dim);
    if (packed) {
        *out_score = lembed__maxsim_core(packed, nq, d, nd, dim);
        free(packed);
    } else {
        *out_score = lembed__maxsim_core_plain(q, nq, d, nd, dim);
    }
    return LEMBED_OK;
}

lembed_status_t lembed_maxsim_batch(const float* q, int nq, const float* docs,
                                    const int64_t* offsets, int num_docs,
                                    int dim, float* out_scores) {
    if (!offsets || !out_scores)
        return lembed__maxsim_invalid("lembed_maxsim_batch: offsets or out_scores is NULL");
    if (nq < 0 || num_docs < 0)
        return lembed__maxsim_invalid("lembed_maxsim_batch: negative count");
    if (dim <= 0) return lembed__maxsim_invalid("lembed_maxsim_batch: dim must be > 0");
    if (offsets[0] != 0)
        return lembed__maxsim_invalid("lembed_maxsim_batch: offsets[0] must be 0");
    for (int i = 0; i < num_docs; i++) {
        if (offsets[i + 1] < offsets[i])
            return lembed__maxsim_invalid("lembed_maxsim_batch: offsets must be non-decreasing");
        if (offsets[i + 1] - offsets[i] > 2147483647LL)
            return lembed__maxsim_invalid("lembed_maxsim_batch: document has too many rows");
    }
    if (nq > 0 && !q)
        return lembed__maxsim_invalid("lembed_maxsim_batch: q is NULL with a non-zero count");
    if (offsets[num_docs] > 0 && !docs)
        return lembed__maxsim_invalid("lembed_maxsim_batch: docs is NULL with a non-zero row count");
    float* packed = (nq > 0 && num_docs > 0) ? lembed__maxsim_pack(q, nq, dim) : NULL;
    for (int i = 0; i < num_docs; i++) {
        int nd = (int)(offsets[i + 1] - offsets[i]);
        if (nq == 0 || nd == 0) {
            out_scores[i] = 0.0f;
            continue;
        }
        const float* doc = docs + (size_t)offsets[i] * (size_t)dim;
        out_scores[i] = packed ? lembed__maxsim_core(packed, nq, doc, nd, dim)
                               : lembed__maxsim_core_plain(q, nq, doc, nd, dim);
    }
    free(packed);
    return LEMBED_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_SIMILARITY_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_SIMILARITY_H */




