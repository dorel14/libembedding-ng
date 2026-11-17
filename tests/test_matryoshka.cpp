/*
 * libembedding - test_matryoshka.cpp
 * Unit tests for lembed_embeddings_truncate (Matryoshka truncation).
 * Hermetic: synthetic vectors only, no model and no network.
 *
 * Auteur: David Orel
 * Version: 1.14.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int passes = 0, failures = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

/* Deterministic pseudo-random vectors (LCG): no <random> distribution, whose
 * output is implementation-defined, so the data is identical everywhere. */
static std::vector<float> make_vectors(int n, int dim, unsigned seed) {
    std::vector<float> v((size_t)n * dim);
    unsigned s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
    return v;
}

/* Reference: prefix of k components, then L2 renormalisation in double. */
static std::vector<float> reference(const std::vector<float>& v, int n, int dim,
                                    int k, bool renorm) {
    std::vector<float> out((size_t)n * k);
    for (int i = 0; i < n; i++) {
        double ss = 0.0;
        for (int j = 0; j < k; j++) {
            double x = v[(size_t)i * dim + j];
            ss += x * x;
        }
        double inv = (renorm && ss > 0.0) ? 1.0 / std::sqrt(ss) : 1.0;
        for (int j = 0; j < k; j++)
            out[(size_t)i * k + j] = (float)(v[(size_t)i * dim + j] * inv);
    }
    return out;
}

static float max_abs_diff(const float* a, const float* b, size_t count) {
    float m = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void check_truncate(int n, int dim, int k, bool renorm, unsigned seed) {
    std::vector<float> src = make_vectors(n, dim, seed);
    std::vector<float> buf = src;
    std::vector<float> ref = reference(src, n, dim, k, renorm);

    lembed_embeddings_t emb;
    emb.data = buf.data();
    emb.num_embeddings = n;
    emb.dim = dim;

    lembed_status_t s = lembed_embeddings_truncate(&emb, k, renorm ? 1 : 0);
    char msg[128];
    snprintf(msg, sizeof(msg), "truncate %d->%d renorm=%d returns OK", dim, k, (int)renorm);
    ASSERT(s == LEMBED_OK, msg);
    ASSERT(emb.dim == k, "dim updated");
    ASSERT(emb.num_embeddings == n, "num_embeddings unchanged");
    ASSERT(emb.data == buf.data(), "buffer is not reallocated");

    float diff = max_abs_diff(emb.data, ref.data(), (size_t)n * k);
    snprintf(msg, sizeof(msg), "truncate %d->%d matches reference (max diff %.3g)", dim, k, (double)diff);
    if (renorm) {
        ASSERT(diff <= 1e-6f, msg);
        for (int i = 0; i < n; i++) {
            double ss = 0.0;
            for (int j = 0; j < k; j++) {
                double x = emb.data[(size_t)i * k + j];
                ss += x * x;
            }
            ASSERT(std::fabs(std::sqrt(ss) - 1.0) < 1e-5, "row is unit norm");
        }
    } else {
        /* No arithmetic without renormalisation: a pure move, bit-exact. */
        ASSERT(std::memcmp(emb.data, ref.data(), (size_t)n * k * sizeof(float)) == 0, msg);
    }
}

int main(void) {
    /* Matryoshka sizes from a 768-dim and a 1024-dim model. */
    const int dims_768[] = {768, 512, 256, 128, 64, 1};
    for (int k : dims_768) check_truncate(9, 768, k, true, 42u + (unsigned)k);
    const int dims_1024[] = {1024, 768, 512, 256, 128, 64};
    for (int k : dims_1024) check_truncate(5, 1024, k, true, 7u + (unsigned)k);

    /* Without renormalisation: plain prefix. */
    check_truncate(6, 64, 16, false, 3u);
    /* Single row: nothing to move, still renormalised. */
    check_truncate(1, 32, 8, true, 11u);

    /* Zero prefix stays zero (no division by zero, no NaN). */
    {
        std::vector<float> buf = make_vectors(3, 8, 5u);
        for (int j = 0; j < 4; j++) buf[8 + j] = 0.0f; /* row 1, first 4 */
        lembed_embeddings_t emb = { buf.data(), 3, 8 };
        ASSERT(lembed_embeddings_truncate(&emb, 4, 1) == LEMBED_OK, "zero row OK");
        bool zeros = true;
        for (int j = 0; j < 4; j++) zeros = zeros && emb.data[4 + j] == 0.0f;
        ASSERT(zeros, "zero row stays zero");
        bool finite = true;
        for (int j = 0; j < 12; j++) finite = finite && std::isfinite(emb.data[j]);
        ASSERT(finite, "no NaN/Inf after zero row");
    }

    /* Empty result: valid, only the dim changes. */
    {
        lembed_embeddings_t emb = { nullptr, 0, 8 };
        ASSERT(lembed_embeddings_truncate(&emb, 4, 1) == LEMBED_OK, "empty result OK");
        ASSERT(emb.dim == 4, "empty result dim updated");
    }

    /* Invalid arguments: status, detailed error, result untouched. */
    {
        std::vector<float> src = make_vectors(2, 16, 9u);
        const int bad_dims[] = {0, -3, 17};
        for (int k : bad_dims) {
            std::vector<float> buf = src;
            lembed_embeddings_t emb = { buf.data(), 2, 16 };
            lembed_status_t s = lembed_embeddings_truncate(&emb, k, 1);
            ASSERT(s == LEMBED_ERROR_INVALID_ARGUMENT, "bad new_dim -> INVALID_ARGUMENT");
            ASSERT(std::strstr(lembed_last_error(), "new_dim") != nullptr,
                   "bad new_dim -> lembed_last_error mentions new_dim");
            ASSERT(emb.dim == 16, "bad new_dim leaves dim untouched");
            ASSERT(std::memcmp(buf.data(), src.data(), src.size() * sizeof(float)) == 0,
                   "bad new_dim leaves data untouched");
        }

        ASSERT(lembed_embeddings_truncate(nullptr, 4, 1) == LEMBED_ERROR_INVALID_ARGUMENT,
               "NULL result -> INVALID_ARGUMENT");
        ASSERT(lembed_last_error()[0] != '\0', "NULL result sets an error message");

        lembed_embeddings_t no_data = { nullptr, 2, 16 };
        ASSERT(lembed_embeddings_truncate(&no_data, 4, 1) == LEMBED_ERROR_INVALID_ARGUMENT,
               "NULL data with rows -> INVALID_ARGUMENT");

        std::vector<float> buf = src;
        lembed_embeddings_t neg_rows = { buf.data(), -1, 16 };
        ASSERT(lembed_embeddings_truncate(&neg_rows, 4, 1) == LEMBED_ERROR_INVALID_ARGUMENT,
               "negative num_embeddings -> INVALID_ARGUMENT");
        lembed_embeddings_t zero_dim = { buf.data(), 2, 0 };
        ASSERT(lembed_embeddings_truncate(&zero_dim, 1, 1) == LEMBED_ERROR_INVALID_ARGUMENT,
               "dim 0 -> INVALID_ARGUMENT");
    }

    printf("test_matryoshka: %d passed, %d failed\n", passes, failures);
    return failures == 0 ? 0 : 1;
}
