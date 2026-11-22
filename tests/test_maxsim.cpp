/*
 * test_maxsim.cpp - Unit tests for lembed_maxsim / lembed_maxsim_batch
 * No model downloads required.
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

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabs((double)(a) - (double)(b)) > (eps)) { \
        fprintf(stderr, "FAIL: %s (line %d): %.6f != %.6f\n", msg, __LINE__, (double)(a), (double)(b)); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

int main(void) {
    int passes = 0, failures = 0;

    /* 2 query vectors, 3 doc vectors, dim 2.
     * q0=(1,0): dots 0.5, 0.9, -1 -> 0.9
     * q1=(0,1): dots 0.5, 0.1,  2 -> 2.0   => 2.9 */
    const float q[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float d[] = {0.5f, 0.5f, 0.9f, 0.1f, -1.0f, 2.0f};

    {
        float s = -1.0f;
        ASSERT(lembed_maxsim(q, 2, d, 3, 2, &s) == LEMBED_OK, "maxsim ok");
        ASSERT_NEAR(s, 2.9f, 1e-5f, "maxsim hand-computed 2.9");
    }

    /* All-negative dots: the max is negative, not clamped to 0. */
    {
        const float qn[] = {1.0f, 0.0f};
        const float dn[] = {-0.5f, 0.0f, -0.25f, 0.0f};
        float s = 0.0f;
        ASSERT(lembed_maxsim(qn, 1, dn, 2, 2, &s) == LEMBED_OK, "maxsim negative ok");
        ASSERT_NEAR(s, -0.25f, 1e-6f, "maxsim negative max = -0.25");
    }

    /* Empty cases */
    {
        float s = 7.0f;
        ASSERT(lembed_maxsim(NULL, 0, d, 3, 2, &s) == LEMBED_OK, "nq=0 ok");
        ASSERT(s == 0.0f, "nq=0 -> 0");
        s = 7.0f;
        ASSERT(lembed_maxsim(q, 2, NULL, 0, 2, &s) == LEMBED_OK, "nd=0 ok");
        ASSERT(s == 0.0f, "nd=0 -> 0");
    }

    /* Invalid arguments */
    {
        float s = 0.0f;
        ASSERT(lembed_maxsim(q, 2, d, 3, 2, NULL) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL out");
        ASSERT(lembed_maxsim(NULL, 2, d, 3, 2, &s) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL q");
        ASSERT(lembed_maxsim(q, 2, NULL, 3, 2, &s) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL d");
        ASSERT(lembed_maxsim(q, -1, d, 3, 2, &s) == LEMBED_ERROR_INVALID_ARGUMENT, "negative nq");
        ASSERT(lembed_maxsim(q, 2, d, -1, 2, &s) == LEMBED_ERROR_INVALID_ARGUMENT, "negative nd");
        ASSERT(lembed_maxsim(q, 2, d, 3, 0, &s) == LEMBED_ERROR_INVALID_ARGUMENT, "dim 0");
        ASSERT(lembed_maxsim(q, 2, d, 3, -2, &s) == LEMBED_ERROR_INVALID_ARGUMENT, "dim < 0");
        ASSERT(strlen(lembed_last_error()) > 0, "error message set");
    }

    /* Batch: 3 docs of 3, 0 and 1 rows; batch == loop of single */
    {
        const int64_t off[] = {0, 3, 3, 4};
        const float docs[] = {0.5f, 0.5f, 0.9f, 0.1f, -1.0f, 2.0f, 0.0f, 3.0f};
        float out[3] = {-1.0f, -1.0f, -1.0f};
        ASSERT(lembed_maxsim_batch(q, 2, docs, off, 3, 2, out) == LEMBED_OK, "batch ok");
        for (int i = 0; i < 3; i++) {
            float one = -1.0f;
            int64_t n = off[i + 1] - off[i];
            ASSERT(lembed_maxsim(q, 2, docs + off[i] * 2, (int)n, 2, &one) == LEMBED_OK, "single ok");
            ASSERT(out[i] == one, "batch == single");
        }
        ASSERT_NEAR(out[0], 2.9f, 1e-5f, "batch doc0 = 2.9");
        ASSERT(out[1] == 0.0f, "batch empty doc = 0");
        ASSERT_NEAR(out[2], 3.0f, 1e-6f, "batch doc2 = 0 + 3 = 3");
    }

    /* Batch: zero documents, nq = 0 */
    {
        const int64_t off0[] = {0};
        float out[1] = {5.0f};
        ASSERT(lembed_maxsim_batch(q, 2, NULL, off0, 0, 2, out) == LEMBED_OK, "batch 0 docs ok");
        ASSERT(out[0] == 5.0f, "batch 0 docs leaves out untouched");

        const int64_t off1[] = {0, 3};
        out[0] = 5.0f;
        ASSERT(lembed_maxsim_batch(NULL, 0, d, off1, 1, 2, out) == LEMBED_OK, "batch nq=0 ok");
        ASSERT(out[0] == 0.0f, "batch nq=0 -> 0");
    }

    /* Batch: invalid arguments */
    {
        const int64_t off[] = {0, 3};
        const int64_t bad_first[] = {1, 3};
        const int64_t decreasing[] = {0, 3, 2};
        float out[2] = {0.0f, 0.0f};
        ASSERT(lembed_maxsim_batch(q, 2, d, NULL, 1, 2, out) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL offsets");
        ASSERT(lembed_maxsim_batch(q, 2, d, off, 1, 2, NULL) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL out_scores");
        ASSERT(lembed_maxsim_batch(NULL, 2, d, off, 1, 2, out) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL q");
        ASSERT(lembed_maxsim_batch(q, 2, NULL, off, 1, 2, out) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL docs");
        ASSERT(lembed_maxsim_batch(q, -1, d, off, 1, 2, out) == LEMBED_ERROR_INVALID_ARGUMENT, "negative nq");
        ASSERT(lembed_maxsim_batch(q, 2, d, off, -1, 2, out) == LEMBED_ERROR_INVALID_ARGUMENT, "negative num_docs");
        ASSERT(lembed_maxsim_batch(q, 2, d, off, 1, 0, out) == LEMBED_ERROR_INVALID_ARGUMENT, "dim 0");
        ASSERT(lembed_maxsim_batch(q, 2, d, bad_first, 1, 2, out) == LEMBED_ERROR_INVALID_ARGUMENT, "offsets[0] != 0");
        ASSERT(lembed_maxsim_batch(q, 2, d, decreasing, 2, 2, out) == LEMBED_ERROR_INVALID_ARGUMENT, "decreasing offsets");
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
