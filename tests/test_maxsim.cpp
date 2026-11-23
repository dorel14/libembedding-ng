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
#include <vector>

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

    /* Kernel fidelity: random data against a double-precision reference.
     * Dims cover multiples of the 16-lane kernel (96, 128), non-multiples (7,
     * 17, 131) and tails only (1); counts cover 1 row up to 234 rows. */
    {
        const int dims[] = {1, 7, 8, 15, 16, 17, 96, 128, 131};
        const int counts[][2] = {{1, 1}, {1, 5}, {3, 1}, {5, 4}, {32, 234}, {2, 33}};
        unsigned long long seed = 0x9E3779B97F4A7C15ULL;
        double worst = 0.0;
        for (int dim : dims) {
            for (const auto& c : counts) {
                const int nq = c[0], nd = c[1];
                std::vector<float> qm((size_t)nq * dim), dm((size_t)nd * dim);
                for (auto* m : {&qm, &dm}) {
                    for (auto& v : *m) {
                        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                        v = (float)((double)(seed >> 40) / (double)(1ULL << 24) * 2.0 - 1.0);
                    }
                }
                double ref = 0.0;
                for (int i = 0; i < nq; i++) {
                    double best = -1e300;
                    for (int j = 0; j < nd; j++) {
                        double dot = 0.0;
                        for (int k = 0; k < dim; k++)
                            dot += (double)qm[(size_t)i * dim + k] * (double)dm[(size_t)j * dim + k];
                        if (dot > best) best = dot;
                    }
                    ref += best;
                }
                float s = 0.0f, s2 = 0.0f;
                ASSERT(lembed_maxsim(qm.data(), nq, dm.data(), nd, dim, &s) == LEMBED_OK, "fidelity ok");
                ASSERT(lembed_maxsim(qm.data(), nq, dm.data(), nd, dim, &s2) == LEMBED_OK, "fidelity ok 2");
                ASSERT(s == s2, "deterministic");
                /* Float rounding of a dot of `dim` terms of magnitude <= 1,
                 * summed over nq maxima: 1e-5 relative to nq * (1 + sqrt(dim)). */
                double tol = 1e-5 * (double)nq * (1.0 + std::sqrt((double)dim));
                double err = fabs((double)s - ref);
                if (err > worst) worst = err;
                ASSERT(err <= tol, "kernel within float rounding of the double reference");
                /* batch over the same data split in two documents == per-doc calls */
                if (nd >= 2) {
                    int h = nd / 2;
                    const int64_t off[] = {0, h, nd};
                    float out2[2] = {0.0f, 0.0f}, a = 0.0f, b = 0.0f;
                    ASSERT(lembed_maxsim_batch(qm.data(), nq, dm.data(), off, 2, dim, out2) == LEMBED_OK, "split batch ok");
                    lembed_maxsim(qm.data(), nq, dm.data(), h, dim, &a);
                    lembed_maxsim(qm.data(), nq, dm.data() + (size_t)h * dim, nd - h, dim, &b);
                    ASSERT(out2[0] == a && out2[1] == b, "split batch == per-doc");
                }
            }
        }
        printf("maxsim kernel: worst abs error vs double reference = %.3g\n", worst);
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
