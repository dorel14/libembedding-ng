/*
 * test_pooling.cpp - Unit tests for pooling and normalization
 * No model downloads required.
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/detail/pooling.hpp>
#include <libembedding/detail/normalize.hpp>
#include <libembedding/detail/sparse_postprocess.hpp>

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
    if (std::fabs((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL: %s (line %d): %.6f != %.6f\n", msg, __LINE__, (double)(a), (double)(b)); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

int main() {
    int passes = 0, failures = 0;

    /* Test L2 normalization */
    {
        float vec[] = { 3.0f, 4.0f };
        lembed::detail::l2_normalize(vec, 1, 2);
        ASSERT_NEAR(vec[0], 0.6f, 1e-6f, "norm [3,4] -> [0.6, 0.8] x");
        ASSERT_NEAR(vec[1], 0.8f, 1e-6f, "norm [3,4] -> [0.6, 0.8] y");

        /* Verify L2 norm is ~1.0 */
        float norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
        ASSERT_NEAR(norm, 1.0f, 1e-6f, "normalized vector has unit norm");
    }

    /* Test batch normalization */
    {
        float vecs[] = { 3.0f, 4.0f, 1.0f, 0.0f };
        lembed::detail::l2_normalize(vecs, 2, 2);
        ASSERT_NEAR(vecs[0], 0.6f, 1e-6f, "batch norm vec0.x");
        ASSERT_NEAR(vecs[1], 0.8f, 1e-6f, "batch norm vec0.y");
        ASSERT_NEAR(vecs[2], 1.0f, 1e-6f, "batch norm vec1.x");
        ASSERT_NEAR(vecs[3], 0.0f, 1e-6f, "batch norm vec1.y");
    }

    /* Test zero vector normalization */
    {
        float vec[] = { 0.0f, 0.0f };
        lembed::detail::l2_normalize(vec, 1, 2);
        ASSERT_NEAR(vec[0], 0.0f, 1e-6f, "zero vec stays zero x");
        ASSERT_NEAR(vec[1], 0.0f, 1e-6f, "zero vec stays zero y");
    }

    /* Test CLS pooling with 3D tensor [2, 3, 2] */
    {
        /* batch=2, seq_len=3, dim=2 */
        float tensor[] = {
            /* batch 0 */
            1.0f, 2.0f,   /* token 0 (CLS) */
            3.0f, 4.0f,   /* token 1 */
            5.0f, 6.0f,   /* token 2 */
            /* batch 1 */
            7.0f, 8.0f,   /* token 0 (CLS) */
            9.0f, 10.0f,  /* token 1 */
            11.0f, 12.0f, /* token 2 */
        };
        float out[4];
        lembed::detail::pool_cls(tensor, 2, 3, 2, 3, out);
        ASSERT_NEAR(out[0], 1.0f, 1e-6f, "CLS batch0 dim0");
        ASSERT_NEAR(out[1], 2.0f, 1e-6f, "CLS batch0 dim1");
        ASSERT_NEAR(out[2], 7.0f, 1e-6f, "CLS batch1 dim0");
        ASSERT_NEAR(out[3], 8.0f, 1e-6f, "CLS batch1 dim1");
    }

    /* Test CLS pooling with 2D tensor [2, 2] (already pooled) */
    {
        float tensor[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float out[4];
        lembed::detail::pool_cls(tensor, 2, 0, 2, 2, out);
        ASSERT_NEAR(out[0], 1.0f, 1e-6f, "CLS 2D passthrough 0");
        ASSERT_NEAR(out[1], 2.0f, 1e-6f, "CLS 2D passthrough 1");
        ASSERT_NEAR(out[2], 3.0f, 1e-6f, "CLS 2D passthrough 2");
        ASSERT_NEAR(out[3], 4.0f, 1e-6f, "CLS 2D passthrough 3");
    }

    /* Test Mean pooling with 3D tensor [1, 3, 2] */
    {
        /* batch=1, seq_len=3, dim=2 */
        float tensor[] = {
            1.0f, 2.0f,  /* token 0 */
            3.0f, 4.0f,  /* token 1 */
            5.0f, 6.0f,  /* token 2 */
        };
        int64_t mask[] = { 1, 1, 0 }; /* only first 2 tokens are real */
        float out[2];
        lembed::detail::pool_mean(tensor, mask, 1, 3, 2, 3, out);
        ASSERT_NEAR(out[0], 2.0f, 1e-6f, "Mean (1+3)/2 = 2.0");
        ASSERT_NEAR(out[1], 3.0f, 1e-6f, "Mean (2+4)/2 = 3.0");
    }

    /* Test Mean pooling with all tokens masked in */
    {
        float tensor[] = { 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f };
        int64_t mask[] = { 1, 1, 1 };
        float out[2];
        lembed::detail::pool_mean(tensor, mask, 1, 3, 2, 3, out);
        ASSERT_NEAR(out[0], 6.0f, 1e-6f, "Mean all tokens dim0");
        ASSERT_NEAR(out[1], 8.0f, 1e-6f, "Mean all tokens dim1");
    }

    /* Test Mean pooling with 2D tensor (passthrough) */
    {
        float tensor[] = { 1.0f, 2.0f };
        int64_t mask[] = { 1 };
        float out[2];
        lembed::detail::pool_mean(tensor, mask, 1, 1, 2, 2, out);
        ASSERT_NEAR(out[0], 1.0f, 1e-6f, "Mean 2D passthrough 0");
        ASSERT_NEAR(out[1], 2.0f, 1e-6f, "Mean 2D passthrough 1");
    }

    /* SPLADE max pooling should log once per vocabulary term and reuse scratch. */
    {
        const float logits[] = {
            -2.0f, 0.5f, 0.0f,
             1.0f, 0.25f, 0.0f,
             4.0f, 0.1f, -1.0f
        };
        const int64_t first_mask[] = { 1, 1, 0 };
        const int64_t second_mask[] = { 0, 0, 1 };
        std::vector<float> scores(3, -1.0f);
        const float* scratch_data = scores.data();
        lembed::detail::SparseResult sparse;

        lembed::detail::splade_pool(logits, 3, first_mask, 3, sparse, scores);
        ASSERT(sparse.indices.size() == 2, "SPLADE drops zero and masked terms");
        ASSERT(sparse.indices[0] == 0 && sparse.indices[1] == 1,
               "SPLADE preserves vocabulary indices");
        ASSERT_NEAR(sparse.values[0], std::log(2.0f), 1e-6f,
                    "SPLADE logs the maximum positive logit");
        ASSERT_NEAR(sparse.values[1], std::log(1.5f), 1e-6f,
                    "SPLADE applies log after max pooling");
        ASSERT(scores.data() == scratch_data,
               "SPLADE reuses the caller-provided score buffer");

        lembed::detail::splade_pool(logits, 3, second_mask, 3, sparse, scores);
        ASSERT(sparse.indices.size() == 2,
               "SPLADE resets the sparse result and scratch between documents");
        ASSERT(sparse.indices[0] == 0 && sparse.indices[1] == 1,
               "SPLADE returns only terms from the current document");
        ASSERT_NEAR(sparse.values[0], std::log(5.0f), 1e-6f,
                    "SPLADE pools only unmasked token logits");
        ASSERT_NEAR(sparse.values[1], std::log(1.1f), 1e-6f,
                    "SPLADE recomputes each term after scratch reuse");
    }

    /* Sparse pruning compacts in place and does not sort top-k results twice. */
    {
        lembed::detail::SparseResult sparse;
        sparse.indices = { 4, 1, 3, 2, 0 };
        sparse.values = { 0.8f, 0.7f, 0.6f, 0.1f, 0.4f };
        lembed::detail::sparse_prune_and_sort(
            sparse, 2, 0.3f, LEMBED_SPARSE_FORMAT_DICT);
        ASSERT(sparse.indices.size() == 2, "Sparse pruning applies top-k");
        ASSERT(sparse.indices[0] == 4 && sparse.indices[1] == 1,
               "Sparse top-k stays sorted by descending weight");
        ASSERT_NEAR(sparse.values[0], 0.8f, 1e-6f,
                    "Sparse top-k keeps the highest weight");
        ASSERT_NEAR(sparse.values[1], 0.7f, 1e-6f,
                    "Sparse top-k keeps the second-highest weight");

        sparse.indices = { 4, 1, 3, 2, 0 };
        sparse.values = { 0.8f, 0.7f, 0.6f, 0.1f, 0.4f };
        lembed::detail::sparse_prune_and_sort(
            sparse, 2, 0.3f, LEMBED_SPARSE_FORMAT_INDEX_ORDER);
        ASSERT(sparse.indices.size() == 2, "Index-order format keeps top-k");
        ASSERT(sparse.indices[0] == 1 && sparse.indices[1] == 4,
               "Index-order format sorts selected terms by index");
        ASSERT_NEAR(sparse.values[0], 0.7f, 1e-6f,
                    "Index-order format preserves the selected weight");
        ASSERT_NEAR(sparse.values[1], 0.8f, 1e-6f,
                    "Index-order format keeps index-weight associations");
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
