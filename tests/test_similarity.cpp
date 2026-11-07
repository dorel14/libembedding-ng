/*
 * test_similarity.cpp - Unit tests for similarity functions
 * No model downloads required.
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <cmath>
#include <cstdio>

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

    /* Test: cosine similarity — identical vectors = 1.0 */
    {
        float a[] = {1.0f, 2.0f, 3.0f};
        float b[] = {1.0f, 2.0f, 3.0f};
        float sim = lembed_cosine_similarity(a, b, 3);
        ASSERT_NEAR(sim, 1.0f, 1e-5f, "cosine identical = 1.0");
    }

    /* Test: cosine similarity — orthogonal vectors = 0.0 */
    {
        float a[] = {1.0f, 0.0f};
        float b[] = {0.0f, 1.0f};
        float sim = lembed_cosine_similarity(a, b, 2);
        ASSERT_NEAR(sim, 0.0f, 1e-5f, "cosine orthogonal = 0.0");
    }

    /* Test: cosine similarity — opposite vectors = -1.0 */
    {
        float a[] = {1.0f, 2.0f};
        float b[] = {-1.0f, -2.0f};
        float sim = lembed_cosine_similarity(a, b, 2);
        ASSERT_NEAR(sim, -1.0f, 1e-5f, "cosine opposite = -1.0");
    }

    /* Test: dot product */
    {
        float a[] = {1.0f, 2.0f, 3.0f};
        float b[] = {4.0f, 5.0f, 6.0f};
        float dot = lembed_dot_product(a, b, 3);
        ASSERT_NEAR(dot, 32.0f, 1e-5f, "dot product 1*4+2*5+3*6 = 32");
    }

    /* Test: euclidean distance — same point = 0 */
    {
        float a[] = {3.0f, 4.0f};
        float b[] = {3.0f, 4.0f};
        float dist = lembed_euclidean_distance(a, b, 2);
        ASSERT_NEAR(dist, 0.0f, 1e-5f, "euclidean same point = 0");
    }

    /* Test: euclidean distance — classic 3-4-5 */
    {
        float a[] = {0.0f, 0.0f};
        float b[] = {3.0f, 4.0f};
        float dist = lembed_euclidean_distance(a, b, 2);
        ASSERT_NEAR(dist, 5.0f, 1e-5f, "euclidean 3-4-5 = 5");
    }

    /* Test: NULL inputs return 0 */
    {
        ASSERT(lembed_cosine_similarity(NULL, NULL, 3) == 0.0f, "cosine NULL = 0");
        ASSERT(lembed_dot_product(NULL, NULL, 3) == 0.0f, "dot NULL = 0");
        ASSERT(lembed_euclidean_distance(NULL, NULL, 3) == 0.0f, "euclidean NULL = 0");
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
