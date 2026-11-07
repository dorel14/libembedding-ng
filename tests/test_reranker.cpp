/*
 * test_reranker.c - Integration test for reranker
 * Requires network access for model download.
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, lembed_last_error()); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

int main(void) {
    int passes = 0, failures = 0;

    lembed_reranker_options_t opts = lembed_reranker_options_default();
    lembed_reranker_t* ctx = NULL;

    lembed_status_t s = lembed_reranker_create(&opts, &ctx);
    ASSERT(s == LEMBED_OK, "create reranker");
    if (s != LEMBED_OK) {
        printf("Cannot create reranker: %s\n", lembed_last_error());
        return 1;
    }

    const char* query = "What is machine learning?";
    const char* docs[] = {
        "Machine learning is a branch of AI that learns from data",
        "The Eiffel Tower is located in Paris, France",
        "Deep learning is a type of machine learning using neural networks",
        "Pizza is a popular Italian dish"
    };

    lembed_rerank_results_t result = {0};
    s = lembed_reranker_rerank(ctx, query, docs, 4, 0, &result);
    ASSERT(s == LEMBED_OK, "rerank documents");
    ASSERT(result.count == 4, "got 4 rerank results");

    if (s == LEMBED_OK) {
        /* Results should be sorted by score descending */
        for (int i = 1; i < result.count; i++) {
            ASSERT(result.items[i-1].score >= result.items[i].score,
                   "results sorted by score descending");
        }

        printf("Rerank results:\n");
        for (int i = 0; i < result.count; i++) {
            printf("  [%d] score=%.4f doc_idx=%d\n",
                   i, result.items[i].score, result.items[i].index);
        }

        /* ML-related docs should rank higher than unrelated ones */
        int top_idx = result.items[0].index;
        ASSERT(top_idx == 0 || top_idx == 2,
               "ML-related document ranks first");
    }

    lembed_rerank_results_free(&result);
    lembed_reranker_free(ctx);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
