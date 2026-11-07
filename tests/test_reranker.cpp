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

    /* Token-budgeted batching: a request of mixed lengths is split into several
     * length-sorted batches, and every score must equal the score of the same
     * pair reranked alone (padding is masked, so batching may not change it). */
    {
        static const char* words[] = {"neural", "ranking", "query", "document", "relevance",
                                      "search", "model", "token", "score", "attention"};
        enum { N = 24 };
        static char storage[N][4096];
        const char* mixed[N];
        for (int d = 0; d < N; d++) {
            int n_words = 2 + (d * 37) % 400;
            int pos = 0;
            for (int w = 0; w < n_words && pos < 4000; w++)
                pos += snprintf(storage[d] + pos, sizeof(storage[d]) - (size_t)pos, "%s%s",
                                w ? " " : "", words[(d + w * 3) % 10]);
            mixed[d] = storage[d];
        }

        lembed_stats_t before = {0}, after = {0};
        lembed_reranker_stats(ctx, &before);
        lembed_rerank_results_t batched = {0};
        s = lembed_reranker_rerank(ctx, query, mixed, N, 0, &batched);
        ASSERT(s == LEMBED_OK, "rerank a mixed-length request");
        lembed_reranker_stats(ctx, &after);
        ASSERT(after.batches_run - before.batches_run > 1,
               "mixed-length request is split by the token budget");

        if (s == LEMBED_OK) {
            float by_index[N] = {0};
            for (int i = 0; i < batched.count; i++)
                by_index[batched.items[i].index] = batched.items[i].score;
            double worst = 0.0;
            for (int d = 0; d < N; d++) {
                lembed_rerank_results_t single = {0};
                if (lembed_reranker_rerank(ctx, query, &mixed[d], 1, 0, &single) == LEMBED_OK) {
                    double diff = by_index[d] - single.items[0].score;
                    if (diff < 0) diff = -diff;
                    if (diff > worst) worst = diff;
                }
                lembed_rerank_results_free(&single);
            }
            printf("batched vs single: worst |score diff| = %.6g\n", worst);
            ASSERT(worst <= 1e-4, "batched scores match pairs reranked alone");
        }
        lembed_rerank_results_free(&batched);
    }

    lembed_reranker_free(ctx);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
