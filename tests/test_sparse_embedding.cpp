/*
 * test_sparse_embedding.c - Integration test for sparse embeddings
 * Requires network access for model download.
 */

#define LIBEMBEDDING_IMPLEMENTATION
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

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    lembed_sparse_embedding_ctx_t* ctx = NULL;

    lembed_status_t s = lembed_sparse_text_embedding_create(&opts, &ctx);
    ASSERT(s == LEMBED_OK, "create sparse embedding");
    if (s != LEMBED_OK) {
        printf("Cannot create sparse embedder: %s\n", lembed_last_error());
        return 1;
    }

    const char* texts[] = {
        "Machine learning is a subset of artificial intelligence",
        "The weather is sunny today"
    };

    lembed_sparse_embeddings_t result = {0};
    s = lembed_sparse_text_embedding_embed(ctx, texts, 2, 0, NULL, &result);
    ASSERT(s == LEMBED_OK, "embed sparse texts");
    ASSERT(result.count == 2, "got 2 sparse results");

    if (s == LEMBED_OK) {
        for (int i = 0; i < result.count; i++) {
            ASSERT(result.items[i].length > 0, "sparse result has nonzero entries");
            ASSERT(result.items[i].indices != NULL, "indices non-null");
            ASSERT(result.items[i].values != NULL, "values non-null");

            printf("Text %d: %d nonzero sparse entries\n", i, result.items[i].length);

            /* All values should be positive */
            for (int j = 0; j < result.items[i].length; j++) {
                ASSERT(result.items[i].values[j] > 0.0f, "sparse value > 0");
            }
        }
    }

    lembed_sparse_embeddings_free(&result);
    lembed_sparse_text_embedding_free(ctx);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
