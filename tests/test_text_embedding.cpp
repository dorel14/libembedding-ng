/*
 * test_text_embedding.c - Integration test for text embeddings
 * Requires network access for model download.
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, lembed_last_error()); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

static float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

int main(void) {
    int passes = 0, failures = 0;

    /* Create text embedding with default model (BGE small EN v1.5) */
    lembed_text_options_t opts = lembed_text_options_default();
    lembed_text_embedding_t* ctx = NULL;

    lembed_status_t s = lembed_text_embedding_create(&opts, &ctx);
    ASSERT(s == LEMBED_OK, "create text embedding");
    if (s != LEMBED_OK) {
        printf("Cannot create embedder: %s\n", lembed_last_error());
        return 1;
    }

    /* Check dimension */
    int dim = lembed_text_embedding_dim(ctx);
    ASSERT(dim == 384, "embedding dim is 384");

    /* Embed similar and dissimilar texts */
    const char* texts[] = {
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "Quantum computing uses qubits for computation"
    };

    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(ctx, texts, 3, 0, &result);
    ASSERT(s == LEMBED_OK, "embed 3 texts");
    ASSERT(result.num_embeddings == 3, "got 3 embeddings");
    ASSERT(result.dim == 384, "result dim is 384");

    if (s == LEMBED_OK) {
        /* Similar texts should have higher cosine similarity */
        float sim_similar = cosine_similarity(
            result.data, result.data + 384, 384);
        float sim_different = cosine_similarity(
            result.data, result.data + 2 * 384, 384);

        printf("Cosine similarity (similar):   %.4f\n", sim_similar);
        printf("Cosine similarity (different): %.4f\n", sim_different);

        ASSERT(sim_similar > sim_different,
               "similar texts have higher cosine similarity");
        ASSERT(sim_similar > 0.5f, "similar texts sim > 0.5");

        /* Verify L2 norm ~1.0 */
        float norm = 0;
        for (int i = 0; i < 384; i++) {
            norm += result.data[i] * result.data[i];
        }
        norm = sqrtf(norm);
        ASSERT(fabsf(norm - 1.0f) < 0.01f, "embedding L2 norm ~1.0");
    }

    lembed_embeddings_free(&result);

    /* Batching must not change the vectors. Mixed lengths (2 to ~400 tokens) make
     * the length-sorted batches split by token budget; each text is compared with
     * the same text embedded alone, and must land in its own input slot. */
    {
        const int n = 24;
        std::vector<std::string> store;
        for (int i = 0; i < n; i++) {
            std::string t;
            int words = (i % 4 == 0) ? 2 : (i % 4 == 1) ? 30 : (i % 4 == 2) ? 120 : 380;
            for (int w = 0; w < words; w++) t += (w ? " alpha" : "topic") + std::to_string((i * 7 + w) % 13);
            store.push_back(t);
        }
        std::vector<const char*> ptrs;
        for (auto& t : store) ptrs.push_back(t.c_str());

        lembed_stats_t before = {0}, after = {0};
        lembed_text_embedding_stats(ctx, &before);
        lembed_embeddings_t batched = {0};
        s = lembed_text_embedding_embed(ctx, ptrs.data(), n, 0, &batched);
        ASSERT(s == LEMBED_OK, "embed mixed-length corpus in default batches");
        lembed_text_embedding_stats(ctx, &after);
        /* The 380-word texts alone exceed the budget: more than one batch. */
        ASSERT(after.batches_run - before.batches_run > 1, "token budget splits the corpus");

        if (s == LEMBED_OK) {
            float worst = 1.0f;
            for (int i = 0; i < n; i++) {
                lembed_embeddings_t single = {0};
                if (lembed_text_embedding_embed(ctx, &ptrs[i], 1, 0, &single) != LEMBED_OK) {
                    ASSERT(0, "embed one text alone");
                    continue;
                }
                float c = cosine_similarity(batched.data + (size_t)i * dim, single.data, dim);
                if (c < worst) worst = c;
                lembed_embeddings_free(&single);
            }
            printf("Worst batched-vs-alone cosine: %.6f\n", worst);
            ASSERT(worst >= 0.9999f, "batched embeddings match single-text embeddings");
        }
        lembed_embeddings_free(&batched);
    }

    lembed_text_embedding_free(ctx);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
