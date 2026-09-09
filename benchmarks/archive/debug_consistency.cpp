/*
 * Debug: Batch vs Individual inconsistency
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main() {
    printf("=== Debug Batch vs Individual ===\n\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2_Q;
    opts.num_threads = 1;  /* single thread for determinism */
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    if (lembed_text_embedding_create(&opts, &emb) != LEMBED_OK) {
        printf("Erreur: %s\n", lembed_last_error());
        return 1;
    }

    const char* text = "Hello world";

    /* Individual */
    lembed_embeddings_t ind = {0};
    lembed_text_embedding_embed(emb, &text, 1, 1, &ind);

    printf("Individual (1 text): dim=%d\n", ind.dim);
    printf("  [0:5]: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", ind.data[i]);
    printf("\n");

    /* Batch of 1 */
    lembed_embeddings_t batch1 = {0};
    lembed_text_embedding_embed(emb, &text, 1, 1, &batch1);

    printf("Batch (1 text): dim=%d\n", batch1.dim);
    printf("  [0:5]: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", batch1.data[i]);
    printf("\n");

    /* Diff */
    double diff = 0;
    for (int i = 0; i < ind.dim; i++) {
        double d = ind.data[i] - batch1.data[i];
        diff += d * d;
    }
    printf("  Diff: %.8f\n\n", sqrt(diff));

    /* Test with different batch sizes */
    const char* texts[] = {"Hello world", "Machine learning", "Natural language"};
    int n = 3;

    /* Individual for each */
    printf("Individual results:\n");
    for (int i = 0; i < n; i++) {
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(emb, &texts[i], 1, 1, &r);
        printf("  [%d] [0:5]: ", i);
        for (int j = 0; j < 5; j++) printf("%.6f ", r.data[j]);
        printf("\n");
        lembed_embeddings_free(&r);
    }

    /* Batch of 3 */
    std::vector<const char*> ctexts;
    for (int i = 0; i < n; i++) ctexts.push_back(texts[i]);

    lembed_embeddings_t batch3 = {0};
    lembed_text_embedding_embed(emb, ctexts.data(), n, n, &batch3);

    printf("\nBatch (3 texts):\n");
    for (int i = 0; i < batch3.num_embeddings; i++) {
        printf("  [%d] [0:5]: ", i);
        for (int j = 0; j < 5; j++) printf("%.6f ", batch3.data[i * batch3.dim + j]);
        printf("\n");
    }

    /* Compare */
    printf("\nDifferences (individual vs batch):\n");
    for (int i = 0; i < n; i++) {
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(emb, &texts[i], 1, 1, &r);

        double d = 0;
        for (int j = 0; j < r.dim; j++) {
            double diff = r.data[j] - batch3.data[i * batch3.dim + j];
            d += diff * diff;
        }
        printf("  [%d] diff: %.8f\n", i, sqrt(d));
        lembed_embeddings_free(&r);
    }

    lembed_embeddings_free(&ind);
    lembed_embeddings_free(&batch1);
    lembed_embeddings_free(&batch3);
    lembed_text_embedding_free(emb);

    printf("\n=== Termine ===\n");
    return 0;
}
