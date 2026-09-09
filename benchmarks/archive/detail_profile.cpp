/*
 * Profilage detaille par etape
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <libembedding/detail/tokenizer_impl.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main() {
    printf("=== Profilage detaille ===\n\n");

    const char* model_path = "C:/Users/david/.cache/libembedding/models--Xenova-bge-small-en-v1.5";

    /* Load model */
    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.num_threads = 4;
    opts.pooling = LEMBED_POOLING_MEAN;
    opts.dim = 384;

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_path(model_path, &opts, &embedder);
    if (s != LEMBED_OK) {
        fprintf(stderr, "Erreur: %s\n", lembed_last_error());
        return 1;
    }

    printf("Modele charge (dim=%d, max_len=%d)\n",
           lembed_text_embedding_dim(embedder),
           lembed_text_embedding_max_length(embedder));

    /* Test text */
    const char* text = "The quick brown fox jumps over the lazy dog.";
    const char* texts[64];
    for (int i = 0; i < 64; i++) texts[i] = text;

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        lembed_embeddings_t w = {0};
        lembed_text_embedding_embed(embedder, texts, 4, 4, &w);
        lembed_embeddings_free(&w);
    }

    /* Profile batch=1 repeatedly */
    printf("\n--- Mesures individuelles (batch=1) ---\n");
    for (int i = 0; i < 10; i++) {
        lembed_embeddings_t result = {0};
        double t0 = now_ms();
        lembed_text_embedding_embed(embedder, texts, 1, 1, &result);
        double t1 = now_ms();
        printf("  Run %d: %.2f ms\n", i, t1 - t0);
        lembed_embeddings_free(&result);
    }

    /* Profile batch=4 repeatedly */
    printf("\n--- Mesures individuelles (batch=4) ---\n");
    for (int i = 0; i < 10; i++) {
        lembed_embeddings_t result = {0};
        double t0 = now_ms();
        lembed_text_embedding_embed(embedder, texts, 4, 4, &result);
        double t1 = now_ms();
        printf("  Run %d: %.2f ms (%.2f par texte)\n", i, t1 - t0, (t1-t0)/4.0);
        lembed_embeddings_free(&result);
    }

    /* Profile batch=16 repeatedly */
    printf("\n--- Mesures individuelles (batch=16) ---\n");
    for (int i = 0; i < 10; i++) {
        lembed_embeddings_t result = {0};
        double t0 = now_ms();
        lembed_text_embedding_embed(embedder, texts, 16, 16, &result);
        double t1 = now_ms();
        printf("  Run %d: %.2f ms (%.2f par texte)\n", i, t1 - t0, (t1-t0)/16.0);
        lembed_embeddings_free(&result);
    }

    /* Test with different thread counts for ONNX */
    printf("\n--- Test pure inference (pas de tokenisation) ---\n");
    printf("Le delai mesure inclut: tokenisation + inference + pooling + normalisation\n");
    printf("Pour un texte court de 9 mots, la tokenisation devrait prendre <0.5ms\n");
    printf("L'inférence ONNX pour 1 texte devrait prendre ~2-5ms\n");
    printf("Le pooling/normalisation devrait prendre <0.1ms\n");
    printf("Total attendu: ~3-6ms, mesure: ~10-16ms\n");

    lembed_text_embedding_free(embedder);
    printf("\n=== Termine ===\n");
    return 0;
}
