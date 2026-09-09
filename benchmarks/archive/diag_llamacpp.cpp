/*
 * Test diagnostique : verifie si les embeddings sont reellement calcules
 * Toute sortie va sur stderr (non bufferise) pour ne rien perdre en cas de crash
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

static float checksum(const float* v, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i] * (float)(i + 1);
    return s;
}

static float sum(const float* v, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i];
    return s;
}

#define LOG(...) fprintf(stderr, __VA_ARGS__)

int main(int argc, char** argv) {
    const char* model_path = R"(C:\Users\david\.cache\libembedding\models--BAAI-bge-small-en-v1.5\bge-small-en-v1.5-Q8_0.gguf)";
    if (argc > 1) model_path = argv[1];

    LOG("=== Test diagnostique llama.cpp ===\n\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.show_download_progress = 0;

    LOG("Creating embedder...\n");
    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(model_path, &opts, &embedder);
    LOG("Status: %d, embedder=%p\n", s, (void*)embedder);
    if (s != LEMBED_OK) {
        LOG("Error: %s\n", lembed_last_error());
        return 1;
    }

    int dim = lembed_text_embedding_dim(embedder);
    LOG("dim=%d\n\n", dim);

    /* Test 1: Textes differents doivent donner embeddings differents */
    LOG("--- Test 1: Textes differents ---\n");
    {
        const char* texts[] = {
            "chat",
            "chien",
            "voiture",
            "ordinateur",
            "maison",
            "arbre",
            "soleil",
            "lune"
        };
        int n = 8;
        lembed_embeddings_t result = {0};
        s = lembed_text_embedding_embed(embedder, texts, n, n, &result);
        LOG("Status: %d, num_embeddings=%d\n", s, result.num_embeddings);
        if (s != LEMBED_OK) {
            LOG("Error: %s\n", lembed_last_error());
            return 1;
        }

        for (int i = 0; i < n; i++) {
            float* emb = result.data + (size_t)i * dim;
            LOG("  [%d] sum=%.4f checksum=%.4f  (%s)\n",
                i, sum(emb, dim), checksum(emb, dim), texts[i]);
        }
        lembed_embeddings_free(&result);
    }

    /* Test 2: Meme texte repete N fois doit donner embeddings identiques */
    LOG("\n--- Test 2: Meme texte repete 4 fois ---\n");
    {
        const char* texts[] = {
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog."
        };
        int n = 4;
        lembed_embeddings_t result = {0};
        s = lembed_text_embedding_embed(embedder, texts, n, n, &result);
        LOG("Status: %d, num_embeddings=%d\n", s, result.num_embeddings);
        if (s != LEMBED_OK) {
            LOG("Error: %s\n", lembed_last_error());
            return 1;
        }

        for (int i = 0; i < n; i++) {
            float* emb = result.data + (size_t)i * dim;
            LOG("  [%d] sum=%.4f checksum=%.4f\n", i, sum(emb, dim), checksum(emb, dim));
        }
        lembed_embeddings_free(&result);
    }

    /* Test 3: Embedding individuel vs batch pour un texte */
    LOG("\n--- Test 3: Individuel vs Batch (MSE) ---\n");
    {
        const char* text = "The quick brown fox jumps over the lazy dog.";
        lembed_embeddings_t r1 = {0};
        lembed_text_embedding_embed(embedder, &text, 1, 1, &r1);

        const char* texts[4] = { text, text, text, text };
        lembed_embeddings_t r4 = {0};
        lembed_text_embedding_embed(embedder, texts, 4, 4, &r4);

        for (int i = 0; i < 4; i++) {
            float* a = r1.data;
            float* b = r4.data + (size_t)i * dim;
            float mse = 0;
            for (int j = 0; j < dim; j++) {
                float d = a[j] - b[j];
                mse += d * d;
            }
            mse /= dim;
            LOG("  r1[0] vs r4[%d]: MSE=%.8f\n", i, mse);
        }
        lembed_embeddings_free(&r1);
        lembed_embeddings_free(&r4);
    }

    /* Test 4: Textes tres differents, batch de 8 */
    LOG("\n--- Test 4: 8 textes tres differents ---\n");
    {
        const char* texts[] = {
            "artificial intelligence machine learning",
            "cooking recipe pasta tomato basil",
            "quantum physics particle wave duality",
            "ancient Rome empire gladiator colosseum",
            "ocean marine biology coral reef fish",
            "stock market investment portfolio dividend",
            "soccer football goal penalty championship",
            "music symphony orchestra conductor Beethoven"
        };
        int n = 8;
        lembed_embeddings_t result = {0};
        s = lembed_text_embedding_embed(embedder, texts, n, n, &result);
        LOG("Status: %d, num_embeddings=%d\n", s, result.num_embeddings);
        if (s != LEMBED_OK) {
            LOG("Error: %s\n", lembed_last_error());
            return 1;
        }

        for (int i = 0; i < n; i++) {
            float* emb = result.data + (size_t)i * dim;
            LOG("  [%d] sum=%.4f  (%s)\n", i, sum(emb, dim), texts[i]);
        }

        int identical = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                float* a = result.data + (size_t)i * dim;
                float* b = result.data + (size_t)j * dim;
                float mse = 0;
                for (int k = 0; k < dim; k++) {
                    float d = a[k] - b[k];
                    mse += d * d;
                }
                mse /= dim;
                if (mse < 0.0001f) {
                    LOG("  WARNING: [%d] et [%d] identiques (MSE=%.8f)\n", i, j, mse);
                    identical++;
                }
            }
        }
        if (identical == 0) LOG("  Tous les embeddings sont differents OK\n");
        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(embedder);
    LOG("\n=== Termine ===\n");
    return 0;
}


