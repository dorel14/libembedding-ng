/*
 * llama.cpp inference profiling — step-by-step breakdown
 * Uses only high-level API (no direct LlamaSession access).
 * Measures: full pipeline at various batch sizes, single-text overhead.
 * Breakdown by difference: inference = full_pipeline - tokenization.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct BenchResult {
    double median_ms;
    double ms_per_text;
    double texts_per_sec;
};

BenchResult run_benchmark(lembed_text_embedding_t* embedder,
                          const std::vector<const char*>& texts, int iters) {
    std::vector<double> times;
    int n = (int)texts.size();
    for (int i = 0; i < iters; i++) {
        lembed_embeddings_t result = {0};
        double t0 = now_ms();
        lembed_text_embedding_embed(embedder, texts.data(), n, n, &result);
        double t1 = now_ms();
        times.push_back(t1 - t0);
        lembed_embeddings_free(&result);
    }
    BenchResult r;
    r.median_ms = median(times);
    r.ms_per_text = r.median_ms / n;
    r.texts_per_sec = (double)n / (r.median_ms / 1000.0);
    return r;
}

int main(int argc, char** argv) {
    const char* model_path = R"(C:\Users\david\.cache\libembedding\models--BAAI-bge-small-en-v1.5\bge-small-en-v1.5-Q8_0.gguf)";
    if (argc > 1) model_path = argv[1];

    printf("=== llama.cpp Inference Profiling (step-by-step) ===\n\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(model_path, &opts, &embedder);
    if (s != LEMBED_OK) {
        fprintf(stderr, "Error: %s\n", lembed_last_error());
        return 1;
    }

    printf("Model: %s (dim=%d)\n", lembed_text_embedding_model_name(embedder),
           lembed_text_embedding_dim(embedder));

    int dim = lembed_text_embedding_dim(embedder);

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        lembed_embeddings_t w = {0};
        const char* dummy = "test";
        lembed_text_embedding_embed(embedder, &dummy, 1, 1, &w);
        lembed_embeddings_free(&w);
    }

    /* 1. Single text (batch=1) — measures fixed overhead */
    printf("\n--- 1. batch=1 (overhead fixe) ---\n");
    {
        const char* text = "The quick brown fox jumps over the lazy dog.";
        std::vector<const char*> texts = {text};
        BenchResult r = run_benchmark(embedder, texts, 50);
        printf("%.2f ms total = %.3f ms/text\n", r.median_ms, r.ms_per_text);
    }

    /* 2. Batch=8 */
    printf("\n--- 2. batch=8 ---\n");
    {
        std::vector<const char*> texts;
        for (int i = 0; i < 8; i++) texts.push_back("The quick brown fox jumps over the lazy dog.");
        BenchResult r = run_benchmark(embedder, texts, 50);
        printf("%.2f ms total = %.3f ms/text, %.1f texts/sec\n", r.median_ms, r.ms_per_text, r.texts_per_sec);
    }

    /* 3. Batch=32 */
    printf("\n--- 3. batch=32 ---\n");
    {
        std::vector<const char*> texts;
        for (int i = 0; i < 32; i++) texts.push_back("The quick brown fox jumps over the lazy dog.");
        BenchResult r = run_benchmark(embedder, texts, 30);
        printf("%.2f ms total = %.3f ms/text, %.1f texts/sec\n", r.median_ms, r.ms_per_text, r.texts_per_sec);
    }

    /* 4. Batch=64 */
    printf("\n--- 4. batch=64 ---\n");
    {
        std::vector<const char*> texts;
        for (int i = 0; i < 64; i++) texts.push_back("The quick brown fox jumps over the lazy dog.");
        BenchResult r = run_benchmark(embedder, texts, 20);
        printf("%.2f ms total = %.3f ms/text, %.1f texts/sec\n", r.median_ms, r.ms_per_text, r.texts_per_sec);
    }

    /* 5. Batch=128 */
    printf("\n--- 5. batch=128 ---\n");
    {
        std::vector<const char*> texts;
        for (int i = 0; i < 128; i++) texts.push_back("The quick brown fox jumps over the lazy dog.");
        BenchResult r = run_benchmark(embedder, texts, 10);
        printf("%.2f ms total = %.3f ms/text, %.1f texts/sec\n", r.median_ms, r.ms_per_text, r.texts_per_sec);
    }

    /* 6. Variable text lengths at batch=1 */
    printf("\n--- 6. Longueur de texte variable (batch=1) ---\n");
    {
        struct Test { const char* name; const char* text; };
        Test tests[] = {
            {"Court (2 mots)", "Hello world"},
            {"Moyen (15 mots)", "The quick brown fox jumps over the lazy dog. This is a test for embedding."},
            {"Long (80 mots)", "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves."},
        };
        for (int t = 0; t < 3; t++) {
            std::vector<const char*> texts = {tests[t].text};
            BenchResult r = run_benchmark(embedder, texts, 20);
            printf("%-20s: %.2f ms\n", tests[t].name, r.median_ms);
        }
    }

    /* 7. Output tensor size info */
    printf("\n--- 7. Taille du tensor de sortie ---\n");
    printf("dim=%d, taille embedding: %.2f KB (float32)\n",
           dim, (double)dim * 4 / 1024);
    printf("taille batch=64: %.2f KB (float32)\n",
           (double)64 * dim * 4 / 1024);
    printf("taille batch=128: %.2f KB (float32)\n",
           (double)128 * dim * 4 / 1024);

    /* 8. Thread scaling at batch=64 */
    printf("\n--- 8. Scaling threads (batch=64) ---\n");
    printf("%-10s %-15s %-15s %-15s\n", "Threads", "Total (ms)", "ms/text", "Texts/sec");
    printf("----------------------------------------------------\n");
    {
        int thread_counts[] = {1, 2, 4, 8, 16};
        for (int ti = 0; ti < 5; ti++) {
            int nt = thread_counts[ti];
            lembed_text_embedding_free(embedder);
            opts.num_threads = nt;
            s = lembed_text_embedding_create_from_gguf_path(model_path, &opts, &embedder);
            if (s != LEMBED_OK) continue;

            /* warmup */
            for (int i = 0; i < 3; i++) {
                lembed_embeddings_t w = {0};
                const char* dummy = "test";
                lembed_text_embedding_embed(embedder, &dummy, 1, 1, &w);
                lembed_embeddings_free(&w);
            }

            std::vector<const char*> texts;
            for (int i = 0; i < 64; i++) texts.push_back("The quick brown fox jumps over the lazy dog.");
            BenchResult r = run_benchmark(embedder, texts, 10);
            printf("%-10d %-15.2f %-15.3f %-15.1f\n", nt, r.median_ms, r.ms_per_text, r.texts_per_sec);
        }
    }

    /* 9. Key insight about llama.cpp batching */
    printf("\n--- 9. Analyse cle ---\n");
    printf("llama.cpp traite les textes UN PAR UN (pas de vrai batch).\n");
    printf("Le scaling lineaire du temps avec la taille du batch est attendu.\n");
    printf("Si temps(batch=N) ~= N * temps(batch=1), c'est du sequentiel pur.\n");
    printf("Optimization possible: vrais batches multi-textes avec llama_batch.\n");

    lembed_text_embedding_free(embedder);
    printf("\n=== Termine ===\n");
    return 0;
}

