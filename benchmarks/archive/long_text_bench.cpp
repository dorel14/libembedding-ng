/*
 * Long text benchmark - throughput only
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
static double now_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
static double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        clk::now().time_since_epoch()
    );
    return (double)us.count() / 1000.0;
}
#endif

std::string make_text(int n_tokens, int seed = 0) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "algorithms", "process", "data", "efficiently",
        "natural", "language", "processing", "enables", "understanding", "semantic",
        "embeddings", "represent", "meaningful", "vector", "representations",
    };
    int nwords = sizeof(words) / sizeof(words[0]);
    std::string result;
    for (int i = 0; i < n_tokens; i++) {
        if (i > 0) result += " ";
        result += words[(i + seed) % nwords];
    }
    return result;
}

int main() {
    printf("============================================================\n");
    printf("  LONG TEXT THROUGHPUT BENCHMARK\n");
    printf("  Model: MiniLM-L6-v2 (FP32)\n");
    printf("  Config: 8 workers x 1 thread\n");
    printf("============================================================\n\n");

    printf("%-8s %-12s %-12s %-12s\n", "Tokens", "Docs/s", "ms/batch", "ms/text");
    printf("------------------------------------------------\n");

    int token_counts[] = {16, 64, 128, 256};
    int n_counts = sizeof(token_counts) / sizeof(token_counts[0]);

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;
    opts.num_threads = 1;
    opts.offline = 1;
    opts.show_download_progress = 0;

    const int N_WORKERS = 8;
    const int N_TEXTS = 32;  /* texts per benchmark */

    for (int ci = 0; ci < n_counts; ci++) {
        int ntok = token_counts[ci];

        /* Generate corpus */
        std::vector<std::string> texts;
        for (int i = 0; i < N_TEXTS; i++)
            texts.push_back(make_text(ntok, i * 7));

        /* Create workers */
        std::vector<lembed_text_embedding_t*> emb(N_WORKERS);
        for (int i = 0; i < N_WORKERS; i++)
            lembed_text_embedding_create(&opts, &emb[i]);

        int per = texts.size() / N_WORKERS;

        /* Warmup (3 batches) */
        for (int w = 0; w < 3; w++) {
            std::vector<std::thread> th;
            for (int i = 0; i < N_WORKERS; i++) {
                th.emplace_back([&emb, i, &texts, per]() {
                    std::vector<const char*> ct;
                    for (int k = 0; k < per; k++)
                        ct.push_back(texts[i*per+k].c_str());
                    lembed_embeddings_t r = {0};
                    lembed_text_embedding_embed(emb[i], ct.data(), per, per, &r);
                    lembed_embeddings_free(&r);
                });
            }
            for (auto& t : th) t.join();
        }

        /* Benchmark: 5 batches */
        double t0 = now_ms();
        int n_batches = 5;
        for (int b = 0; b < n_batches; b++) {
            std::vector<std::thread> th;
            for (int i = 0; i < N_WORKERS; i++) {
                th.emplace_back([&emb, i, &texts, per]() {
                    std::vector<const char*> ct;
                    for (int k = 0; k < per; k++)
                        ct.push_back(texts[i*per+k].c_str());
                    lembed_embeddings_t r = {0};
                    lembed_text_embedding_embed(emb[i], ct.data(), per, per, &r);
                    lembed_embeddings_free(&r);
                });
            }
            for (auto& t : th) t.join();
        }
        double t1 = now_ms();

        double total_texts = n_batches * texts.size();
        double total_ms = t1 - t0;
        double per_batch = total_ms / n_batches;
        double per_text = per_batch / texts.size();
        double docs_per_sec = total_texts / (total_ms / 1000.0);

        printf("%-8d %-12.0f %-12.1f %-12.1f\n",
               ntok, docs_per_sec, per_batch, per_text);

        for (int i = 0; i < N_WORKERS; i++)
            lembed_text_embedding_free(emb[i]);
    }

    printf("\nDone.\n");
    return 0;
}
