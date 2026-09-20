/*
 * Profilage par etape pour identifier le goulot d'etranglement
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static double current_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
}
#else
static double current_rss_mb() { return 0.0; }
#endif

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    printf("=== Profilage des performances ===\n\n");

    /* Setup */
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 0; /* auto */
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);
    if (s != LEMBED_OK) {
        fprintf(stderr, "Erreur: %s\n", lembed_last_error());
        return 1;
    }

    printf("Modele: %s (dim=%d)\n", lembed_text_embedding_model_name(embedder),
           lembed_text_embedding_dim(embedder));

    /* Test texts of various sizes */
    const char* short_text = "Hello world";
    const char* medium_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for embedding.";
    const char* long_text = "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.";

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        lembed_embeddings_t warmup = {0};
        lembed_text_embedding_embed(embedder, &short_text, 1, 1, &warmup);
        lembed_embeddings_free(&warmup);
    }

    /* Profile single texts */
    struct TestCase { const char* name; const char* text; };
    TestCase tests[] = {
        {"Court (2 mots)", short_text},
        {"Moyen (15 mots)", medium_text},
        {"Long (80 mots)", long_text},
    };

    printf("\n--- Latence par taille de texte ---\n");
    printf("%-20s %-12s %-12s\n", "Texte", "Temps (ms)", "Per token (ms)");
    printf("----------------------------------------------\n");

    for (int t = 0; t < 3; t++) {
        const char* text = tests[t].text;
        std::vector<double> times;

        for (int i = 0; i < 20; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, &text, 1, 1, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }

        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        printf("%-20s %-12.2f\n", tests[t].name, med);
    }

    /* Profile batch scaling */
    printf("\n--- Scaling batch (meme texte repete) ---\n");
    printf("%-10s %-15s %-15s %-15s\n", "Batch", "Total (ms)", "Textes/sec", "Par texte (ms)");
    printf("-----------------------------------------------------------\n");

    const int batch_sizes[] = {1, 4, 16, 64, 128, 256};
    for (int bi = 0; bi < 6; bi++) {
        int bsz = batch_sizes[bi];
        std::vector<const char*> texts(bsz, medium_text);
        std::vector<double> times;

        for (int i = 0; i < 5; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), bsz, bsz, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }

        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        double tps = (double)bsz / (med / 1000.0);
        printf("%-10d %-15.2f %-15.1f %-15.3f\n", bsz, med, tps, med / bsz);
    }

    /* Profile memory usage per batch size */
    printf("\n--- Memoire par taille de batch ---\n");
    printf("%-10s %-15s %-15s\n", "Batch", "RSS (MB)", "Delta (MB)");
    printf("----------------------------------------------\n");

    double rss_base = current_rss_mb();
    for (int bi = 0; bi < 6; bi++) {
        int bsz = batch_sizes[bi];
        std::vector<const char*> texts(bsz, medium_text);

        lembed_embeddings_t result = {0};
        lembed_text_embedding_embed(embedder, texts.data(), bsz, bsz, &result);
        double rss = current_rss_mb();
        lembed_embeddings_free(&result);

        printf("%-10d %-15.1f %-15.1f\n", bsz, rss, rss - rss_base);
    }

    /* Thread scaling test */
    printf("\n--- Scaling threads (batch=64) ---\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Total (ms)", "Textes/sec");
    printf("----------------------------------------------\n");

    int thread_counts[] = {1, 2, 4, 8, 16};
    for (int ti = 0; ti < 5; ti++) {
        int nt = thread_counts[ti];

        lembed_text_embedding_free(embedder);
        opts.num_threads = nt;
        s = lembed_text_embedding_create(&opts, &embedder);
        if (s != LEMBED_OK) continue;

        int bsz = 64;
        std::vector<const char*> texts(bsz, medium_text);
        std::vector<double> times;

        /* warmup */
        for (int i = 0; i < 2; i++) {
            lembed_embeddings_t w = {0};
            lembed_text_embedding_embed(embedder, texts.data(), bsz, bsz, &w);
            lembed_embeddings_free(&w);
        }

        for (int i = 0; i < 5; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), bsz, bsz, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }

        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        double tps = (double)bsz / (med / 1000.0);
        printf("%-10d %-15.2f %-15.1f\n", nt, med, tps);
    }

    lembed_text_embedding_free(embedder);
    printf("\n=== Profilage termine ===\n");
    return 0;
}
