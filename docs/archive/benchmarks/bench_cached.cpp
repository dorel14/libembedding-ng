/*
 * libembedding benchmark - utilise le modele bge-small-en-v1.5 deja en cache
 * Aucun telechargement necessaire. Utilise le cache HuggingFace standard.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static double peak_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}
static double current_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}
#else
static double peak_rss_mb() { return 0.0; }
static double current_rss_mb() { return 0.0; }
#endif

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    size_t n = v.size();
    if (n == 0) return 0.0;
    std::sort(v.begin(), v.end());
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
}

static std::vector<std::string> load_corpus(const char* path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

int main(int argc, char** argv) {
    const char* corpus_path = "corpus.txt";
    if (argc > 1) corpus_path = argv[1];

    auto corpus = load_corpus(corpus_path);
    if (corpus.empty()) {
        fprintf(stderr, "Erreur: corpus vide ou fichier non trouve: %s\n", corpus_path);
        fprintf(stderr, "Usage: %s [chemin_vers_corpus.txt]\n", argv[0]);
        return 1;
    }

    printf("Corpus: %d textes charges\n", (int)corpus.size());

    /* Prepare C-string pointers */
    std::vector<const char*> texts;
    for (auto& s : corpus) texts.push_back(s.c_str());

    const int WARMUP = 2;
    const int ITERS = 5;
    const int BATCH_SIZES[] = {1, 8, 32, 128, 256, 512};
    const int NUM_BATCH_SIZES = 6;

    /* ── Benchmark A: Model load time ── */
    printf("\n--- Chargement du modele ---\n");
    double rss_before = current_rss_mb();

    std::vector<double> load_times;
    lembed_text_embedding_t* embedder = nullptr;

    for (int i = 0; i < WARMUP + ITERS; i++) {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.num_threads = 0; /* auto */
        opts.offline = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* tmp = nullptr;
        double t0 = now_ms();
        lembed_status_t s = lembed_text_embedding_create(&opts, &tmp);
        double t1 = now_ms();

        if (s != LEMBED_OK) {
            fprintf(stderr, "Erreur de chargement: %s\n", lembed_last_error());
            return 1;
        }
        if (i >= WARMUP) load_times.push_back(t1 - t0);
        if (i == WARMUP + ITERS - 1) embedder = tmp;
        else lembed_text_embedding_free(tmp);
    }

    double rss_after_load = current_rss_mb();
    printf("Load time: %.1f ms (mediane sur %d runs)\n", median(load_times), ITERS);
    printf("RAM avant chargement: %.1f MB\n", rss_before);
    printf("RAM apres chargement: %.1f MB (+%.1f MB)\n", rss_after_load, rss_after_load - rss_before);
    printf("Dimension: %d\n", lembed_text_embedding_dim(embedder));

    /* ── Benchmark B: Single text latency ── */
    printf("\n--- Latence texte unique ---\n");
    std::vector<double> single_times;
    const char* single_text[] = {"The quick brown fox jumps over the lazy dog."};

    for (int i = 0; i < WARMUP + ITERS; i++) {
        lembed_embeddings_t result = {0};
        double t0 = now_ms();
        lembed_text_embedding_embed(embedder, single_text, 1, 1, &result);
        double t1 = now_ms();
        if (i >= WARMUP) single_times.push_back(t1 - t0);
        lembed_embeddings_free(&result);
    }
    printf("Latence: %.2f ms (mediane)\n", median(single_times));

    /* ── Benchmark C: Batch throughput ── */
    printf("\n--- Debit par batch ---\n");
    printf("%-10s %-15s %-15s %-15s\n", "Batch", "Temps (ms)", "Textes/sec", "Per texte (ms)");
    printf("-----------------------------------------------------------\n");

    for (int bi = 0; bi < NUM_BATCH_SIZES; bi++) {
        int bsz = BATCH_SIZES[bi];
        int n = std::min(bsz, (int)texts.size());
        std::vector<double> times;

        for (int i = 0; i < WARMUP + ITERS; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), n, bsz, &result);
            double t1 = now_ms();
            if (i >= WARMUP) times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }

        double med_ms = median(times);
        double texts_per_sec = (double)n / (med_ms / 1000.0);
        double per_text = med_ms / (double)n;
        printf("%-10d %-15.2f %-15.1f %-15.3f\n", bsz, med_ms, texts_per_sec, per_text);
    }

    /* ── Benchmark D: Full corpus throughput ── */
    printf("\n--- Debit corpus complet (%d textes) ---\n", (int)corpus.size());
    for (int bi = 0; bi < NUM_BATCH_SIZES; bi++) {
        int bsz = BATCH_SIZES[bi];
        std::vector<double> times;

        for (int i = 0; i < WARMUP + ITERS; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), (int)texts.size(), bsz, &result);
            double t1 = now_ms();
            if (i >= WARMUP) times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }

        double med_ms = median(times);
        double texts_per_sec = (double)texts.size() / (med_ms / 1000.0);
        printf("batch=%-5d  %.1f textes/sec  (%.1f ms total)\n", bsz, texts_per_sec, med_ms);
    }

    double peak_rss = peak_rss_mb();
    double final_rss = current_rss_mb();

    printf("\n--- Memoire ---\n");
    printf("RSS actuel: %.1f MB\n", final_rss);
    printf("RSS peak:   %.1f MB\n", peak_rss);
    printf("Modele:     %s\n", lembed_text_embedding_model_name(embedder));

    lembed_text_embedding_free(embedder);
    printf("\nOK.\n");
    return 0;
}
