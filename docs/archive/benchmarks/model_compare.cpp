/*
 * Benchmark comparatif de tous les modeles
 * Teste les modeles quantized et multilingues
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void embed_subbatch(lembed_text_embedding_t* embedder, const char** texts, int n, int bsz) {
    lembed_embeddings_t res = {0};
    lembed_text_embedding_embed(embedder, texts, n, bsz, &res);
    lembed_embeddings_free(&res);
}

struct ModelConfig {
    const char* name;
    lembed_text_model_t model;
    int expected_dim;
};

struct BenchResult {
    double time_ms;
    double texts_per_sec;
    double peak_rss_mb;
    int dim;
};

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static double get_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
}
#else
static double get_rss_mb() { return 0.0; }
#endif

BenchResult benchmark_model(ModelConfig& cfg, int bsz, int nsess, int nthreads,
                            const std::vector<const char*>& texts) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = cfg.model;
    opts.num_threads = nthreads;
    opts.offline = 0; /* allow download */
    opts.show_download_progress = 0;

    double rss_before = get_rss_mb();

    std::vector<lembed_text_embedding_t*> emb(nsess);
    for (int i = 0; i < nsess; i++) {
        lembed_status_t s = lembed_text_embedding_create(&opts, &emb[i]);
        if (s != LEMBED_OK) {
            fprintf(stderr, "  Erreur: %s\n", lembed_last_error());
            return {0, 0, 0, 0};
        }
    }

    double rss_after_load = get_rss_mb();
    int sub = bsz / nsess;

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < nsess; j++) {
            lembed_embeddings_t w = {0};
            lembed_text_embedding_embed(emb[j], texts.data() + j*sub, sub, sub, &w);
            lembed_embeddings_free(&w);
        }
    }

    /* Benchmark */
    std::vector<double> times;
    for (int i = 0; i < 5; i++) {
        double t0 = now_ms();
        std::vector<std::thread> th;
        for (int j = 0; j < nsess; j++) {
            th.emplace_back([&, j]() {
                lembed_embeddings_t res = {0};
                lembed_status_t s = lembed_text_embedding_embed(emb[j], texts.data() + j*sub, sub, sub, &res);
                lembed_embeddings_free(&res);
            });
        }
        for (auto& t : th) t.join();
        double t1 = now_ms();
        times.push_back(t1 - t0);
    }

    int dim = lembed_text_embedding_dim(emb[0]);
    for (int i = 0; i < nsess; i++) lembed_text_embedding_free(emb[i]);

    BenchResult r;
    r.time_ms = median(times);
    r.texts_per_sec = (double)bsz / (r.time_ms / 1000.0);
    r.peak_rss_mb = rss_after_load - rss_before;
    r.dim = dim;
    return r;
}

int main() {
    printf("=== Benchmark Comparatif Modeles ===\n\n");

    int bsz = 64;
    const char* text = "The quick brown fox jumps over the lazy dog. This is a test sentence.";
    std::vector<const char*> texts(bsz, text);

    /* Modeles a tester */
    ModelConfig models[] = {
        {"MiniLM-L6-v2", LEMBED_TEXT_ALL_MINILM_L6_V2, 384},
        {"MiniLM-L6-v2-Q", LEMBED_TEXT_ALL_MINILM_L6_V2_Q, 384},
        {"BGE-small-en", LEMBED_TEXT_BGE_SMALL_EN_V15, 384},
        {"BGE-small-en-Q", LEMBED_TEXT_BGE_SMALL_EN_V15_Q, 384},
        {"BGE-base-en", LEMBED_TEXT_BGE_BASE_EN_V15, 768},
        {"BGE-base-en-Q", LEMBED_TEXT_BGE_BASE_EN_V15_Q, 768},
        {"Multilingual-MiniLM", LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2, 384},
        {"Multilingual-E5-small", LEMBED_TEXT_MULTILINGUAL_E5_SMALL, 384},
        {"Multilingual-E5-base", LEMBED_TEXT_MULTILINGUAL_E5_BASE, 768},
        {"MiniLM-L12-v2", LEMBED_TEXT_ALL_MINILM_L12_V2, 384},
        {"Nomic-embed-v1.5", LEMBED_TEXT_NOMIC_EMBED_TEXT_V15, 768},
    };
    int nmodels = sizeof(models) / sizeof(models[0]);

    /* Configurations a tester */
    struct Config { int nsess; int nthreads; };
    Config configs[] = {
        {1, 4},
        {8, 1},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    printf("%-25s %-6s %-12s %-12s %-10s %-8s\n",
           "Modele", "Dim", "Config", "Docs/s", "RAM (MB)", "Status");
    printf("------------------------------------------------------------------------\n");

    for (int mi = 0; mi < nmodels; mi++) {
        printf("%-25s ", models[mi].name);

        for (int ci = 0; ci < nconfigs; ci++) {
            int nsess = configs[ci].nsess;
            int nthreads = configs[ci].nthreads;

            BenchResult r = benchmark_model(models[mi], bsz, nsess, nthreads, texts);

            if (r.texts_per_sec > 0) {
                printf("%-6d %-12s %-12.1f %-10.1f %-8s\n",
                       r.dim,
                       ci == 0 ? "1x4" : "8x1",
                       r.texts_per_sec,
                       r.peak_rss_mb,
                       "OK");
            } else {
                printf("%-6s %-12s %-12s %-10s %-8s\n",
                       "?", "1x4", "-", "-", "FAIL");
            }
        }
        printf("\n");
    }

    printf("=== Termine ===\n");
    return 0;
}
