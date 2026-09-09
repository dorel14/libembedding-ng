/*
 * Verification reproductibilite + investigation BGE-small-Q anomaly
 * 3 runs, moyenne, ecart-type
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

static double stddev(const std::vector<double>& v, double m) {
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}

void embed_task(lembed_text_embedding_t* emb, const char** texts, int n, int bsz) {
    lembed_embeddings_t res = {0};
    lembed_text_embedding_embed(emb, texts, n, bsz, &res);
    lembed_embeddings_free(&res);
}

struct Result {
    double mean_tps;
    double std_tps;
    double mean_ms;
};

    Result benchmark(lembed_text_model_t model, int nsess, int nthreads, int bsz,
                 const std::vector<const char*>& texts, int niter = 5) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = model;
    opts.num_threads = nthreads;
    opts.offline = 1;
    opts.show_download_progress = 0;

    std::vector<lembed_text_embedding_t*> emb(nsess);
    for (int i = 0; i < nsess; i++) {
        if (lembed_text_embedding_create(&opts, &emb[i]) != LEMBED_OK) {
            fprintf(stderr, "Erreur: %s\n", lembed_last_error());
            return {0, 0, 0};
        }
    }

    int sub = bsz / nsess;

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < nsess; j++) {
            lembed_embeddings_t w = {0};
            lembed_text_embedding_embed(emb[j], texts.data() + j*sub, sub, sub, &w);
            lembed_embeddings_free(&w);
        }
    }

    /* Benchmark */
    std::vector<double> tps_samples;
    std::vector<double> ms_samples;

    for (int i = 0; i < niter; i++) {
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

        double ms = t1 - t0;
        double tps = (double)bsz / (ms / 1000.0);
        ms_samples.push_back(ms);
        tps_samples.push_back(tps);
    }

    for (int i = 0; i < nsess; i++) lembed_text_embedding_free(emb[i]);

    Result r;
    r.mean_tps = mean(tps_samples);
    r.std_tps = stddev(tps_samples, r.mean_tps);
    r.mean_ms = mean(ms_samples);
    return r;
}

int main() {
    printf("=== Verification Reproductibilite ===\n");
    printf("10 iterations par config, moyenne +/- ecart-type\n\n");

    int bsz = 64;
    const char* text = "The quick brown fox jumps over the lazy dog.";
    std::vector<const char*> texts(bsz, text);

    struct Test {
        const char* name;
        lembed_text_model_t model;
        int nsess;
        int nthreads;
    };

    Test tests[] = {
        {"MiniLM-L6-v2      ", LEMBED_TEXT_ALL_MINILM_L6_V2, 8, 1},
        {"MiniLM-L6-v2-Q    ", LEMBED_TEXT_ALL_MINILM_L6_V2_Q, 8, 1},
        {"BGE-small-en      ", LEMBED_TEXT_BGE_SMALL_EN_V15, 8, 1},
        {"BGE-small-en-Q    ", LEMBED_TEXT_BGE_SMALL_EN_V15_Q, 8, 1},
    };

    int ntests = sizeof(tests) / sizeof(tests[0]);

    printf("%-22s %-12s %-12s %-12s\n", "Modele", "Docs/s", "Ecart-type", "ms/batch");
    printf("-----------------------------------------------------------\n");

    for (int i = 0; i < ntests; i++) {
        Result r = benchmark(tests[i].model, tests[i].nsess, tests[i].nthreads,
                             bsz, texts, 10);

        printf("%-22s %-12.1f %-12.1f %-12.1f\n",
               tests[i].name, r.mean_tps, r.std_tps, r.mean_ms);
    }

    printf("\n=== Termine ===\n");
    return 0;
}
