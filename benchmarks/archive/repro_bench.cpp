/*
 * Benchmark reproductible avec vrai corpus multilingue
 * Utilise corpus.txt (392 phrases EN) + corpus multilingue integre
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}

static double stddev(const std::vector<double>& v, double m) {
    double s = 0; for (double x : v) s += (x-m)*(x-m); return std::sqrt(s/v.size());
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

struct Result {
    double mean_tps, std_tps, mean_ms;
};

Result benchmark(lembed_text_embedding_t* emb, const std::vector<std::string>& texts,
                 int bsz, int niter = 10) {
    /* Warmup */
    for (int i = 0; i < 5; i++) {
        int n = std::min(4, (int)texts.size());
        std::vector<const char*> ct;
        for (int j = 0; j < n; j++) ct.push_back(texts[j].c_str());
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(emb, ct.data(), n, n, &r);
        lembed_embeddings_free(&r);
    }

    std::vector<double> tps, ms;
    int n = texts.size();
    for (int i = 0; i < niter; i++) {
        double t0 = now_ms();
        for (int off = 0; off < n; off += bsz) {
            int cnt = std::min(bsz, n - off);
            std::vector<const char*> ct;
            for (int j = off; j < off+cnt; j++) ct.push_back(texts[j].c_str());
            lembed_embeddings_t r = {0};
            lembed_text_embedding_embed(emb, ct.data(), cnt, cnt, &r);
            lembed_embeddings_free(&r);
        }
        double t1 = now_ms();
        ms.push_back(t1 - t0);
        tps.push_back((double)n / ((t1-t0)/1000.0));
    }

    Result r;
    r.mean_tps = mean(tps);
    r.std_tps = stddev(tps, r.mean_tps);
    r.mean_ms = mean(ms);
    return r;
}

int main(int argc, char** argv) {
    printf("=== Benchmark Reproductible ===\n\n");

    /* Load corpus */
    const char* corpus_path = argc > 1 ? argv[1] : "corpus.txt";
    auto corpus = load_corpus(corpus_path);
    if (corpus.empty()) {
        fprintf(stderr, "Erreur: corpus vide ou non trouve: %s\n", corpus_path);
        return 1;
    }
    printf("Corpus: %d phrases chargees depuis %s\n", (int)corpus.size(), corpus_path);

    /* Stats sur le corpus */
    double avg_len = 0;
    for (auto& s : corpus) avg_len += s.size();
    avg_len /= corpus.size();
    printf("Longueur moyenne: %.0f caracteres\n\n", avg_len);

    /* Modeles a tester */
    struct Test {
        const char* name;
        lembed_text_model_t model;
        int dim;
    };
    Test tests[] = {
        {"MiniLM-L6-v2    ", LEMBED_TEXT_ALL_MINILM_L6_V2, 384},
        {"MiniLM-L6-v2-Q  ", LEMBED_TEXT_ALL_MINILM_L6_V2_Q, 384},
        {"BGE-small-en    ", LEMBED_TEXT_BGE_SMALL_EN_V15, 384},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    /* Configurations */
    struct Config { int nsess; int nthreads; const char* label; };
    Config configs[] = {
        {1, 4, "1x4"},
        {8, 1, "8x1"},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    printf("%-20s %-8s %-12s %-12s %-12s\n", "Modele", "Config", "Docs/s", "Ecart-type", "ms total");
    printf("--------------------------------------------------------------------\n");

    for (int ti = 0; ti < ntests; ti++) {
        printf("%-20s\n", tests[ti].name);

        for (int ci = 0; ci < nconfigs; ci++) {
            int nsess = configs[ci].nsess;
            int nthreads = configs[ci].nthreads;

            /* Create workers */
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = tests[ti].model;
            opts.num_threads = nthreads;
            opts.offline = 1;
            opts.show_download_progress = 0;

            std::vector<lembed_text_embedding_t*> emb(nsess);
            for (int i = 0; i < nsess; i++) {
                if (lembed_text_embedding_create(&opts, &emb[i]) != LEMBED_OK) {
                    fprintf(stderr, "  Erreur: %s\n", lembed_last_error());
                    return 1;
                }
            }

            /* Warmup all workers */
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < nsess; j++) {
                    lembed_embeddings_t w = {0};
                    std::vector<const char*> ct;
                    int per = corpus.size() / nsess;
                    for (int k = 0; k < per && (j*per+k) < (int)corpus.size(); k++)
                        ct.push_back(corpus[j*per+k].c_str());
                    if (!ct.empty())
                        lembed_text_embedding_embed(emb[j], ct.data(), (int)ct.size(), (int)ct.size(), &w);
                    lembed_embeddings_free(&w);
                }
            }

            /* Benchmark avec workers paralleles */
            std::vector<double> tps_samples, ms_samples;
            int per = corpus.size() / nsess;

            for (int iter = 0; iter < 5; iter++) {
                double t0 = now_ms();
                std::vector<std::thread> th;
                std::vector<double> partial_ms(nsess, 0);

                for (int j = 0; j < nsess; j++) {
                    th.emplace_back([&, j]() {
                        double wt0 = now_ms();
                        int off = j * per;
                        int cnt = (j == nsess-1) ? (int)corpus.size() - off : per;
                        std::vector<const char*> ct;
                        for (int k = 0; k < cnt; k++) ct.push_back(corpus[off+k].c_str());
                        lembed_embeddings_t r = {0};
                        lembed_text_embedding_embed(emb[j], ct.data(), cnt, cnt, &r);
                        lembed_embeddings_free(&r);
                        double wt1 = now_ms();
                        partial_ms[j] = wt1 - wt0;
                    });
                }
                for (auto& t : th) t.join();
                double t1 = now_ms();

                double total_ms = t1 - t0;
                double total_tps = (double)corpus.size() / (total_ms / 1000.0);
                ms_samples.push_back(total_ms);
                tps_samples.push_back(total_tps);
            }

            double mean_tps = mean(tps_samples);
            double std_tps = stddev(tps_samples, mean_tps);
            double mean_ms = mean(ms_samples);

            printf("  %-16s %-8s %-12.1f %-12.1f %-12.1f\n",
                   "", configs[ci].label, mean_tps, std_tps, mean_ms);

            for (int i = 0; i < nsess; i++) lembed_text_embedding_free(emb[i]);
        }
        printf("\n");
    }

    printf("=== Termine ===\n");
    return 0;
}
