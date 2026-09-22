/*
 * Benchmark avec corpus.txt (392 phrases reelles)
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cmath>
#include <cstdio>
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

static std::vector<std::string> load_corpus(const char* path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

void embed_task(lembed_text_embedding_t* emb, const std::vector<std::string>& texts, int off, int cnt) {
    std::vector<const char*> ct;
    for (int i = off; i < off+cnt; i++) ct.push_back(texts[i].c_str());
    lembed_embeddings_t r = {0};
    lembed_text_embedding_embed(emb, ct.data(), cnt, cnt, &r);
    lembed_embeddings_free(&r);
}

int main(int argc, char** argv) {
    const char* corpus_path = argc > 1 ? argv[1] : "corpus.txt";
    auto corpus = load_corpus(corpus_path);
    if (corpus.empty()) { fprintf(stderr, "Corpus vide\n"); return 1; }

    printf("=== Benchmark avec corpus reel ===\n");
    printf("Corpus: %d phrases depuis %s\n\n", (int)corpus.size(), corpus_path);

    /* Stats corpus */
    double avg_len = 0;
    for (auto& s : corpus) avg_len += s.size();
    avg_len /= corpus.size();
    printf("Longueur moyenne: %.0f caracteres (~%.0f tokens)\n\n", avg_len, avg_len/5);

    struct Test { const char* name; lembed_text_model_t model; };
    Test tests[] = {
        {"MiniLM-L6-v2-Q", LEMBED_TEXT_ALL_MINILM_L6_V2_Q},
        {"BGE-small-en  ", LEMBED_TEXT_BGE_SMALL_EN_V15},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    printf("%-20s %-12s %-12s %-12s\n", "Modele", "1x4", "8x1", "Units");
    printf("----------------------------------------------------\n");

    for (int ti = 0; ti < ntests; ti++) {
        printf("%-20s", tests[ti].name);

        /* 1x4 */
        {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = tests[ti].model;
            opts.num_threads = 4;
            opts.offline = 1;
            opts.show_download_progress = 0;
            lembed_text_embedding_t* emb = nullptr;
            lembed_text_embedding_create(&opts, &emb);

            /* warmup */
            for (int i = 0; i < 3; i++) embed_task(emb, corpus, 0, 4);

            std::vector<double> tps;
            for (int iter = 0; iter < 5; iter++) {
                double t0 = now_ms();
                for (int off = 0; off < (int)corpus.size(); off += 64) {
                    int cnt = std::min(64, (int)corpus.size() - off);
                    embed_task(emb, corpus, off, cnt);
                }
                double t1 = now_ms();
                tps.push_back((double)corpus.size() / ((t1-t0)/1000.0));
            }
            printf(" %-12.0f", mean(tps));
            lembed_text_embedding_free(emb);
        }

        /* 8x1 */
        {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = tests[ti].model;
            opts.num_threads = 1;
            opts.offline = 1;
            opts.show_download_progress = 0;

            const int NSess = 8;
            std::vector<lembed_text_embedding_t*> emb(NSess);
            for (int i = 0; i < NSess; i++)
                lembed_text_embedding_create(&opts, &emb[i]);

            int per = corpus.size() / NSess;

            /* warmup */
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < NSess; j++)
                    embed_task(emb[j], corpus, j*per, per);

            std::vector<double> tps;
            for (int iter = 0; iter < 5; iter++) {
                double t0 = now_ms();
                std::vector<std::thread> th;
                for (int j = 0; j < NSess; j++) {
                    th.emplace_back([&, j]() { embed_task(emb[j], corpus, j*per, per); });
                }
                for (auto& t : th) t.join();
                double t1 = now_ms();
                tps.push_back((double)corpus.size() / ((t1-t0)/1000.0));
            }
            printf(" %-12.0f", mean(tps));

            for (int i = 0; i < NSess; i++) lembed_text_embedding_free(emb[i]);
        }

        printf(" docs/s\n");
    }

    printf("\n=== Termine ===\n");
    return 0;
}
