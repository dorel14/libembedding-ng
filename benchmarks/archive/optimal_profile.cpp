/*
 * Recherche configuration optimale sessions × threads
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
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

struct SubResult {
    std::vector<float> data;
    int num_embeddings = 0;
    int dim = 0;
};

void embed_subbatch(lembed_text_embedding_t* embedder,
                    const char** texts, int n, int bsz,
                    SubResult& result) {
    lembed_embeddings_t res = {0};
    lembed_status_t s = lembed_text_embedding_embed(embedder, texts, n, bsz, &res);
    if (s == LEMBED_OK) {
        result.dim = res.dim;
        result.num_embeddings = res.num_embeddings;
        result.data.assign(res.data, res.data + (size_t)res.num_embeddings * res.dim);
        lembed_embeddings_free(&res);
    }
}

int main() {
    printf("=== Optimisation sessions x threads ===\n\n");

    int bsz = 64;
    const char* text = "The quick brown fox jumps over the lazy dog.";
    std::vector<const char*> texts(bsz, text);

    /* Toutes les combinaisons sessions × threads */
    int configs[][2] = {
        {1, 1}, {1, 2}, {1, 4},
        {2, 1}, {2, 2},
        {4, 1}, {4, 2},
        {8, 1},
        {16, 1}
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    printf("%-15s %-15s %-15s %-15s\n", "Sessions", "Threads/sess", "Total (ms)", "Textes/sec");
    ----------------------------------------------------------------\n");

    for (int ci = 0; ci < nconfigs; ci++) {
        int nsess = configs[ci][0];
        int nthreads = configs[ci][1];

        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.num_threads = nthreads;
        opts.offline = 1;
        opts.show_download_progress = 0;

        std::vector<lembed_text_embedding_t*> emb(nsess);
        for (int i = 0; i < nsess; i++) {
            lembed_text_embedding_create(&opts, &emb[i]);
        }

        int sub = bsz / nsess;
        if (sub < 1) sub = 1;

        /* warmup */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < nsess; j++) {
                lembed_embeddings_t w = {0};
                int n = std::min(sub, bsz - j*sub);
                if (n > 0) lembed_text_embedding_embed(emb[j], texts.data() + j*sub, n, n, &w);
                lembed_embeddings_free(&w);
            }
        }

        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            double t0 = now_ms();

            std::vector<SubResult> r(nsess);
            std::vector<std::thread> th(nsess);
            for (int j = 0; j < nsess; j++) {
                int n = std::min(sub, bsz - j*sub);
                if (n > 0) {
                    th[j] = std::thread(embed_subbatch, emb[j], texts.data() + j*sub, n, n, std::ref(r[j]));
                }
            }
            for (int j = 0; j < nsess; j++) {
                if (th[j].joinable()) th[j].join();
            }

            double t1 = now_ms();
            times.push_back(t1 - t0);
        }

        double med = median(times);
        double tps = (double)bsz / (med / 1000.0);
        printf("%-15d %-15d %-15.2f %-15.1f\n", nsess, nthreads, med, tps);

        for (int i = 0; i < nsess; i++) lembed_text_embedding_free(emb[i]);
    }

    printf("\n=== Termine ===\n");
    return 0;
}
