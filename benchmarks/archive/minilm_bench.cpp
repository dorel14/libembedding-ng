/*
 * Test MiniLM-L6-v2 - modele plus leger
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

void embed_subbatch(lembed_text_embedding_t* embedder, const char** texts, int n, int bsz) {
    lembed_embeddings_t res = {0};
    lembed_text_embedding_embed(embedder, texts, n, bsz, &res);
    lembed_embeddings_free(&res);
}

int main() {
    printf("=== MiniLM-L6-v2 Benchmark ===\n\n");

    const char* model_path = "C:/Users/david/.cache/libembedding/models--Qdrant-all-MiniLM-L6-v2-onnx";

    int bsz = 64;
    const char* text = "The quick brown fox jumps over the lazy dog.";
    std::vector<const char*> texts(bsz, text);

    struct Config { int nsess; int nthreads; };
    Config configs[] = {
        {1, 4}, {1, 2},
        {2, 2}, {2, 1},
        {4, 1},
        {8, 1},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    printf("%-15s %-15s %-15s %-15s\n", "Sessions", "Threads/sess", "Total (ms)", "Textes/sec");
    printf("-----------------------------------------------------------\n");

    for (int ci = 0; ci < nconfigs; ci++) {
        int nsess = configs[ci].nsess;
        int nthreads = configs[ci].nthreads;

        lembed_text_options_t opts;
        memset(&opts, 0, sizeof(opts));
        opts.num_threads = nthreads;
        opts.pooling = LEMBED_POOLING_CLS;
        opts.dim = 384;

        std::vector<lembed_text_embedding_t*> emb(nsess);
        for (int i = 0; i < nsess; i++) {
            lembed_status_t s = lembed_text_embedding_create_from_path(model_path, &opts, &emb[i]);
            if (s != LEMBED_OK) {
                fprintf(stderr, "Erreur chargement: %s\n", lembed_last_error());
                return 1;
            }
        }

        int sub = bsz / nsess;

        /* Warmup */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < nsess; j++) {
                embed_subbatch(emb[j], texts.data() + j*sub, sub, sub);
            }
        }

        /* Benchmark */
        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            double t0 = now_ms();
            std::vector<std::thread> th;
            for (int j = 0; j < nsess; j++) {
                th.emplace_back(embed_subbatch, emb[j], texts.data() + j*sub, sub, sub);
            }
            for (auto& t : th) t.join();
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
