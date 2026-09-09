/*
 * Test : traitement parallele par sub-batches
 * Divise un batch en sous-batchs traites par plusieurs threads
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

/* Structure pour le resultat d'un sub-batch */
struct SubResult {
    std::vector<float> data;
    int num_embeddings;
    int dim;
};

/* Embed un sub-batch dans un thread separat */
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
    printf("=== Test paralleisme par sub-batches ===\n\n");

    int bsz = 64;
    const char* text = "The quick brown fox jumps over the lazy dog.";
    std::vector<const char*> texts(bsz, text);

    printf("Batch total: %d textes\n", bsz);

    /* Reference : 1 thread */
    printf("\n--- 1 session, 1 thread ---\n");
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.num_threads = 4;
        opts.offline = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* embedder = nullptr;
        lembed_text_embedding_create(&opts, &embedder);

        /* warmup */
        for (int i = 0; i < 3; i++) {
            lembed_embeddings_t w = {0};
            lembed_text_embedding_embed(embedder, texts.data(), bsz, bsz, &w);
            lembed_embeddings_free(&w);
        }

        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), bsz, bsz, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }
        printf("%.2f ms total = %.1f textes/sec\n", median(times), (double)bsz / (median(times)/1000.0));

        lembed_text_embedding_free(embedder);
    }

    /* 2 sessions en parallele */
    printf("\n--- 2 sessions paralleles ---\n");
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.num_threads = 2;
        opts.offline = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* emb1 = nullptr;
        lembed_text_embedding_t* emb2 = nullptr;
        lembed_text_embedding_create(&opts, &emb1);
        lembed_text_embedding_create(&opts, &emb2);

        int half = bsz / 2;

        /* warmup */
        for (int i = 0; i < 3; i++) {
            lembed_embeddings_t w1 = {0}, w2 = {0};
            lembed_text_embedding_embed(emb1, texts.data(), half, half, &w1);
            lembed_text_embedding_embed(emb2, texts.data() + half, half, half, &w2);
            lembed_embeddings_free(&w1);
            lembed_embeddings_free(&w2);
        }

        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            double t0 = now_ms();

            SubResult r1, r2;
            std::thread th1(embed_subbatch, emb1, texts.data(), half, half, std::ref(r1));
            std::thread th2(embed_subbatch, emb2, texts.data() + half, half, half, std::ref(r2));
            th1.join();
            th2.join();

            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.2f ms total = %.1f textes/sec\n", median(times), (double)bsz / (median(times)/1000.0));

        lembed_text_embedding_free(emb1);
        lembed_text_embedding_free(emb2);
    }

    /* 4 sessions en parallele */
    printf("\n--- 4 sessions paralleles ---\n");
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.num_threads = 1;
        opts.offline = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* emb[4];
        for (int i = 0; i < 4; i++) {
            lembed_text_embedding_create(&opts, &emb[i]);
        }

        int quarter = bsz / 4;

        /* warmup */
        for (int i = 0; i < 3; i++) {
            lembed_embeddings_t w[4];
            for (int j = 0; j < 4; j++) {
                w[j] = {0};
                lembed_text_embedding_embed(emb[j], texts.data() + j*quarter, quarter, quarter, &w[j]);
            }
            for (int j = 0; j < 4; j++) lembed_embeddings_free(&w[j]);
        }

        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            double t0 = now_ms();

            SubResult r[4];
            std::thread th[4];
            for (int j = 0; j < 4; j++) {
                th[j] = std::thread(embed_subbatch, emb[j], texts.data() + j*quarter, quarter, quarter, std::ref(r[j]));
            }
            for (int j = 0; j < 4; j++) th[j].join();

            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.2f ms total = %.1f textes/sec\n", median(times), (double)bsz / (median(times)/1000.0));

        for (int i = 0; i < 4; i++) lembed_text_embedding_free(emb[i]);
    }

    /* 8 sessions en parallele */
    printf("\n--- 8 sessions paralleles ---\n");
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.num_threads = 1;
        opts.offline = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* emb[8];
        for (int i = 0; i < 8; i++) {
            lembed_text_embedding_create(&opts, &emb[i]);
        }

        int sub = bsz / 8; /* 8 textes par session */

        /* warmup */
        for (int i = 0; i < 3; i++) {
            lembed_embeddings_t w[8];
            for (int j = 0; j < 8; j++) {
                w[j] = {0};
                lembed_text_embedding_embed(emb[j], texts.data() + j*sub, sub, sub, &w[j]);
            }
            for (int j = 0; j < 8; j++) lembed_embeddings_free(&w[j]);
        }

        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            double t0 = now_ms();

            SubResult r[8];
            std::thread th[8];
            for (int j = 0; j < 8; j++) {
                th[j] = std::thread(embed_subbatch, emb[j], texts.data() + j*sub, sub, sub, std::ref(r[j]));
            }
            for (int j = 0; j < 8; j++) th[j].join();

            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.2f ms total = %.1f textes/sec\n", median(times), (double)bsz / (median(times)/1000.0));

        for (int i = 0; i < 8; i++) lembed_text_embedding_free(emb[i]);
    }

    /* 16 sessions en parallele */
    printf("\n--- 16 sessions paralleles ---\n");
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.num_threads = 1;
        opts.offline = 1;
        opts.show_download_progress = 0;

        const int NSESS = 16;
        std::vector<lembed_text_embedding_t*> emb(NSESS);
        for (int i = 0; i < NSESS; i++) {
            lembed_text_embedding_create(&opts, &emb[i]);
        }

        int sub = bsz / NSESS; /* 4 textes par session */

        /* warmup */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < NSESS; j++) {
                lembed_embeddings_t w = {0};
                lembed_text_embedding_embed(emb[j], texts.data() + j*sub, sub, sub, &w);
                lembed_embeddings_free(&w);
            }
        }

        std::vector<double> times;
        for (int i = 0; i < 10; i++) {
            double t0 = now_ms();

            std::vector<SubResult> r(NSESS);
            std::vector<std::thread> th(NSESS);
            for (int j = 0; j < NSESS; j++) {
                th[j] = std::thread(embed_subbatch, emb[j], texts.data() + j*sub, sub, sub, std::ref(r[j]));
            }
            for (int j = 0; j < NSESS; j++) th[j].join();

            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.2f ms total = %.1f textes/sec\n", median(times), (double)bsz / (median(times)/1000.0));

        for (int i = 0; i < NSESS; i++) lembed_text_embedding_free(emb[i]);
    }

    printf("\n=== Termine ===\n");
    return 0;
}
