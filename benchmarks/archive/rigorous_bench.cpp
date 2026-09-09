/*
 * Rigorous production tests:
 * 1. Memory stability: create/embed/destroy cycles
 * 2. Batch consistency: cosine, L2, max abs diff
 * 3. Various batch sizes for same text
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

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

/* Embed a single text */
static void embed_one(lembed_text_embedding_t* emb, const char* text, float* out, int* dim) {
    lembed_embeddings_t res = {0};
    lembed_status_t s = lembed_text_embedding_embed(emb, &text, 1, 1, &res);
    if (s == LEMBED_OK && res.num_embeddings == 1) {
        *dim = res.dim;
        memcpy(out, res.data, res.dim * sizeof(float));
    }
    lembed_embeddings_free(&res);
}

/* Cosine similarity */
static double cosine_sim(const float* a, const float* b, int dim) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

/* L2 distance */
static double l2_dist(const float* a, const float* b, int dim) {
    double s = 0;
    for (int i = 0; i < dim; i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

/* Max absolute difference */
static double max_abs_diff(const float* a, const float* b, int dim) {
    double m = 0;
    for (int i = 0; i < dim; i++) {
        double d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ============ TEST 1: Memory stability with create/embed/destroy ============ */
static void test_memory_stability(lembed_text_model_t model, const char* model_name) {
    printf("=== Memory Stability: %s ===\n", model_name);
    printf("Create → embed → destroy cycles\n\n");

    const char* text = "The quick brown fox jumps over the lazy dog.";
    int cycles[] = {100, 500};

    double rss_start = get_rss_mb();
    printf("Starting RSS: %.1f MB\n\n", rss_start);

    for (int ci = 0; ci < 3; ci++) {
        int n = cycles[ci];
        double t0 = now_ms();

        for (int i = 0; i < n; i++) {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = model;
            opts.num_threads = 1;
            opts.offline = 1;
            opts.show_download_progress = 0;

            lembed_text_embedding_t* emb = nullptr;
            if (lembed_text_embedding_create(&opts, &emb) != LEMBED_OK) {
                printf("  FAIL at cycle %d: %s\n", i, lembed_last_error());
                return;
            }

            lembed_embeddings_t res = {0};
            lembed_text_embedding_embed(emb, &text, 1, 1, &res);
            lembed_embeddings_free(&res);
            lembed_text_embedding_free(emb);
        }

        double t1 = now_ms();
        double rss_now = get_rss_mb();

        printf("  %d cycles: %.0f ms total, RSS = %.1f MB (delta: +%.1f MB)\n",
               n, t1-t0, rss_now, rss_now - rss_start);
    }
    printf("\n");
}

/* ============ TEST 2: Batch consistency with metrics ============ */
static void test_batch_consistency(lembed_text_model_t model, const char* model_name) {
    printf("=== Batch Consistency: %s ===\n", model_name);

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = model;
    opts.num_threads = 1;  /* single thread for determinism */
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    if (lembed_text_embedding_create(&opts, &emb) != LEMBED_OK) {
        printf("  SKIP: %s\n\n", lembed_last_error());
        return;
    }

    const char* target = "Hello world";
    const char* others[] = {
        "Machine learning is great",
        "Natural language processing",
        "The quick brown fox jumps",
        "Artificial intelligence transforms",
        "Data science and analytics",
        "Cloud computing infrastructure",
        "Software engineering practices",
        "Computer vision applications",
        "Deep neural networks",
        "Big data processing"
    };
    int nothers = sizeof(others) / sizeof(others[0]);

    /* Reference: embed target alone */
    float ref[384];
    int dim = 0;
    embed_one(emb, target, ref, &dim);

    printf("  Reference: embed(\"%s\") alone, dim=%d\n", target, dim);

    /* Test different batch sizes */
    int batch_sizes[] = {1, 2, 5};
    int nbatch_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    printf("\n  %-8s %-12s %-12s %-12s\n", "Batch", "Cosine", "L2 Dist", "Max Abs");
    printf("  -----------------------------------------------\n");

    for (int bi = 0; bi < nbatch_sizes; bi++) {
        int bsz = batch_sizes[bi];
        if (bsz == 1) {
            /* Same text alone - should be identical */
            float alone[384];
            int d2;
            embed_one(emb, target, alone, &d2);
            double cos = cosine_sim(ref, alone, dim);
            double l2 = l2_dist(ref, alone, dim);
            double mad = max_abs_diff(ref, alone, dim);
            printf("  %-8d %-12.8f %-12.8f %-12.8f\n", bsz, cos, l2, mad);
        } else {
            /* Batch with target + (bsz-1) other texts */
            int n = std::min(bsz, nothers + 1);
            std::vector<const char*> texts;
            texts.push_back(target);
            for (int i = 1; i < n; i++)
                texts.push_back(others[i-1]);

            lembed_embeddings_t res = {0};
            lembed_text_embedding_embed(emb, texts.data(), n, n, &res);

            if (res.num_embeddings >= 1) {
                float batch_first[384];
                memcpy(batch_first, res.data, res.dim * sizeof(float));

                double cos = cosine_sim(ref, batch_first, dim);
                double l2 = l2_dist(ref, batch_first, dim);
                double mad = max_abs_diff(ref, batch_first, dim);
                printf("  %-8d %-12.8f %-12.8f %-12.8f\n", bsz, cos, l2, mad);
            }
            lembed_embeddings_free(&res);
        }
    }

    lembed_text_embedding_free(emb);
    printf("\n");
}

/* ============ TEST 3: Same text in different batch positions ============ */
static void test_position_consistency(lembed_text_model_t model, const char* model_name) {
    printf("=== Position Consistency: %s ===\n", model_name);

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = model;
    opts.num_threads = 1;
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    if (lembed_text_embedding_create(&opts, &emb) != LEMBED_OK) {
        printf("  SKIP: %s\n\n", lembed_last_error());
        return;
    }

    const char* target = "Hello world";
    const char* filler = "Filler text for batch padding and consistency testing purposes.";

    /* Embed target at different positions in a batch of 10 */
    printf("  Target at different positions in batch of 10:\n");
    printf("  %-10s %-12s %-12s %-12s\n", "Position", "Cosine", "L2 Dist", "Max Abs");
    printf("  -----------------------------------------------\n");

    /* Reference */
    float ref[384];
    int dim = 0;
    embed_one(emb, target, ref, &dim);

    for (int pos = 0; pos < 5; pos++) {
        std::vector<const char*> texts;
        for (int i = 0; i < 10; i++) {
            if (i == pos) texts.push_back(target);
            else texts.push_back(filler);
        }

        lembed_embeddings_t res = {0};
        lembed_text_embedding_embed(emb, texts.data(), 10, 10, &res);

        if (res.num_embeddings == 10) {
            float at_pos[384];
            memcpy(at_pos, res.data + pos * res.dim, res.dim * sizeof(float));

            double cos = cosine_sim(ref, at_pos, dim);
            double l2 = l2_dist(ref, at_pos, dim);
            double mad = max_abs_diff(ref, at_pos, dim);
            printf("  %-10d %-12.8f %-12.8f %-12.8f\n", pos, cos, l2, mad);
        }
        lembed_embeddings_free(&res);
    }

    lembed_text_embedding_free(emb);
    printf("\n");
}

int main() {
    printf("###########################################################\n");
    printf("# BATCH CONSISTENCY TEST                                   #\n");
    printf("###########################################################\n\n");

    struct Test {
        const char* name;
        lembed_text_model_t model;
    };

    Test tests[] = {
        {"MiniLM-L6-v2-Q (INT8 Dynamic)", LEMBED_TEXT_ALL_MINILM_L6_V2_Q},
        {"MiniLM-L6-v2 (FP32)", LEMBED_TEXT_ALL_MINILM_L6_V2},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    /* Only batch consistency test */
    for (int i = 0; i < ntests; i++) {
        test_batch_consistency(tests[i].model, tests[i].name);
    }

    printf("###########################################################\n");
    printf("# END                                                      #\n");
    printf("###########################################################\n");
    return 0;
}
