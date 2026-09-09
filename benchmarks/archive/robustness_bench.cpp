/*
 * Edge cases et tests de robustesse
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
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
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
}
#else
static double get_rss_mb() { return 0.0; }
#endif

static void test_edge_cases(lembed_text_embedding_t* emb, const char* model_name) {
    printf("  Edge cases:\n");

    struct Case {
        const char* name;
        const char* text;
    };

    Case cases[] = {
        {"Empty string", ""},
        {"Single char", "a"},
        {"Whitespace only", "   "},
        {"Numbers", "12345 67890 111213"},
        {"Special chars", "!@#$%^&*()_+-=[]{}|;':\",./<>?"},
        {"Unicode emoji", "Hello 😀 🎉 🌍 world"},
        {"Cyrillic", "Привет мир как дела"},
        {"Arabic", "مرحبا بالعالم"},
        {"Chinese", "你好世界今天天气很好"},
        {"Japanese", "こんにちは世界お元気ですか"},
        {"Korean", "안녕하세요 세계"},
        {"Very long word", "supercalifragilisticexpialidociousantidisestablishmentarianismpneumonoultramicroscopicsilicovolcanoconiosis"},
        {"Newlines", "Line1\nLine2\nLine3\nLine4"},
        {"Tabs", "Col1\tCol2\tCol3\tCol4"},
    };

    int ncases = sizeof(cases) / sizeof(cases[0]);
    int passed = 0;

    for (int i = 0; i < ncases; i++) {
        lembed_embeddings_t res = {0};
        lembed_status_t s = lembed_text_embedding_embed(emb, &cases[i].text, 1, 1, &res);

        if (s == LEMBED_OK && res.num_embeddings == 1) {
            /* Check for NaN */
            int nan = 0;
            for (int j = 0; j < res.dim; j++) {
                if (res.data[j] != res.data[j]) { nan = 1; break; }
            }
            printf("    %-20s OK (dim=%d%s)\n", cases[i].name, res.dim, nan ? " NaN!" : "");
            if (!nan) passed++;
            lembed_embeddings_free(&res);
        } else {
            printf("    %-20s FAIL (status=%d, n=%d)\n", cases[i].name, s, res.num_embeddings);
        }
    }
    printf("    Result: %d/%d passed\n\n", passed, ncases);
}

static void test_memory_stability(const char* model_name, lembed_text_model_t model) {
    printf("  Memory stability test (100 iterations):\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = model;
    opts.num_threads = 2;
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    if (lembed_text_embedding_create(&opts, &emb) != LEMBED_OK) {
        printf("    SKIP\n\n");
        return;
    }

    const char* text = "The quick brown fox jumps over the lazy dog.";

    double rss_before = get_rss_mb();

    /* 100 iterations */
    for (int i = 0; i < 100; i++) {
        lembed_embeddings_t res = {0};
        lembed_text_embedding_embed(emb, &text, 1, 1, &res);
        lembed_embeddings_free(&res);
    }

    double rss_after = get_rss_mb();
    printf("    Before: %.1f MB\n", rss_before);
    printf("    After 100 iter: %.1f MB\n", rss_after);
    printf("    Delta: %.1f MB\n\n", rss_after - rss_before);

    lembed_text_embedding_free(emb);
}

static void test_batch_consistency(lembed_text_embedding_t* emb) {
    printf("  Batch vs individual consistency:\n");

    const char* texts[] = {
        "Hello world",
        "Machine learning is great",
        "Natural language processing",
        "The quick brown fox",
        "Artificial intelligence"
    };
    int n = 5;

    /* Individual embeddings */
    std::vector<std::vector<float>> individual;
    for (int i = 0; i < n; i++) {
        lembed_embeddings_t res = {0};
        lembed_text_embedding_embed(emb, &texts[i], 1, 1, &res);
        individual.emplace_back(res.data, res.data + res.dim);
        lembed_embeddings_free(&res);
    }

    /* Batch embedding */
    std::vector<const char*> ctexts;
    for (int i = 0; i < n; i++) ctexts.push_back(texts[i]);

    lembed_embeddings_t batch = {0};
    lembed_text_embedding_embed(emb, ctexts.data(), n, n, &batch);

    /* Compare */
    double max_diff = 0;
    for (int i = 0; i < n && i < batch.num_embeddings; i++) {
        double diff = 0;
        for (int j = 0; j < batch.dim; j++) {
            double d = individual[i][j] - batch.data[i * batch.dim + j];
            diff += d * d;
        }
        diff = sqrt(diff);
        if (diff > max_diff) max_diff = diff;
    }
    printf("    Max difference (individual vs batch): %.6f\n", max_diff);
    printf("    %s\n\n", max_diff < 0.001 ? "CONSISTENT" : "INCONSISTENT");

    lembed_embeddings_free(&batch);
}

int main() {
    printf("=== Robustness & Edge Cases ===\n\n");

    struct Test { const char* name; lembed_text_model_t model; };
    Test tests[] = {
        {"MiniLM-L6-v2-Q", LEMBED_TEXT_ALL_MINILM_L6_V2_Q},
        {"BGE-small-en  ", LEMBED_TEXT_BGE_SMALL_EN_V15},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    for (int ti = 0; ti < ntests; ti++) {
        printf("--- %s ---\n", tests[ti].name);

        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = tests[ti].model;
        opts.num_threads = 4;
        opts.offline = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* emb = nullptr;
        if (lembed_text_embedding_create(&opts, &emb) != LEMBED_OK) {
            printf("  SKIP (erreur chargement)\n\n");
            continue;
        }

        test_edge_cases(emb, tests[ti].name);
        test_batch_consistency(emb);
        test_memory_stability(tests[ti].name, tests[ti].model);

        lembed_text_embedding_free(emb);
    }

    printf("=== Termine ===\n");
    return 0;
}
