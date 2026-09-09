/*
 * Stress test memoire: cycles create/embed/destroy
 * Avec EmbeddingPool (8 workers)
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

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

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

void embed_task(lembed_text_embedding_t* emb, const char* text) {
    lembed_embeddings_t r = {0};
    lembed_text_embedding_embed(emb, &text, 1, 1, &r);
    lembed_embeddings_free(&r);
}

void embed_task_multi(lembed_text_embedding_t* emb, const std::vector<const char*>& texts) {
    lembed_embeddings_t r = {0};
    lembed_text_embedding_embed(emb, texts.data(), (int)texts.size(), (int)texts.size(), &r);
    lembed_embeddings_free(&r);
}

/* Test 1: Single session create/destroy cycles */
static void test_single_session_cycles(lembed_text_model_t model, const char* name) {
    printf("--- %s: Single Session Create/Destroy ---\n", name);

    const char* text = "Hello world";

    double rss_start = get_rss_mb();
    printf("  Starting RSS: %.1f MB\n", rss_start);

    int cycles[] = {50, 100};
    int ncycles = sizeof(cycles) / sizeof(cycles[0]);
    for (int ci = 0; ci < ncycles; ci++) {
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
                printf("    FAIL at cycle %d\n", i);
                return;
            }

            embed_task(emb, text);
            lembed_text_embedding_free(emb);
        }

        double t1 = now_ms();
        double rss_now = get_rss_mb();
        printf("    %5d cycles: %6.0f ms, RSS: %6.1f MB (delta: %+5.1f MB)\n",
               n, t1-t0, rss_now, rss_now - rss_start);
    }
    printf("\n");
}

/* Test 2: Multiple sessions create/destroy cycles */
static void test_multi_session_cycles(lembed_text_model_t model, const char* name) {
    printf("--- %s: 8 Sessions Create/Destroy ---\n", name);

    const char* texts[] = {"Text one", "Text two", "Text three", "Text four",
                           "Text five", "Text six", "Text seven", "Text eight"};
    int ntexts = 8;

    double rss_start = get_rss_mb();
    printf("  Starting RSS: %.1f MB\n", rss_start);

    int cycles[] = {20, 50};
    int ncycles = sizeof(cycles) / sizeof(cycles[0]);
    for (int ci = 0; ci < ncycles; ci++) {
        int n = cycles[ci];
        double t0 = now_ms();

        for (int i = 0; i < n; i++) {
            std::vector<lembed_text_embedding_t*> emb(ntexts);
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = model;
            opts.num_threads = 1;
            opts.offline = 1;
            opts.show_download_progress = 0;

            for (int j = 0; j < ntexts; j++) {
                if (lembed_text_embedding_create(&opts, &emb[j]) != LEMBED_OK) {
                    printf("    FAIL at cycle %d, session %d\n", i, j);
                    return;
                }
            }

            /* Embed in parallel */
            std::vector<std::thread> th;
            for (int j = 0; j < ntexts; j++)
                th.emplace_back(embed_task, emb[j], texts[j]);
            for (auto& t : th) t.join();

            for (int j = 0; j < ntexts; j++)
                lembed_text_embedding_free(emb[j]);
        }

        double t1 = now_ms();
        double rss_now = get_rss_mb();
        printf("    %5d cycles: %6.0f ms, RSS: %6.1f MB (delta: %+5.1f MB)\n",
               n, t1-t0, rss_now, rss_now - rss_start);
    }
    printf("\n");
}

int main() {
    printf("### MEMORY STRESS TEST ###\n\n");

    struct Test { const char* name; lembed_text_model_t model; };
    Test tests[] = {
        {"MiniLM-L6-v2-Q", LEMBED_TEXT_ALL_MINILM_L6_V2_Q},
    };

    /* Test: just create/destroy sessions */
    printf("--- Create/Destroy Only (no embed) ---\n");
    {
        double rss_start = get_rss_mb();
        printf("  Starting RSS: %.1f MB\n", rss_start);

        for (int i = 0; i < 100; i++) {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = tests[0].model;
            opts.num_threads = 1;
            opts.offline = 1;
            opts.show_download_progress = 0;

            lembed_text_embedding_t* emb = nullptr;
            lembed_text_embedding_create(&opts, &emb);
            lembed_text_embedding_free(emb);
        }

        double rss_now = get_rss_mb();
        printf("  After 100 create/destroy: RSS: %.1f MB (delta: +%.1f MB)\n\n",
               rss_now, rss_now - rss_start);
    }

    test_single_session_cycles(tests[0].model, tests[0].name);

    printf("### DONE ###\n");
    return 0;
}
