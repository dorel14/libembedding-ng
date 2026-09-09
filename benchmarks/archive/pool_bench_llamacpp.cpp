/*
 * Session Pool benchmark for llama.cpp backend
 * Tests N independent sessions processing texts in parallel.
 * Each session has its own context (KV cache) sharing the same model weights.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <libembedding/llama_session_pool.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
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

int main(int argc, char** argv) {
    const char* model_path = R"(C:\Users\david\.cache\libembedding\models--BAAI-bge-small-en-v1.5\bge-small-en-v1.5-Q8_0.gguf)";
    if (argc > 1) model_path = argv[1];

    fprintf(stderr, "=== Session Pool Benchmark ===\n");
    fprintf(stderr, "Model: %s\n\n", model_path);

    /* Generate test texts */
    std::vector<std::string> texts;
    for (int i = 0; i < 128; i++) {
        texts.push_back("Document " + std::to_string(i) + " with unique content about topic " + std::to_string(i % 10));
    }
    int total_texts = (int)texts.size();
    fprintf(stderr, "Test corpus: %d texts\n\n", total_texts);

    /* Test different session counts */
    int session_counts[] = {1, 2, 3, 4, 6, 8};
    int ntests = sizeof(session_counts) / sizeof(session_counts[0]);

    fprintf(stderr, "%-10s %-15s %-15s %-10s\n", "Sessions", "Total (ms)", "Texts/sec", "Speedup");
    fprintf(stderr, "---------------------------------------------------\n");

    double baseline_tps = 0;

    for (int ti = 0; ti < ntests; ti++) {
        int nsess = session_counts[ti];

        /* Create pool */
        lembed::detail::LlamaSessionPool pool;
        try {
            pool.load_from_file(model_path, nsess, 1, 0, 0, false);
        } catch (const std::exception& e) {
            fprintf(stderr, "Failed to create pool with %d sessions: %s\n", nsess, e.what());
            continue;
        }

        /* Warmup */
        for (int i = 0; i < nsess; i++) {
            pool.embed(texts[i].c_str());
        }

        /* Benchmark: process all texts using thread pool */
        std::vector<double> times;
        int iters = 5;

        for (int iter = 0; iter < iters; iter++) {
            std::atomic<int> text_idx{0};
            double t0 = now_ms();

            /* Launch worker threads */
            std::vector<std::thread> workers;
            for (int w = 0; w < nsess; w++) {
                workers.emplace_back([&]() {
                    while (true) {
                        int idx = text_idx.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= total_texts) break;
                        pool.embed(texts[idx].c_str());
                    }
                });
            }
            for (auto& t : workers) t.join();

            double t1 = now_ms();
            times.push_back(t1 - t0);
        }

        double med_ms = median(times);
        double tps = (double)total_texts / (med_ms / 1000.0);
        if (ti == 0) baseline_tps = tps;
        double speedup = tps / baseline_tps;

        fprintf(stderr, "%-10d %-15.1f %-15.1f %-10.2fx\n", nsess, med_ms, tps, speedup);
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}



