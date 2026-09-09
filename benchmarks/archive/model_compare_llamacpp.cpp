/*
 * Multi-model benchmark for llama.cpp backend
 * Tests multiple small GGUF models and compares throughput.
 * Mirrors the ONNX model_compare.cpp methodology.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <libembedding/llama_session_pool.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct ModelInfo {
    const char* name;
    const char* path;
    int expected_dim;
};

int main(int argc, char** argv) {
    const char* modelDir = R"(C:\Users\david\.cache\libembedding)";
    if (argc > 1) modelDir = argv[1];

    ModelInfo models[] = {
        {"MiniLM-L6-Q4",    "models--sentence-transformers-all-MiniLM-L6-v2\\all-MiniLM-L6-v2-Q4_K_M.gguf",     384},
        {"MiniLM-L6-Q8",    "models--sentence-transformers-all-MiniLM-L6-v2\\all-MiniLM-L6-v2-Q8_0.gguf",       384},
        {"Snowflake-XS-Q4", "models--Snowflake-snowflake-arctic-embed-xs\\snowflake-xs-Q4_K_M.gguf",        384},
        {"Snowflake-S-Q4",  "models--Snowflake-snowflake-arctic-embed-s\\snowflake-s-Q4_K_M.gguf",         384},
        {"E5-small-Q4",     "models--intfloat-e5-small-v2\\e5-small-v2-Q4_K_M.gguf",         384},
        {"GIST-small-Q4",   "models--avsolatorio-GIST-small-Embedding-v0\\gist-small-Q4_K_M.gguf",          384},
    };
    int nmodels = sizeof(models) / sizeof(models[0]);

    /* Generate test texts (unique, varied) */
    std::vector<std::string> texts;
    for (int i = 0; i < 64; i++) {
        texts.push_back("Document " + std::to_string(i) + " discusses topic number " + std::to_string(i % 8) + " with various details and context about the subject matter.");
    }
    int total_texts = (int)texts.size();

    fprintf(stderr, "=== Multi-Model Benchmark (llama.cpp) ===\n");
    fprintf(stderr, "Model dir: %s\n", modelDir);
    fprintf(stderr, "Test corpus: %d texts\n\n", total_texts);

    /* Results header */
    fprintf(stderr, "%-20s %-8s %-12s %-12s %-12s %-10s\n",
            "Model", "Dim", "Load(ms)", "Single(ms)", "Texts/sec", "RAM(MB)");
    fprintf(stderr, "------------------------------------------------------------------------\n");

    for (int mi = 0; mi < nmodels; mi++) {
        std::string path = std::string(modelDir) + "\\" + models[mi].path;

        /* Benchmark load time */
        double load_ms = 0;
        lembed_text_embedding_t* embedder = nullptr;
        {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.num_threads = 4;
            opts.show_download_progress = 0;

            double t0 = now_ms();
            lembed_status_t s = lembed_text_embedding_create_from_gguf_path(path.c_str(), &opts, &embedder);
            double t1 = now_ms();
            load_ms = t1 - t0;

            if (s != LEMBED_OK) {
                fprintf(stderr, "%-20s FAILED: %s\n", models[mi].name, lembed_last_error());
                continue;
            }
        }

        int dim = lembed_text_embedding_dim(embedder);

        /* Prepare text pointers */
        std::vector<const char*> text_ptrs;
        for (auto& t : texts) text_ptrs.push_back(t.c_str());

        /* Warmup */
        lembed_embeddings_t w = {0};
        lembed_text_embedding_embed(embedder, text_ptrs.data(), 4, 4, &w);
        lembed_embeddings_free(&w);

        /* Benchmark single text */
        double single_ms = 0;
        {
            std::vector<double> times;
            for (int i = 0; i < 20; i++) {
                lembed_embeddings_t r = {0};
                double t0 = now_ms();
                lembed_text_embedding_embed(embedder, text_ptrs.data(), 1, 1, &r);
                double t1 = now_ms();
                times.push_back(t1 - t0);
                lembed_embeddings_free(&r);
            }
            single_ms = median(times);
        }

        /* Benchmark batch throughput (single session) */
        double tps = 0;
        {
            std::vector<double> times;
            for (int iter = 0; iter < 5; iter++) {
                double t0 = now_ms();
                int off = 0;
                while (off < total_texts) {
                    int n = std::min(16, total_texts - off);
                    lembed_embeddings_t r = {0};
                    lembed_text_embedding_embed(embedder, text_ptrs.data() + off, n, n, &r);
                    lembed_embeddings_free(&r);
                    off += n;
                }
                double t1 = now_ms();
                times.push_back(t1 - t0);
            }
            double med_ms = median(times);
            tps = (double)total_texts / (med_ms / 1000.0);
        }

        fprintf(stderr, "%-20s %-8d %-12.1f %-12.2f %-12.1f\n",
                models[mi].name, dim, load_ms, single_ms, tps);

        lembed_text_embedding_free(embedder);
    }

    /* Now benchmark with session pool for all models */
    fprintf(stderr, "\n=== Session Pool Scaling ===\n");
    fprintf(stderr, "%-20s %-12s %-12s %-12s\n", "Model", "1-sess(tps)", "4-sess(tps)", "Speedup");
    fprintf(stderr, "----------------------------------------------------\n");

    for (int mi = 0; mi < nmodels; mi++) {
        std::string path = std::string(modelDir) + "\\" + models[mi].path;

        auto run_pool = [&](int nsess, int nthreads) -> double {
            lembed::detail::LlamaSessionPool pool;
            try {
                pool.load_from_file(path.c_str(), nsess, nthreads, 0, 0, false);
            } catch (...) { return 0; }

            /* warmup */
            for (int i = 0; i < nsess && i < (int)texts.size(); i++)
                pool.embed(texts[i].c_str());

            std::atomic<int> idx{0};
            double t0 = now_ms();
            std::vector<std::thread> workers;
            for (int w = 0; w < nsess; w++) {
                workers.emplace_back([&]() {
                    while (true) {
                        int i = idx.fetch_add(1, std::memory_order_relaxed);
                        if (i >= total_texts) break;
                        pool.embed(texts[i].c_str());
                    }
                });
            }
            for (auto& t : workers) t.join();
            double t1 = now_ms();
            return (double)total_texts / ((t1 - t0) / 1000.0);
        };

        double tps1 = run_pool(1, 4);
        double tps4 = run_pool(4, 1);
        double speedup = (tps1 > 0) ? tps4 / tps1 : 0;

        fprintf(stderr, "%-20s %-12.1f %-12.1f %-12.2fx\n", models[mi].name, tps1, tps4, speedup);
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}




