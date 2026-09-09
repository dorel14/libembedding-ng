/*
 * Standalone GGUF benchmark using llama.cpp directly (new API).
 * No libembedding dependency - measures load time, single latency, throughput.
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "llama.h"

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static int tokenize(const llama_vocab* vocab, const std::string& text,
                    std::vector<llama_token>& tokens, bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           tokens.data(), (int32_t)tokens.size(),
                           add_special, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           tokens.data(), (int32_t)tokens.size(),
                           add_special, false);
    } else {
        tokens.resize(n);
    }
    return n;
}

struct ModelBench {
    const char* name;
    std::string path;
    int n_ctx;
};

int main(int argc, char** argv) {
    llama_log_set([](enum ggml_log_level, const char* text, void*) {
        if (strstr(text, "error") || strstr(text, "warn")) {
            fprintf(stderr, "[llama] %s", text);
        }
    }, nullptr);

    llama_backend_init();
    atexit(llama_backend_free);

    std::vector<ModelBench> models;
    if (argc > 1) {
        models.push_back({"custom", argv[1], 512});
    } else {
        const char* cache = "C:\\Users\\david\\.cache\\libembedding\\gguf";
        models = {
            {"MiniLM-L6-Q4",      std::string(cache) + "\\all-MiniLM-L6-v2-Q4_K_M.gguf", 512},
            {"MiniLM-L6-Q8",      std::string(cache) + "\\all-MiniLM-L6-v2-Q8_0.gguf",   512},
            {"E5-small-Q4",       std::string(cache) + "\\e5-small-v2-Q4_K_M.gguf",       512},
            {"GIST-small-Q4",     std::string(cache) + "\\gist-small-Q4_K_M.gguf",        512},
            {"Snowflake-XS-Q4",   std::string(cache) + "\\snowflake-xs-Q4_K_M.gguf",      512},
            {"Snowflake-S-Q4",    std::string(cache) + "\\snowflake-s-Q4_K_M.gguf",       512},
        };
    }

    std::vector<std::string> texts;
    for (int i = 0; i < 64; i++) {
        texts.push_back("Document " + std::to_string(i) + " discusses topic " + std::to_string(i % 8) + " with various details.");
    }

    fprintf(stderr, "=== GGUF Benchmark (llama.cpp direct) ===\n");
    fprintf(stderr, "CPU threads: 4 | ctx: 512 | texts: %zu\n\n", texts.size());
    fprintf(stderr, "%-22s %-6s %-10s %-12s %-12s %-10s\n",
            "Model", "Dim", "Load(ms)", "Single(ms)", "Texts/sec", "Status");
    fprintf(stderr, "------------------------------------------------------------------------\n");

    for (const auto& m : models) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;

        double t0 = now_ms();
        llama_model* model = llama_model_load_from_file(m.path.c_str(), mparams);
        double t1 = now_ms();
        if (!model) {
            fprintf(stderr, "%-22s %-6s %-10s %-12s %-12s %-10s\n",
                    m.name, "-", "-", "-", "-", "LOAD FAIL");
            continue;
        }

        const llama_vocab* vocab = llama_model_get_vocab(model);
        const int n_embd = llama_model_n_embd(model);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = m.n_ctx;
        cparams.n_threads = 4;
        cparams.n_threads_batch = 4;
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;

        llama_context* ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            llama_model_free(model);
            fprintf(stderr, "%-22s %-6s %-10s %-12s %-12s %-10s\n",
                    m.name, "-", "-", "-", "-", "CTX FAIL");
            continue;
        }

        std::vector<float> emb(n_embd);

        /* Warmup */
        {
            std::vector<llama_token> tokens(64);
            int n = tokenize(vocab, "Hello world.", tokens, true);
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; j++) batch.token[j] = tokens[j];
            batch.n_tokens = n;
            llama_decode(ctx, batch);
            llama_batch_free(batch);
        }

        /* Single-text latency */
        std::vector<double> latencies;
        for (int i = 0; i < std::min(20, (int)texts.size()); i++) {
            std::vector<llama_token> tokens(2048);
            int n = tokenize(vocab, texts[i], tokens, true);

            double s0 = now_ms();
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; j++) batch.token[j] = tokens[j];
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) == 0) {
                const float* e = llama_get_embeddings(ctx);
                std::memcpy(emb.data(), e, n_embd * sizeof(float));
            }
            llama_batch_free(batch);
            double s1 = now_ms();
            latencies.push_back(s1 - s0);
        }
        double med_lat = median(latencies);

        /* Throughput */
        int total = (int)texts.size();
        double t_th0 = now_ms();
        for (int i = 0; i < total; i++) {
            std::vector<llama_token> tokens(2048);
            int n = tokenize(vocab, texts[i], tokens, true);
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int j = 0; j < n; j++) batch.token[j] = tokens[j];
            batch.n_tokens = n;
            llama_decode(ctx, batch);
            llama_batch_free(batch);
        }
        double t_th1 = now_ms();
        double tps = total / ((t_th1 - t_th0) / 1000.0);

        fprintf(stderr, "%-22s %-6d %-10.0f %-12.2f %-12.1f %-10s\n",
                m.name, n_embd, t1 - t0, med_lat, tps, "OK");

        llama_free(ctx);
        llama_model_free(model);
    }

    return 0;
}
