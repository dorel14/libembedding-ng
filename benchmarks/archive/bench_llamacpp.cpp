/*
 * libembedding llama.cpp backend benchmark harness
 * Measures model load time, single-text latency, and batch throughput.
 * Outputs JSON to stdout for comparison with ONNX backend.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static double peak_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
}
static double current_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
}
#else
static double peak_rss_mb() { return 0.0; }
static double current_rss_mb() { return 0.0; }
#endif

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clk::now().time_since_epoch())
        .count();
}

static double median(std::vector<double>& v) {
    size_t n = v.size();
    if (n == 0) return 0.0;
    std::sort(v.begin(), v.end());
    return (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2.0 : v[n / 2];
}

static double percentile95(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)std::ceil(0.95 * v.size()) - 1;
    return v[std::min(idx, v.size() - 1)];
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

static void json_samples(FILE* fp, const char* key,
                         std::vector<double>& samples) {
    double med = median(samples);
    double p95 = percentile95(samples);
    fprintf(fp, "    \"%s\": { \"median\": %.2f, \"p95\": %.2f }", key, med, p95);
}

int main(int argc, char** argv) {
    const char* corpus_path = "corpus.txt";
    const char* model_path = R"(C:\Users\david\.cache\libembedding\models--BAAI-bge-small-en-v1.5\bge-small-en-v1.5-Q8_0.gguf)";
    if (argc > 1) model_path = argv[1];
    if (argc > 2) corpus_path = argv[2];

    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Backend: llama.cpp (GGUF)\n");

    if (!lembed_llama_backend_available()) {
        fprintf(stderr, "ERROR: llama.cpp backend not compiled in.\n");
        return 1;
    }

    auto corpus = load_corpus(corpus_path);
    if (corpus.empty()) {
        fprintf(stderr, "Failed to load corpus from %s\n", corpus_path);
        return 1;
    }

    std::vector<const char*> texts;
    for (auto& s : corpus) texts.push_back(s.c_str());

    const int WARMUP = 1;
    const int ITERS = 10;
    const int BATCH_SIZES[] = {1, 8, 32, 128, 512};
    const int NUM_BATCH_SIZES = 5;

    /* ── Benchmark A: Model load time ─────────────────────────────── */
    fprintf(stderr, "Benchmarking model load time...\n");
    std::vector<double> load_times;
    for (int i = 0; i < WARMUP + ITERS; i++) {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.num_threads = 1;
        opts.show_download_progress = 0;

        lembed_text_embedding_t* embedder = nullptr;
        double t0 = now_ms();
        lembed_status_t s = lembed_text_embedding_create_from_gguf_path(
            model_path, &opts, &embedder);
        double t1 = now_ms();

        if (s != LEMBED_OK) {
            fprintf(stderr, "Model load failed: %s\n", lembed_last_error());
            return 1;
        }
        if (i >= WARMUP) load_times.push_back(t1 - t0);
        lembed_text_embedding_free(embedder);
    }

    double rss_after_load = 0.0;

    /* Create persistent embedder for remaining benchmarks */
    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* embedder = nullptr;
    lembed_text_embedding_create_from_gguf_path(model_path, &opts, &embedder);
    rss_after_load = current_rss_mb();

    fprintf(stderr, "Model loaded: %s (dim=%d)\n",
            lembed_text_embedding_model_name(embedder),
            lembed_text_embedding_dim(embedder));

    /* ── Benchmark B: Single text latency (1 thread) ─────────────── */
    fprintf(stderr, "Benchmarking single text latency...\n");
    std::vector<double> single_times;
    const char* single_text[] = {"The quick brown fox jumps over the lazy dog."};

    for (int i = 0; i < WARMUP + ITERS; i++) {
        lembed_embeddings_t result = {0};
        double t0 = now_ms();
        lembed_text_embedding_embed(embedder, single_text, 1, 1, &result);
        double t1 = now_ms();
        if (i >= WARMUP) single_times.push_back(t1 - t0);
        lembed_embeddings_free(&result);
    }

    /* ── Benchmark C: Batch throughput (auto threads) ────────────── */
    fprintf(stderr, "Benchmarking batch throughput...\n");
    lembed_text_embedding_free(embedder);
    embedder = nullptr;

    opts.num_threads = 0; /* auto */
    lembed_text_embedding_create_from_gguf_path(model_path, &opts, &embedder);

    std::vector<std::vector<double>> batch_throughput(NUM_BATCH_SIZES);

    for (int bi = 0; bi < NUM_BATCH_SIZES; bi++) {
        int bsz = BATCH_SIZES[bi];
        int n = std::min(bsz, (int)texts.size());
        for (int i = 0; i < WARMUP + ITERS; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), n, bsz, &result);
            double t1 = now_ms();
            double elapsed_sec = (t1 - t0) / 1000.0;
            if (i >= WARMUP && elapsed_sec > 0) {
                batch_throughput[bi].push_back((double)n / elapsed_sec);
            }
            lembed_embeddings_free(&result);
        }
    }

    double peak_rss = peak_rss_mb();

    lembed_text_embedding_free(embedder);

    /* ── Output JSON ─────────────────────────────────────────────── */
    FILE* fp = stdout;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"library\": \"libembedding-llamacpp\",\n");
    fprintf(fp, "  \"model\": \"bge-small-en-v1.5-q8_0.gguf\",\n");
    fprintf(fp, "  \"backend\": \"llama.cpp\",\n");
    fprintf(fp, "  \"platform\": \"Windows x64\",\n");
    fprintf(fp, "  \"benchmarks\": {\n");

    json_samples(fp, "model_load_ms", load_times);
    fprintf(fp, ",\n");

    json_samples(fp, "single_latency_ms", single_times);
    fprintf(fp, ",\n");

    fprintf(fp, "    \"batch_throughput\": {\n");
    for (int bi = 0; bi < NUM_BATCH_SIZES; bi++) {
        double med = median(batch_throughput[bi]);
        fprintf(fp, "      \"%d\": { \"median_texts_per_sec\": %.1f }%s\n",
                BATCH_SIZES[bi], med,
                (bi < NUM_BATCH_SIZES - 1) ? "," : "");
    }
    fprintf(fp, "    },\n");

    fprintf(fp, "    \"memory\": { \"after_load_rss_mb\": %.1f, \"peak_rss_mb\": %.1f }\n",
            rss_after_load, peak_rss);

    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    return 0;
}

