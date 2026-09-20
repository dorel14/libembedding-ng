/*
 * Global Benchmark: ONNX vs llama.cpp
 * Tests all available models on both backends with multiple corpus types
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main() {
    fprintf(stderr, "=== Global Benchmark: ONNX vs llama.cpp ===\n\n");

    /* Hardware info */
    lembed_cache_hardware_info_t hw = {0};
    lembed_cache_detect_hardware(&hw);
    fprintf(stderr, "Hardware: %s (%dP/%dT)\n\n", hw.cpu_name, hw.physical_cores, hw.logical_cores);

    /* Models to test: name | onnx_path | gguf_path */
    struct Model {
        const char* name;
        const char* onnx_path;  /* NULL if not available */
        const char* gguf_path;  /* NULL if not available */
    };

    Model models[] = {
        {"MiniLM-L6",
         "C:\\Users\\david\\.cache\\libembedding\\models--Qdrant-all-MiniLM-L6-v2-onnx",
         "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf"},
        {"BGE-small",
         "C:\\Users\\david\\.cache\\libembedding\\models--Xenova-bge-small-en-v1.5\\onnx",
         "C:\\Users\\david\\.cache\\libembedding\\models--BAAI-bge-small-en-v1.5\\bge-small-en-v1.5-Q8_0.gguf"},
        {"Snowflake-XS",
         NULL,
         "C:\\Users\\david\\.cache\\libembedding\\gguf\\snowflake-xs-Q4_K_M.gguf"},
    };
    int nmodels = 3;

    /* Corpus types */
    struct { lembed_corpus_type_t type; const char* name; } corpora[] = {
        {LEMBED_CORPUS_SHORT, "Short"},
        {LEMBED_CORPUS_MEDIUM, "Medium"},
        {LEMBED_CORPUS_LONG, "Long"},
        {LEMBED_CORPUS_MIXED, "Mixed"},
    };
    int ncorpora = 4;

    /* Print header */
    fprintf(stderr, "%-18s %-10s %-8s %8s %8s %8s %8s %8s\n",
            "Model", "Backend", "Corpus", "Docs/s", "P50(ms)", "P95(ms)", "RAM(MB)", "Dim");
    fprintf(stderr, "------------------------------------------------------------------------------------------\n");

    for (int mi = 0; mi < nmodels; mi++) {
        /* ONNX backend */
        if (models[mi].onnx_path) {
            lembed_backend_config_t cfg = {"onnx", 4, 64, 1};
            for (int ci = 0; ci < ncorpora; ci++) {
                lembed_benchmark_result_t result = {0};
                lembed_status_t s = lembed_benchmark_run(
                    models[mi].onnx_path, "onnx", corpora[ci].type, &cfg, &result);
                if (s == LEMBED_OK) {
                    fprintf(stderr, "%-18s %-10s %-8s %8.1f %8.2f %8.2f %8.0f %8d\n",
                            models[mi].name, "onnx", corpora[ci].name,
                            result.metrics.throughput_docs_sec,
                            result.metrics.latency_p50_ms,
                            result.metrics.latency_p95_ms,
                            result.metrics.peak_memory_mb,
                            result.metrics.dim);
                } else {
                    fprintf(stderr, "%-18s %-10s %-8s FAILED\n",
                            models[mi].name, "onnx", corpora[ci].name);
                }
            }
        }

        /* llama.cpp backend */
        if (models[mi].gguf_path) {
            lembed_backend_config_t cfg = {"llama.cpp", 1, 4, 0};
            for (int ci = 0; ci < ncorpora; ci++) {
                lembed_benchmark_result_t result = {0};
                lembed_status_t s = lembed_benchmark_run(
                    models[mi].gguf_path, "llama.cpp", corpora[ci].type, &cfg, &result);
                if (s == LEMBED_OK) {
                    fprintf(stderr, "%-18s %-10s %-8s %8.1f %8.2f %8.2f %8.0f %8d\n",
                            models[mi].name, "llama.cpp", corpora[ci].name,
                            result.metrics.throughput_docs_sec,
                            result.metrics.latency_p50_ms,
                            result.metrics.latency_p95_ms,
                            result.metrics.peak_memory_mb,
                            result.metrics.dim);
                } else {
                    fprintf(stderr, "%-18s %-10s %-8s FAILED\n",
                            models[mi].name, "llama.cpp", corpora[ci].name);
                }
            }
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "=== Done ===\n");
    return 0;
}
