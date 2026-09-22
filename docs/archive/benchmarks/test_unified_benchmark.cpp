/*
 * Test Unified Backend Benchmark
 * Compares ONNX vs llama.cpp on same corpus
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>

int main(int argc, char** argv) {
    fprintf(stderr, "=== Unified Backend Benchmark ===\n\n");

    /* Get corpus */
    const char* const* texts = nullptr;
    int count = 0;
    lembed_benchmark_get_corpus(LEMBED_CORPUS_MIXED, &texts, &count);
    fprintf(stderr, "Corpus: %d texts (mixed lengths)\n\n", count);

    /* Test llama.cpp */
    const char* gguf = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";
    if (argc > 1) gguf = argv[1];

    fprintf(stderr, "--- llama.cpp ---\n");
    lembed_benchmark_result_t result = {0};
    lembed_status_t s = lembed_benchmark_autotune(gguf, "llama.cpp", LEMBED_OBJECTIVE_THROUGHPUT, &result);
    if (s == LEMBED_OK) {
        /* Extract model name from path */
        const char* name = strrchr(result.model_name, '\\');
        if (!name) name = strrchr(result.model_name, '/');
        if (name) name++; else name = result.model_name;
        fprintf(stderr, "  Model: %s\n", name);
        fprintf(stderr, "  Backend: %s\n", result.backend);
        fprintf(stderr, "  Config: %d sessions, %d threads\n", result.config.batch_size, result.config.num_threads);
        fprintf(stderr, "  Throughput: %.1f docs/s\n", result.metrics.throughput_docs_sec);
        fprintf(stderr, "  Latency p50: %.2f ms\n", result.metrics.latency_p50_ms);
        fprintf(stderr, "  Latency p95: %.2f ms\n", result.metrics.latency_p95_ms);
        fprintf(stderr, "  Memory: %.0f MB\n", result.metrics.peak_memory_mb);
        fprintf(stderr, "  Dim: %d\n", result.metrics.dim);
    } else {
        fprintf(stderr, "  FAILED: %s\n", lembed_last_error());
    }

    /* Test ONNX if available */
    const char* onnx = "C:\\Users\\david\\.cache\\libembedding\\models--Qdrant-all-MiniLM-L6-v2-onnx\\model.onnx";
    if (argc > 2) onnx = argv[2];

    fprintf(stderr, "\n--- ONNX ---\n");
    lembed_benchmark_result_t onnx_result = {0};
    s = lembed_benchmark_autotune(onnx, "onnx", LEMBED_OBJECTIVE_THROUGHPUT, &onnx_result);
    if (s == LEMBED_OK) {
        fprintf(stderr, "  Model: %s\n", onnx_result.model_name);
        fprintf(stderr, "  Backend: %s\n", onnx_result.backend);
        fprintf(stderr, "  Config: %d threads, batch=%d\n", onnx_result.config.num_threads, onnx_result.config.batch_size);
        fprintf(stderr, "  Throughput: %.1f docs/s\n", onnx_result.metrics.throughput_docs_sec);
        fprintf(stderr, "  Latency p50: %.2f ms\n", onnx_result.metrics.latency_p50_ms);
        fprintf(stderr, "  Latency p95: %.2f ms\n", onnx_result.metrics.latency_p95_ms);
        fprintf(stderr, "  Memory: %.0f MB\n", onnx_result.metrics.peak_memory_mb);
        fprintf(stderr, "  Dim: %d\n", onnx_result.metrics.dim);
    } else {
        fprintf(stderr, "  ONNX not available (expected if model not in cache)\n");
    }

    /* Comparison */
    if (result.metrics.throughput_docs_sec > 0 && onnx_result.metrics.throughput_docs_sec > 0) {
        fprintf(stderr, "\n=== Comparison ===\n");
        float ratio = onnx_result.metrics.throughput_docs_sec / result.metrics.throughput_docs_sec;
        fprintf(stderr, "ONNX / llama.cpp ratio: %.1fx\n", ratio);
        if (ratio > 1.0f)
            fprintf(stderr, "ONNX is %.1fx faster\n", ratio);
        else
            fprintf(stderr, "llama.cpp is %.1fx faster\n", 1.0f / ratio);
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
