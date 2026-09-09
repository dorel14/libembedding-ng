/*
 * Benchmark sweep: all models × all corpus types
 * Outputs a comparison table for the unified benchmark
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

int main() {
    fprintf(stderr, "=== Unified Benchmark Sweep ===\n\n");

    /* Models to test (GGUF) */
    struct { const char* name; const char* path; } models[] = {
        {"MiniLM-L6-Q4", "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf"},
        {"Snowflake-XS-Q4", "C:\\Users\\david\\.cache\\libembedding\\gguf\\snowflake-xs-Q4_K_M.gguf"},
        {"Snowflake-S-Q4", "C:\\Users\\david\\.cache\\libembedding\\gguf\\snowflake-s-Q4_K_M.gguf"},
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
    fprintf(stderr, "%-18s %-8s %-10s %-10s %-10s %-8s %-8s\n",
            "Model", "Corpus", "Docs/s", "P50(ms)", "P95(ms)", "RAM(MB)", "Dim");
    fprintf(stderr, "--------------------------------------------------------------------------------\n");

    for (int mi = 0; mi < nmodels; mi++) {
        for (int ci = 0; ci < ncorpora; ci++) {
            /* Get corpus info */
            const char* const* texts = nullptr;
            int count = 0;
            lembed_benchmark_get_corpus(corpora[ci].type, &texts, &count);

            /* Auto-tune llama.cpp */
            lembed_backend_config_t config = {"llama.cpp", 1, 4, 0};
            lembed_benchmark_result_t result = {0};
            lembed_status_t s = lembed_benchmark_run(
                models[mi].path, "llama.cpp", corpora[ci].type, &config, &result);

            if (s == LEMBED_OK) {
                fprintf(stderr, "%-18s %-8s %-10.1f %-10.2f %-10.2f %-8.0f %-8d\n",
                        models[mi].name, corpora[ci].name,
                        result.metrics.throughput_docs_sec,
                        result.metrics.latency_p50_ms,
                        result.metrics.latency_p95_ms,
                        result.metrics.peak_memory_mb,
                        result.metrics.dim);
            } else {
                fprintf(stderr, "%-18s %-8s FAILED\n", models[mi].name, corpora[ci].name);
            }
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "=== Done ===\n");
    return 0;
}
