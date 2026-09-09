/*
 * Test the refactored embedding benchmark autotuner
 * with constraints + configurable weights
 */

#include <libembedding/libembedding.h>
#include <cstdio>

int main(int argc, char** argv) {
    const char* modelDir = nullptr; /* Use default cache */
    if (argc > 1) modelDir = argv[1];

    fprintf(stderr, "=== Autotuner Test: Constraints + Configurable Weights ===\n\n");

    /* List models */
    char paths[20][512];
    int n = lembed_benchmark_list_models(modelDir, (char*)paths, 20);
    fprintf(stderr, "Found %d GGUF models\n\n", n);

    /* Test 1: BALANCED with no constraints */
    fprintf(stderr, "--- Test 1: BALANCED (no constraints) ---\n");
    {
        lembed_benchmark_result_t result = {0};
        lembed_status_t s = lembed_benchmark_select_model(modelDir, LEMBED_OBJECTIVE_BALANCED, NULL, NULL, &result);
        if (s == LEMBED_OK) {
            fprintf(stderr, "  Model: %s, Score: %.3f, TPS: %.1f, Quality: %.2f, Sessions: %d\n",
                    result.model_name, result.score, result.metrics.throughput_docs_sec,
                    result.quality_score, result.config.batch_size);
        } else {
            fprintf(stderr, "  FAILED: %s\n", lembed_last_error());
        }
    }

    /* Test 2: THROUGHPUT with constraints */
    fprintf(stderr, "\n--- Test 2: THROUGHPUT (throughput_min=100, memory_max=25MB) ---\n");
    {
        lembed_benchmark_constraints_t constraints = {0};
        constraints.throughput_min = 100.0f;
        constraints.memory_max_mb = 25.0f;
        lembed_benchmark_result_t result = {0};
        lembed_status_t s = lembed_benchmark_select_model(modelDir, LEMBED_OBJECTIVE_THROUGHPUT, &constraints, NULL, &result);
        if (s == LEMBED_OK) {
            fprintf(stderr, "  Model: %s, Score: %.3f, TPS: %.1f, Sessions: %d\n",
                    result.model_name, result.score, result.metrics.throughput_docs_sec, result.config.batch_size);
        } else {
            fprintf(stderr, "  FAILED: %s\n", lembed_last_error());
        }
    }

    /* Test 3: Custom weights (quality-focused) */
    fprintf(stderr, "\n--- Test 3: Custom weights (Q=0.8, T=0.1, C=0.1) ---\n");
    {
        lembed_benchmark_weights_t custom = lembed_benchmark_custom_weights(0.8f, 0.1f, 0.1f);
        fprintf(stderr, "  Weights: Q=%.2f, T=%.2f, C=%.2f\n",
                custom.quality_weight, custom.throughput_weight, custom.cost_weight);
        lembed_benchmark_result_t result = {0};
        lembed_status_t s = lembed_benchmark_select_model(modelDir, LEMBED_OBJECTIVE_BALANCED, NULL, &custom, &result);
        if (s == LEMBED_OK) {
            fprintf(stderr, "  Model: %s, Score: %.3f, Quality: %.2f\n",
                    result.model_name, result.score, result.quality_score);
        }
    }

    /* Test 4: Detect optimal sessions for a specific model */
    fprintf(stderr, "\n--- Test 4: Detect optimal sessions ---\n");
    {
        const char* path = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";
        int optimal = 0;
        float tps = 0;
        lembed_status_t s = lembed_benchmark_detect_sessions(path, 8, &optimal, &tps);
        if (s == LEMBED_OK) {
            fprintf(stderr, "  MiniLM-L6-Q4: optimal=%d sessions, throughput=%.1f docs/s\n", optimal, tps);
        } else {
            fprintf(stderr, "  FAILED: %s\n", lembed_last_error());
        }
    }

    /* Test 5: MEMORY constraint (max 25 MB) */
    fprintf(stderr, "\n--- Test 5: MEMORY (memory_max_mb=25) ---\n");
    {
        lembed_benchmark_constraints_t constraints = {0};
        constraints.memory_max_mb = 25.0f;
        lembed_benchmark_result_t result = {0};
        lembed_status_t s = lembed_benchmark_select_model(modelDir, LEMBED_OBJECTIVE_MEMORY, &constraints, NULL, &result);
        if (s == LEMBED_OK) {
            fprintf(stderr, "  Model: %s, Size: %.1f MB, Score: %.3f\n",
                    result.model_name, result.file_size_mb, result.score);
        } else {
            fprintf(stderr, "  FAILED: %s\n", lembed_last_error());
        }
    }

    /* Test 6: Default cache dir */
    fprintf(stderr, "\n--- Test 6: Default cache dir ---\n");
    {
        const char* cache = lembed_benchmark_default_cache_dir();
        fprintf(stderr, "  Cache: %s\n", cache);
        int n = lembed_benchmark_list_models(cache, (char*)paths, 20);
        fprintf(stderr, "  Models in cache: %d\n", n);
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
