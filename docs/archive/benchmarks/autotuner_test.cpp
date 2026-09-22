#include <libembedding/libembedding.h>
#include <cstdio>

int main() {
    printf("=== Autotuner Test ===\n\n");

    printf("Running QUICK autotune (5-15s)...\n");
    printf("  Model: Qdrant/all-MiniLM-L6-v2-onnx\n");

    /* Check if model exists in cache */
    lembed_model_info_t info;
    lembed_status_t info_status = lembed_get_text_model_info(
        LEMBED_TEXT_ALL_MINILM_L6_V2, &info);
    printf("  Model info status: %d\n", info_status);
    if (info_status == LEMBED_OK) {
        printf("  Model name: %s\n", info.model_name);
        printf("  Model code: %s\n", info.model_code);
        printf("  Model file: %s\n", info.model_file);
    }

    lembed_tuning_result_t result = {0};
    lembed_status_t s = lembed_autotune(
        "Qdrant/all-MiniLM-L6-v2-onnx",
        LEMBED_AUTOTUNE_QUICK,
        &result
    );

    if (s != LEMBED_OK) {
        printf("Error: %s\n", lembed_last_error());
        return 1;
    }

    printf("\nOptimal configuration:\n");
    printf("  workers:     %d\n", result.workers);
    printf("  threads:     %d\n", result.threads);
    printf("  batch_size:  %d\n", result.batch_size);
    double tp = result.throughput_docs_sec;
    double lat = result.latency_ms;
    double mem = result.memory_mb;
    printf("  throughput:  %f docs/s\n", tp);
    printf("  latency:     %f ms/text\n", lat);
    printf("  memory:      %f MB (est)\n", mem);

    printf("\nDone.\n");
    return 0;
}
