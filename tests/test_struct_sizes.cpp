/*
 * test_struct_sizes.cpp - Guard the ABI of the structs shared with the cffi
 * bindings (python/src/libembedding/_cdefs.h).
 *
 * A silent float/double or int/uint64_t mismatch between the C headers and the
 * cffi declarations corrupts memory without any compile-time error. This test
 * freezes the C sizes; python/tests/test_cdefs.py freezes the same values on
 * the cffi side. Any change to one of the two tables must be reflected in the
 * other.
 *
 * No model downloads required.
 */

#include <libembedding/libembedding.h>

#include <cstddef>
#include <cstdio>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while (0)

#define ASSERT_SIZE(type, expected) do { \
    if (sizeof(type) != (size_t)(expected)) { \
        fprintf(stderr, "FAIL: sizeof(%s) = %zu, expected %d (line %d)\n", \
                #type, sizeof(type), (int)(expected), __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while (0)

int main(void) {
    int passes = 0, failures = 0;

    /* The frozen table below is the 64-bit ABI (LP64 and Win64 agree on every
     * struct of this library: they are all flat, no unions, no bitfields).
     * python/tests/test_cdefs.py asserts the same numbers for the cffi side. */
#if defined(_WIN64) || defined(__LP64__)
    ASSERT_SIZE(lembed_stats_t, 24);
    ASSERT_SIZE(lembed_stats_v2_t, 48);
    ASSERT_SIZE(lembed_model_desc_t, 40);
    ASSERT_SIZE(lembed_model_desc_v2_t, 48);
    ASSERT_SIZE(lembed_embeddings_t, 16);
    ASSERT_SIZE(lembed_sparse_embedding_t, 24);
    ASSERT_SIZE(lembed_rerank_result_t, 8);
    ASSERT_SIZE(lembed_cache_config_t, 24);
    ASSERT_SIZE(lembed_text_options_t, 88);
    ASSERT_SIZE(lembed_text_options_v2_t, 96);
    ASSERT_SIZE(lembed_tuning_result_t, 40);
    ASSERT_SIZE(lembed_sparse_tuning_result_t, 48);
    ASSERT_SIZE(lembed_reranker_tuning_result_t, 48);
    ASSERT_SIZE(lembed_unified_tuning_result_t, 64);
    ASSERT_SIZE(lembed_cache_hardware_info_t, 460);
    ASSERT_SIZE(lembed_tune_config_result_t, 24);
    ASSERT_SIZE(lembed_tune_cache_entry_t, 1104);
#else
    printf("32-bit ABI: sizes not frozen, printing only\n");
#endif

    /* Invariants that must hold on every ABI, including 32-bit */
    ASSERT(sizeof(lembed_stats_v2_t) > sizeof(lembed_stats_t),
           "stats_v2 must be larger than stats");
    ASSERT(sizeof(lembed_model_desc_v2_t) > sizeof(lembed_model_desc_t),
           "model_desc_v2 must be larger than model_desc");
    ASSERT(sizeof(lembed_text_options_v2_t) > sizeof(lembed_text_options_t),
           "text_options_v2 must be larger than text_options");
    ASSERT(offsetof(lembed_stats_v2_t, cache_hits) == sizeof(lembed_stats_t),
           "cache_hits must follow the stats base without padding");
    ASSERT(offsetof(lembed_model_desc_v2_t, quantization) == sizeof(lembed_model_desc_t),
           "quantization must follow the desc base without padding");

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
