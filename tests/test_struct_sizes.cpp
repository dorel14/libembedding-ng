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
#include <cstdint>
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
    /* ptr + provider + device_id (16), 9 ints (36) = 52, padded to 56,
     * cache_dir ptr (8) + 2 ints (8) = 72 */
    ASSERT_SIZE(lembed_multivector_options_t, 72);
    /* 2 ptrs (16), 2 ints (8), int64 */
    ASSERT_SIZE(lembed_multivectors_t, 32);
    ASSERT_SIZE(lembed_mv_mode_t, 4);
    ASSERT_SIZE(lembed_cache_config_t, 24);
    ASSERT_SIZE(lembed_text_options_t, 88);
    ASSERT_SIZE(lembed_text_options_v2_t, 96);
    ASSERT_SIZE(lembed_tuning_result_t, 40);
    ASSERT_SIZE(lembed_sparse_tuning_result_t, 48);
    ASSERT_SIZE(lembed_sparse_tuning_result_v2_t, 64);
    ASSERT_SIZE(lembed_reranker_tuning_result_t, 48);
    ASSERT_SIZE(lembed_unified_tuning_result_t, 64);
    ASSERT_SIZE(lembed_autotune_options_t, 48);
    ASSERT_SIZE(lembed_cache_hardware_info_t, 460);
    ASSERT_SIZE(lembed_tune_config_result_t, 24);
    ASSERT_SIZE(lembed_tune_cache_entry_t, 1104);
    ASSERT_SIZE(lembed_quantization_measurement_t, 32);
    ASSERT_SIZE(lembed_quantization_choice_t, 336);
#else
    /* 32-bit ABI, only where double and 64-bit integers are 8-byte aligned in
     * structs: ARM EABI (armv7l, hard-float), MSVC x86. 4-byte pointers and
     * size_t. i386 System V aligns them to 4, so its structs differ
     * (e.g. lembed_stats_t is 20 there): that target is skipped explicitly.
     * Only the structs holding a pointer or size_t differ from the 64-bit
     * table; the others are identical because they hold no pointer and their
     * doubles/uint64_t already sit on 8-byte boundaries.
     * python/tests/test_cdefs.py asserts the same numbers. */
    if (sizeof(void*) == 4 && alignof(double) == 8 && alignof(uint64_t) == 8) {
        /* 3x 8-byte fields */
        ASSERT_SIZE(lembed_stats_t, 24);
        /* 24 + 2x uint64 (16) + int (4) = 44, padded to 8 */
        ASSERT_SIZE(lembed_stats_v2_t, 48);
        /* const char* (4) + 7 ints/enums (28) */
        ASSERT_SIZE(lembed_model_desc_t, 32);
        /* 32 + 2 ints */
        ASSERT_SIZE(lembed_model_desc_v2_t, 40);
        /* ptr + 2 ints */
        ASSERT_SIZE(lembed_embeddings_t, 12);
        /* 2 ptrs + int */
        ASSERT_SIZE(lembed_sparse_embedding_t, 12);
        ASSERT_SIZE(lembed_rerank_result_t, 8);
        /* ptr + provider + device_id (12), 9 ints (36) = 48, no padding,
         * cache_dir ptr (4) + 2 ints (8) = 60 */
        ASSERT_SIZE(lembed_multivector_options_t, 60);
        /* 2 ptrs (8), 2 ints (8) = 16, int64 @16 */
        ASSERT_SIZE(lembed_multivectors_t, 24);
        ASSERT_SIZE(lembed_mv_mode_t, 4);
        /* size_t, size_t, int */
        ASSERT_SIZE(lembed_cache_config_t, 12);
        /* model, provider, device_id (12) + cache_dir ptr (16) + 15 ints/enums
         * (60) = 76; no padding, max alignment is 4 */
        ASSERT_SIZE(lembed_text_options_t, 76);
        ASSERT_SIZE(lembed_text_options_v2_t, 80);
        /* 3 ints (12) padded to 16, + 3 doubles */
        ASSERT_SIZE(lembed_tuning_result_t, 40);
        /* int, float, 3 ints (20) padded to 24, + 3 doubles */
        ASSERT_SIZE(lembed_sparse_tuning_result_t, 48);
        /* v1 (48) + 3 ints = 60, padded to 64 */
        ASSERT_SIZE(lembed_sparse_tuning_result_v2_t, 64);
        /* 3 ints padded to 16, + 4 doubles */
        ASSERT_SIZE(lembed_reranker_tuning_result_t, 48);
        /* 8 ints/floats (32) + 4 doubles */
        ASSERT_SIZE(lembed_unified_tuning_result_t, 64);
        /* 3 enums (12), texts ptr @12, n_texts @16, padded to 24, double @24,
         * min_tokens @32, padded to 40 */
        ASSERT_SIZE(lembed_autotune_options_t, 40);
        ASSERT(offsetof(lembed_autotune_options_t, texts) == 12, "texts offset");
        ASSERT(offsetof(lembed_autotune_options_t, max_latency_ms) == 24,
               "max_latency_ms offset");
        /* Only char arrays and 4-byte scalars below: same as 64-bit */
        ASSERT_SIZE(lembed_cache_hardware_info_t, 460);
        ASSERT_SIZE(lembed_tune_config_result_t, 24);
        ASSERT_SIZE(lembed_tune_cache_entry_t, 1104);
        /* int @0, 3 doubles @8/16/24 */
        ASSERT_SIZE(lembed_quantization_measurement_t, 32);
        /* 4 ints + reason[192] = 208, + 4 x 32 */
        ASSERT_SIZE(lembed_quantization_choice_t, 336);

        /* Field offsets that move with the 4-byte pointer / size_t. A size can
         * match by luck while a field shifts, which corrupts cffi reads. */
        ASSERT(offsetof(lembed_model_desc_t, dimension) == 4, "desc.dimension offset");
        ASSERT(offsetof(lembed_model_desc_t, device_id) == 28, "desc.device_id offset");
        ASSERT(offsetof(lembed_model_desc_v2_t, cache_size) == 36,
               "desc_v2.cache_size offset");
        ASSERT(offsetof(lembed_embeddings_t, num_embeddings) == 4,
               "embeddings.num_embeddings offset");
        ASSERT(offsetof(lembed_embeddings_t, dim) == 8, "embeddings.dim offset");
        ASSERT(offsetof(lembed_sparse_embedding_t, values) == 4,
               "sparse.values offset");
        ASSERT(offsetof(lembed_sparse_embedding_t, length) == 8, "sparse.length offset");
        ASSERT(offsetof(lembed_multivector_options_t, num_threads) == 12,
               "multivector_options.num_threads offset");
        ASSERT(offsetof(lembed_multivector_options_t, attend_to_expansion) == 44,
               "multivector_options.attend_to_expansion offset");
        ASSERT(offsetof(lembed_multivector_options_t, cache_dir) == 48,
               "multivector_options.cache_dir offset");
        ASSERT(offsetof(lembed_multivectors_t, num_docs) == 8, "multivectors.num_docs offset");
        ASSERT(offsetof(lembed_multivectors_t, total_vectors) == 16,
               "multivectors.total_vectors offset");
        ASSERT(offsetof(lembed_cache_config_t, current_size) == 4,
               "cache_config.current_size offset");
        ASSERT(offsetof(lembed_cache_config_t, ttl_seconds) == 8,
               "cache_config.ttl_seconds offset");
        ASSERT(offsetof(lembed_text_options_t, cache_dir) == 12,
               "text_options.cache_dir offset");
        ASSERT(offsetof(lembed_text_options_t, max_length) == 16,
               "text_options.max_length offset");
        ASSERT(offsetof(lembed_text_options_t, batch_strategy) == 72,
               "text_options.batch_strategy offset");
        ASSERT(offsetof(lembed_text_options_v2_t, quantization) == 76,
               "text_options_v2.quantization offset");
        ASSERT(offsetof(lembed_stats_v2_t, cache_hits) == 24, "stats_v2.cache_hits offset");
        ASSERT(offsetof(lembed_quantization_measurement_t, docs_per_sec) == 8,
               "measurement.docs_per_sec offset");
    } else {
        printf("32-bit ABI with 4-byte double alignment (i386): sizes not frozen\n");
    }
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
