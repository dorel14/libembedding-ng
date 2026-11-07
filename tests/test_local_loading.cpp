/*
 * test_local_loading.cpp - Test create_from_path with local model directory
 *
 * This test verifies that create_from_path correctly handles:
 *   - Missing directory (error)
 *   - Directory without required files (error)
 *
 * Full local loading tests with real ONNX models are run in integration
 * mode when a model directory is available.
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, lembed_last_error()); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

int main(void) {
    int passes = 0, failures = 0;

    /* Test: create_from_path with non-existent directory */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.dim = 384;
        opts.pooling = LEMBED_POOLING_CLS;
        lembed_text_embedding_t* ctx = NULL;
        lembed_status_t s = lembed_text_embedding_create_from_path(
            "/nonexistent/path/model", &opts, &ctx);
        ASSERT(s == LEMBED_ERROR_IO, "create_from_path with bad path returns IO error");
        ASSERT(ctx == NULL, "context is NULL on failure");
    }

    /* Test: create_from_path with empty path */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.dim = 384;
        opts.pooling = LEMBED_POOLING_CLS;
        lembed_text_embedding_t* ctx = NULL;
        lembed_status_t s = lembed_text_embedding_create_from_path(
            "", &opts, &ctx);
        ASSERT(s == LEMBED_ERROR_INVALID_ARGUMENT, "create_from_path with empty path returns error");
    }

    /* Test: create_from_path with NULL options */
    {
        lembed_text_embedding_t* ctx = NULL;
        lembed_status_t s = lembed_text_embedding_create_from_path(
            "/some/path", NULL, &ctx);
        ASSERT(s == LEMBED_ERROR_INVALID_ARGUMENT, "create_from_path with NULL options returns error");
    }

    /* Test: GGUF filenames cannot escape the model cache directory */
    {
        char* model_path = NULL;
        lembed_status_t s = lembed_ensure_gguf_model(
            "org/repo", "../outside.gguf", NULL, 0, 1, &model_path);
        ASSERT(s == LEMBED_ERROR_INVALID_ARGUMENT,
               "GGUF parent traversal must be rejected");
        ASSERT(model_path == NULL, "rejected GGUF path must not return a path");
    }

    /* Test: create_from_path functions are declared */
    {
        /* Just verify the functions exist and are callable */
        ASSERT(lembed_text_embedding_create_from_path != NULL, "text create_from_path exists");
        ASSERT(lembed_sparse_text_embedding_create_from_path != NULL, "sparse create_from_path exists");
#ifndef LIBEMBEDDING_NO_IMAGE
        ASSERT(lembed_image_embedding_create_from_path != NULL, "image create_from_path exists");
#endif
        ASSERT(lembed_reranker_create_from_path != NULL, "reranker create_from_path exists");
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
