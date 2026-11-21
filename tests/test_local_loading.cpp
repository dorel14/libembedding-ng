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

    /* Test: a ".GGUF" path is routed to the GGUF backend like ".gguf".
     * A garbage file is refused by the GGUF loader, never by the ONNX path. */
    {
        const char* names[2] = {"lembed_garbage_case.GGUF", "lembed_garbage_case.gguf"};
        lembed_status_t st[2] = {LEMBED_OK, LEMBED_OK};
        for (int i = 0; i < 2; i++) {
            FILE* f = fopen(names[i], "wb");
            ASSERT(f != NULL, "garbage gguf file created");
            if (!f) continue;
            fwrite("not a gguf model", 1, 16, f);
            fclose(f);
            lembed_text_options_t opts = lembed_text_options_default();
            lembed_text_embedding_t* ctx = NULL;
            st[i] = lembed_text_embedding_create_from_path(names[i], &opts, &ctx);
            ASSERT(st[i] != LEMBED_OK, "garbage gguf is refused");
            ASSERT(ctx == NULL, "context is NULL for garbage gguf");
            remove(names[i]);
        }
        ASSERT(st[0] == st[1], ".GGUF routed like .gguf (same status)");
        ASSERT(st[0] == LEMBED_ERROR_LLAMA || st[0] == LEMBED_ERROR_GGUF,
               ".GGUF gets a GGUF-backend error");
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
