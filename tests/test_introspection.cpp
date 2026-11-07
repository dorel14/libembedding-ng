/*
 * test_introspection.cpp - Integration test for model introspection
 * Requires network access for model download.
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

    lembed_text_options_t opts = lembed_text_options_default();
    opts.batch_size = 64;
    opts.num_threads = 2;

    lembed_text_embedding_t* ctx = NULL;
    lembed_status_t s = lembed_text_embedding_create(&opts, &ctx);
    ASSERT(s == LEMBED_OK, "create text embedding");
    if (s != LEMBED_OK) {
        printf("Cannot create embedder: %s\n", lembed_last_error());
        return 1;
    }

    /* Test: model name */
    {
        const char* name = lembed_text_embedding_model_name(ctx);
        ASSERT(name != NULL, "model name non-NULL");
        ASSERT(strstr(name, "bge-small") != NULL, "model name contains bge-small");
    }

    /* Test: max_length */
    {
        int ml = lembed_text_embedding_max_length(ctx);
        ASSERT(ml > 0, "max_length > 0");
        ASSERT(ml == 512, "max_length == 512");
    }

    /* Test: desc struct */
    {
        const lembed_model_desc_t* desc = lembed_text_embedding_desc(ctx);
        ASSERT(desc != NULL, "desc non-NULL");
        ASSERT(desc->dimension == 384, "desc dimension == 384");
        ASSERT(desc->max_length == 512, "desc max_length == 512");
        ASSERT(desc->batch_size == 64, "desc batch_size == 64");
        ASSERT(desc->num_threads == 2, "desc num_threads == 2");
        ASSERT(desc->provider == LEMBED_PROVIDER_CPU, "desc provider == CPU");
        ASSERT(desc->device_id == 0, "desc device_id == 0");
        ASSERT(desc->name != NULL, "desc name non-NULL");
    }

    lembed_text_embedding_free(ctx);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
