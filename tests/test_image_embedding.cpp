/*
 * test_image_embedding.c - Integration test for image embeddings
 * Requires network access for model download and a test image.
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, lembed_last_error()); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

/* Generate a simple test PPM image file */
static int create_test_image(const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    int w = 64, h = 64;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char r = (unsigned char)(x * 4);
            unsigned char g = (unsigned char)(y * 4);
            unsigned char b = (unsigned char)((x + y) * 2);
            fwrite(&r, 1, 1, f);
            fwrite(&g, 1, 1, f);
            fwrite(&b, 1, 1, f);
        }
    }
    fclose(f);
    return 1;
}

int main(void) {
    int passes = 0, failures = 0;

    /* Create a test image */
    const char* test_img = "/tmp/libembedding_test.ppm";
    ASSERT(create_test_image(test_img), "create test image");

    lembed_image_options_t opts = lembed_image_options_default();
    lembed_image_embedding_t* ctx = NULL;

    lembed_status_t s = lembed_image_embedding_create(&opts, &ctx);
    ASSERT(s == LEMBED_OK, "create image embedding");
    if (s != LEMBED_OK) {
        printf("Cannot create image embedder: %s\n", lembed_last_error());
        return 1;
    }

    int dim = lembed_image_embedding_dim(ctx);
    ASSERT(dim == 512, "CLIP ViT-B-32 dim is 512");

    const char* files[] = { test_img };
    lembed_embeddings_t result = {0};
    s = lembed_image_embedding_embed_files(ctx, files, 1, 0, &result);
    ASSERT(s == LEMBED_OK, "embed image");
    ASSERT(result.num_embeddings == 1, "got 1 embedding");
    ASSERT(result.dim == 512, "result dim 512");

    if (s == LEMBED_OK) {
        /* Verify L2 norm ~1.0 */
        float norm = 0;
        for (int i = 0; i < 512; i++) {
            norm += result.data[i] * result.data[i];
        }
        norm = sqrtf(norm);
        ASSERT(fabsf(norm - 1.0f) < 0.01f, "image embedding L2 norm ~1.0");
    }

    lembed_embeddings_free(&result);
    lembed_image_embedding_free(ctx);
    remove(test_img);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
