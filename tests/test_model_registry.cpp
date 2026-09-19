/*
 * test_model_registry.c - Unit tests for the model registry
 * No model downloads required.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

int main(void) {
    int passes = 0, failures = 0;

    /* Test: list text models */
    {
        const lembed_model_info_t* models;
        int count;
        lembed_status_t s = lembed_list_text_models(&models, &count);
        ASSERT(s == LEMBED_OK, "lembed_list_text_models returns OK");
        ASSERT(count == LEMBED_TEXT_MODEL_COUNT, "text model count matches enum");
        ASSERT(count > 0, "at least one text model");
    }

    /* Test: get specific text model info */
    {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_text_model_info(LEMBED_TEXT_BGE_SMALL_EN_V15, &info);
        ASSERT(s == LEMBED_OK, "get BGE small EN v1.5 OK");
        ASSERT(info.dim == 384, "BGE small dim is 384");
        ASSERT(strcmp(info.model_code, "Xenova/bge-small-en-v1.5") == 0, "BGE small model_code");
        ASSERT(info.pooling == LEMBED_POOLING_CLS, "BGE small uses CLS pooling");
    }

    /* Test: all text models have valid fields */
    {
        for (int i = 0; i < LEMBED_TEXT_MODEL_COUNT; i++) {
            lembed_model_info_t info;
            lembed_status_t s = lembed_get_text_model_info((lembed_text_model_t)i, &info);
            ASSERT(s == LEMBED_OK, "text model info OK");
            ASSERT(info.model_name != NULL && info.model_name[0] != '\0', "model_name non-empty");
            ASSERT(info.model_code != NULL && info.model_code[0] != '\0', "model_code non-empty");
            ASSERT(info.model_file != NULL && info.model_file[0] != '\0', "model_file non-empty");
            ASSERT(info.dim > 0, "dim > 0");
            ASSERT(info.max_tokens > 0, "max_tokens > 0");
        }
    }

    /* Test: invalid model enum */
    {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_text_model_info((lembed_text_model_t)999, &info);
        ASSERT(s == LEMBED_ERROR_INVALID_ARGUMENT, "invalid model returns error");
    }

    /* Test: sparse models */
    {
        const lembed_model_info_t* models;
        int count;
        lembed_status_t s = lembed_list_sparse_models(&models, &count);
        ASSERT(s == LEMBED_OK, "list sparse models OK");
        ASSERT(count == LEMBED_SPARSE_MODEL_COUNT, "sparse count matches");
        ASSERT(count == 2, "2 sparse models");
    }

    /* Test: image models */
    {
        const lembed_model_info_t* models;
        int count;
        lembed_status_t s = lembed_list_image_models(&models, &count);
        ASSERT(s == LEMBED_OK, "list image models OK");
        ASSERT(count == LEMBED_IMAGE_MODEL_COUNT, "image count matches");

        lembed_model_info_t info;
        s = lembed_get_image_model_info(LEMBED_IMAGE_CLIP_VIT_B32, &info);
        ASSERT(s == LEMBED_OK, "CLIP ViT-B-32 info OK");
        ASSERT(info.dim == 512, "CLIP dim is 512");
    }

    /* Test: reranker models */
    {
        const lembed_model_info_t* models;
        int count;
        lembed_status_t s = lembed_list_reranker_models(&models, &count);
        ASSERT(s == LEMBED_OK, "list reranker models OK");
        ASSERT(count == LEMBED_RERANKER_MODEL_COUNT, "reranker count matches");
    }

    /* Test: find model by code */
    {
        int idx = lembed_find_text_model_by_code("Xenova/bge-small-en-v1.5");
        ASSERT(idx == LEMBED_TEXT_BGE_SMALL_EN_V15, "find by code works");

        idx = lembed_find_text_model_by_code("nonexistent/model");
        ASSERT(idx == -1, "nonexistent model returns -1");
    }

    /* Test: default options */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        ASSERT(opts.model == LEMBED_TEXT_MODEL_DEFAULT, "default text model");
        ASSERT(opts.provider == LEMBED_PROVIDER_CPU, "default provider CPU");
        ASSERT(opts.show_download_progress == 1, "default show progress");
        ASSERT(opts.batch_size == LEMBED_DEFAULT_BATCH_SIZE, "default batch_size 256");
        ASSERT(opts.offline == 0, "default offline false");
        ASSERT(opts.pooling == LEMBED_POOLING_MEAN, "default pooling MEAN");
    }

    /* Test: model_desc_t is available */
    {
        lembed_model_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        ASSERT(sizeof(desc) > 0, "model_desc_t is non-empty");
    }

    /* Test: error messages */
    {
        const char* msg = lembed_status_message(LEMBED_OK);
        ASSERT(strcmp(msg, "Success") == 0, "OK message");
        msg = lembed_status_message(LEMBED_ERROR_ONNX_RUNTIME);
        ASSERT(msg != NULL && strlen(msg) > 0, "error message non-empty");
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
