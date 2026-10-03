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

    /* Test: resolve_text_model accepts BOTH name forms.
     * The autotune entry points take a model as a string, and the registry
     * gives every model two different strings for it. Requiring the caller to
     * know which one a given function wants was a runtime trap; both must
     * resolve to the same entry. */
    {
        int by_code = lembed_resolve_text_model("Qdrant/all-MiniLM-L6-v2-onnx");
        int by_name = lembed_resolve_text_model("sentence-transformers/all-MiniLM-L6-v2");
        ASSERT(by_code == LEMBED_TEXT_ALL_MINILM_L6_V2, "resolve by model_code");
        ASSERT(by_name == LEMBED_TEXT_ALL_MINILM_L6_V2, "resolve by canonical model_name");
        ASSERT(by_code == by_name, "both forms resolve to the same entry");
    }

/* Resolution is exact-code-first, then exact-name. That has a consequence worth
     * pinning down: a repo belongs to exactly one entry, so a quantized repo
     * resolves to the quantized entry even when the caller passed the name of
     * its FP32 sibling. The canonical name carries no quantization, so it
     * resolves to the first entry declaring it -- the FP32 one. */
    {
        int by_q_repo = lembed_resolve_text_model("Xenova/all-MiniLM-L6-v2");
        int variant = lembed_find_text_model_variant(
            "sentence-transformers/all-MiniLM-L6-v2", LEMBED_QUANTIZATION_DYNAMIC);
        ASSERT(by_q_repo == LEMBED_TEXT_ALL_MINILM_L6_V2_Q,
               "a quantized repo resolves to the quantized entry");
        ASSERT(by_q_repo == variant, "and it is the entry variant lookup returns");

        int by_fp32_repo = lembed_resolve_text_model("Qdrant/all-MiniLM-L6-v2-onnx");
        ASSERT(by_fp32_repo == LEMBED_TEXT_ALL_MINILM_L6_V2, "FP32 repo resolves to FP32 entry");
        ASSERT(by_fp32_repo != by_q_repo, "the two are distinct sets of weights");
    }

    /* An unknown model must fail, and say so, rather than silently resolving to
     * entry 0 -- that bug once made every HuggingFace name load MiniLM-L6. */
    {
        ASSERT(lembed_resolve_text_model("does-not/exist-at-all") == -1,
               "unknown model returns -1");
        ASSERT(lembed_last_error() != NULL && strlen(lembed_last_error()) > 0,
               "unknown model sets a diagnostic");
    }
    ASSERT(lembed_resolve_text_model(NULL) == -1, "NULL model returns -1");

    /* Reranker: both forms resolve, and the INT8 entry keeps its own name so
     * it is deliberately *not* reachable as a "sibling" of the FP32 one. */
    {
        int f = lembed_resolve_reranker_model("jinaai/jina-reranker-v1-turbo-en");
        int q = lembed_resolve_reranker_model("jinaai/jina-reranker-v1-turbo-en-quantized");
        ASSERT(f == LEMBED_RERANKER_JINA_V1_TURBO_EN, "reranker resolve by name");
        ASSERT(q == LEMBED_RERANKER_JINA_V1_TURBO_EN_QUANTIZED, "reranker INT8 is its own entry");
        ASSERT(f != q, "reranker INT8 is a distinct entry");
    }

    /* The Qdrant/*-onnx-Q entries ship FP16 weights behind a _Q suffix and a
     * "Quantized" description. Their initializers are all FLOAT16 and their graph
     * is ORT-optimized, so declaring them STATIC made quantization="static" hand
     * back float16 weights: smaller on disk, slower than FP32 on any CPU without
     * native FP16 arithmetic, and not what the caller asked for. They now declare
     * LEMBED_QUANTIZATION_FP16, which also means quantization="static" correctly
     * finds no entry for these models instead of silently returning FP16. */
    {
        struct { int enum_value; const char* model_code; } fp16_entries[] = {
            { LEMBED_TEXT_BGE_BASE_EN_V15_Q,   "Qdrant/bge-base-en-v1.5-onnx-Q" },
            { LEMBED_TEXT_BGE_LARGE_EN_V15_Q,  "Qdrant/bge-large-en-v1.5-onnx-Q" },
            { LEMBED_TEXT_BGE_SMALL_EN_V15_Q,  "Qdrant/bge-small-en-v1.5-onnx-Q" },
            { LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2_Q,
              "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q" },
        };
        for (size_t i = 0; i < sizeof(fp16_entries) / sizeof(fp16_entries[0]); i++) {
            lembed_model_info_t info;
            memset(&info, 0, sizeof(info));
            ASSERT(lembed_get_text_model_info((lembed_text_model_t)fp16_entries[i].enum_value,
                                             &info) == LEMBED_OK,
                   "FP16 entry is readable");
            ASSERT(info.quantization == LEMBED_QUANTIZATION_FP16,
                   "Qdrant -Q entry declares fp16, not static");
            ASSERT(strcmp(info.model_code, fp16_entries[i].model_code) == 0,
                   "FP16 entry model_code");
            ASSERT(strstr(info.description, "FP16") != NULL,
                   "FP16 entry description names the format");
            ASSERT(strstr(info.description, "Quantized") == NULL,
                   "FP16 entry description no longer claims it is quantized");
            ASSERT(lembed_resolve_text_model(info.model_code) == fp16_entries[i].enum_value,
                   "FP16 entry resolves by its repo");
        }

        /* And the mode that used to be accepted here must now be refused. */
        ASSERT(lembed_find_text_model_variant("BAAI/bge-small-en-v1.5",
                                              LEMBED_QUANTIZATION_STATIC) == -1,
               "bge-small has no static (INT8) variant: asking must fail, not return FP16");

        /* The genuine INT8 sibling keeps declaring dynamic. */
        lembed_model_info_t minilm_q;
        ASSERT(lembed_get_text_model_info(LEMBED_TEXT_ALL_MINILM_L6_V2_Q, &minilm_q) == LEMBED_OK,
               "MiniLM _Q readable");
        ASSERT(minilm_q.quantization == LEMBED_QUANTIZATION_DYNAMIC,
               "MiniLM _Q still declares dynamic INT8");
    }

    /* BGE-small now has a real INT8 sibling, so quantization="dynamic" resolves
     * to int8 weights instead of failing. This is the entry the benchmark's INT8
     * row measures; without it "no INT8 variant exists" was the honest answer. */
    {
        int idx = lembed_find_text_model_variant("BAAI/bge-small-en-v1.5",
                                                 LEMBED_QUANTIZATION_DYNAMIC);
        ASSERT(idx == LEMBED_TEXT_BGE_SMALL_EN_V15_INT8,
               "bge-small dynamic resolves to the INT8 entry");

        lembed_model_info_t info;
        memset(&info, 0, sizeof(info));
        ASSERT(lembed_get_text_model_info(LEMBED_TEXT_BGE_SMALL_EN_V15_INT8, &info) == LEMBED_OK,
               "BGE INT8 entry is readable");
        ASSERT(info.quantization == LEMBED_QUANTIZATION_DYNAMIC, "declares dynamic");
        ASSERT(strcmp(info.model_code, "onnx-community/bge-small-en-v1.5-ONNX") == 0,
               "INT8 repo is onnx-community");
        ASSERT(strcmp(info.model_file, "onnx/model_quantized.onnx") == 0,
               "INT8 graph file");
        ASSERT(info.dim == 384, "INT8 sibling keeps the same dimension");
        ASSERT(info.pooling == LEMBED_POOLING_CLS, "INT8 sibling keeps the same pooling");

        /* External data sits next to the graph; without the additional-files
         * entry the downloader fetches 414 KB of graph and the session cannot
         * load the 33 MB of weights. */
        ASSERT(strcmp(info.model_code, lembed__text_models[LEMBED_TEXT_BGE_SMALL_EN_V15].model_code)
                   != 0,
               "INT8 entry is a different repo from its FP32 sibling");
    }

    /* Guard the enum/table alignment.
     *
     * lembed__text_models is indexed directly by lembed_text_model_t, so the
     * table and the enum must stay in the same order. Adding a registry entry
     * in the middle of the table while appending its enum value at the end
     * compiles, passes the count check, and silently shifts every later model by
     * one: an enum that used to name Snowflake Arctic now loads BGE. Pin the
     * order with a few anchors spread across the table, including the last entry. */
    {
        struct { int expected; const char* code; } anchors[] = {
            { LEMBED_TEXT_ALL_MINILM_L6_V2,           "Qdrant/all-MiniLM-L6-v2-onnx" },
            { LEMBED_TEXT_ALL_MINILM_L6_V2_Q,         "Xenova/all-MiniLM-L6-v2" },
            { LEMBED_TEXT_BGE_SMALL_EN_V15,           "Xenova/bge-small-en-v1.5" },
            { LEMBED_TEXT_BGE_SMALL_EN_V15_Q,         "Qdrant/bge-small-en-v1.5-onnx-Q" },
            { LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_L_Q, "snowflake/snowflake-arctic-embed-l" },
            { LEMBED_TEXT_BGE_SMALL_EN_V15_INT8,      "onnx-community/bge-small-en-v1.5-ONNX" },
        };
        for (size_t i = 0; i < sizeof(anchors) / sizeof(anchors[0]); i++) {
            int idx = anchors[i].expected;
            ASSERT(idx >= 0 && idx < LEMBED_TEXT_MODEL_COUNT, "anchor index in range");
            ASSERT(strcmp(lembed__text_models[idx].model_code, anchors[i].code) == 0,
                   "enum index selects the model it names");
        }
        /* The appended enum value must be the last table slot. */
        ASSERT(strcmp(lembed__text_models[LEMBED_TEXT_MODEL_COUNT - 1].model_code,
                      "onnx-community/bge-small-en-v1.5-ONNX") == 0,
               "last table slot is the appended BGE INT8 entry");
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
