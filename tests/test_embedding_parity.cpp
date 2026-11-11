/*
 * libembedding - test_embedding_parity.cpp
 * Cross-platform parity: the embeddings of this build match the x86_64
 * reference (python/tests/data/parity_bge_small_en_v15.json) with a cosine of
 * at least 0.999 per text, at the same dimension. Integration test: it needs
 * bge-small-en-v1.5, downloaded or already in the cache.
 *
 * Runs where pytest cannot: on the Android emulator and the iOS simulator,
 * from the native test binaries.
 *
 *   test_embedding_parity [reference.json]
 *
 * The reference path is argv[1], else LEMBED_PARITY_REF, else the source-tree
 * path compiled in. LEMBED_PARITY_CACHE, when set, is used as the model cache
 * in offline mode (a cache filled beforehand, e.g. by the host of a simulator).
 *
 * Auteur: David Orel
 * Version: 1.13.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <cJSON.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#ifndef LEMBED_PARITY_REF_DEFAULT
#define LEMBED_PARITY_REF_DEFAULT "parity_bge_small_en_v15.json"
#endif

static std::string read_file(const char* path) {
    std::string out;
    FILE* f = fopen(path, "rb");
    if (!f) return out;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

int main(int argc, char** argv) {
    const char* ref_path = argc > 1 ? argv[1] : getenv("LEMBED_PARITY_REF");
    if (!ref_path || !ref_path[0]) ref_path = LEMBED_PARITY_REF_DEFAULT;

    const std::string json = read_file(ref_path);
    cJSON* root = json.empty() ? nullptr : cJSON_Parse(json.c_str());
    if (!root) {
        fprintf(stderr, "FAIL: cannot read the parity reference %s\n", ref_path);
        printf("\n0 passed, 1 failed\n");
        return 1;
    }
    const cJSON* texts_j = cJSON_GetObjectItem(root, "texts");
    const cJSON* emb_j = cJSON_GetObjectItem(root, "embeddings");
    const int dim = cJSON_GetObjectItem(root, "dim")->valueint;
    const double min_cos = cJSON_GetObjectItem(root, "min_cosine")->valuedouble;
    const int n = cJSON_GetArraySize(texts_j);

    std::vector<const char*> texts;
    for (int i = 0; i < n; i++) texts.push_back(cJSON_GetArrayItem(texts_j, i)->valuestring);

    int passes = 0, failures = 0;
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.show_download_progress = 0;
    const char* cache = getenv("LEMBED_PARITY_CACHE");
    if (cache && cache[0]) {
        opts.cache_dir = cache;
        opts.offline = 1;
    }

    lembed_text_embedding_t* ctx = nullptr;
    if (lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
        fprintf(stderr, "FAIL: create bge-small-en-v1.5: %s\n", lembed_last_error());
        cJSON_Delete(root);
        printf("\n0 passed, 1 failed\n");
        return 1;
    }

    lembed_embeddings_t result = {0};
    if (lembed_text_embedding_embed(ctx, texts.data(), n, 0, &result) != LEMBED_OK) {
        fprintf(stderr, "FAIL: embed: %s\n", lembed_last_error());
        failures++;
    } else if (result.num_embeddings != n || result.dim != dim) {
        fprintf(stderr, "FAIL: got %d x %d, reference is %d x %d\n",
                result.num_embeddings, result.dim, n, dim);
        failures++;
    } else {
        passes++;
        for (int i = 0; i < n; i++) {
            const cJSON* row = cJSON_GetArrayItem(emb_j, i);
            const float* got = result.data + (size_t)i * dim;
            double dot = 0, na = 0, nb = 0;
            for (int k = 0; k < dim; k++) {
                const double e = cJSON_GetArrayItem(row, k)->valuedouble;
                dot += got[k] * e;
                na += (double)got[k] * got[k];
                nb += e * e;
            }
            const double cosv = dot / (sqrt(na) * sqrt(nb) + 1e-12);
            printf("text %d: cosine vs x86_64 reference = %.6f\n", i, cosv);
            if (cosv >= min_cos) {
                passes++;
            } else {
                fprintf(stderr, "FAIL: text %d cosine %.6f < %.3f\n", i, cosv, min_cos);
                failures++;
            }
        }
    }

    lembed_embeddings_free(&result);
    lembed_text_embedding_free(ctx);
    cJSON_Delete(root);
    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
