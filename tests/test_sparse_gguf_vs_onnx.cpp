/*
 * test_sparse_gguf_vs_onnx.cpp - Sparse GGUF against sparse ONNX.
 *
 * The fidelity criterion. Both models derive from the same HuggingFace export,
 * prithivida/Splade_PP_en_v1, so on the same text they must produce nearly the
 * same sparse vector. Cosine >= 0.90 is the bar; the measured value is printed so
 * a regression shows up as a drift, not just a failure.
 *
 * Requires network: two separate downloads. Guarded by
 * LIBEMBEDDING_INTEGRATION_TESTS in tests/CMakeLists.txt, so a run without that
 * option never reaches this file at all.
 *
 * Two deliberate choices:
 *
 *   - The texts are plain ASCII English. The tokeniser is shared between the two
 *     backends, so on any text they agree -- but both are approximations of
 *     HuggingFace's BertNormalizer (ASCII-only lowercasing, no accent
 *     stripping). Feeding accented text here would measure the approximation
 *     instead of the runtime, which is not what this test is for.
 *
 *   - top_k is 0, meaning no truncation. A cosine between two truncated vectors
 *     is dominated by which terms each side happened to keep, and would measure
 *     the pruning rather than the embeddings.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/libembedding.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); g_fail++; } \
} while (0)

namespace {

const char* kTexts[] = {
    "a fast brown fox jumps over the lazy dog",
    "how to bake sourdough bread at home",
    "the cat sat on the mat and looked at me",
    "vector databases store embeddings and search them",
    "machine learning models learn from data",
    "the weather today is cold and rainy",
    "installing a package manager on windows",
    "she sells seashells by the seashore",
    "what is the capital city of france",
    "an apple a day keeps the doctor away",
    "programming in C++ is a joy",
    "the library ships a sparse embedding backend",
};
const int kNumTexts = (int)(sizeof(kTexts) / sizeof(kTexts[0]));

const char* kGgufRepo     = "cstr/splade-pp-en-v1-GGUF";
const char* kGgufFilename = "splade-pp-en-v1-q8_0.gguf";

const float kMinCosine = 0.90f;

/* Sparse cosine over the shared ids. Neither vector is dense and neither is
 * normalised, so this has to be computed over the intersection. */
double sparse_cosine(const lembed_sparse_embedding_t& a,
                     const lembed_sparse_embedding_t& b) {
    /* Both come out sorted by descending weight, so the shorter walk is a merge
     * from the front; but the ids are not sorted, so look the rest up. */
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int i = 0; i < a.length; i++) na += (double)a.values[i] * (double)a.values[i];
    for (int i = 0; i < b.length; i++) nb += (double)b.values[i] * (double)b.values[i];
    if (na <= 0.0 || nb <= 0.0) return 0.0;

    for (int i = 0; i < a.length; i++) {
        for (int j = 0; j < b.length; j++) {
            if (a.indices[i] == b.indices[j]) {
                dot += (double)a.values[i] * (double)b.values[j];
                break;
            }
        }
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

bool embed_all(lembed_sparse_embedding_ctx_t* ctx, lembed_sparse_embeddings_t* out) {
    memset(out, 0, sizeof(*out));
    return lembed_sparse_text_embedding_embed(ctx, kTexts, kNumTexts, 8, nullptr,
                                             out) == LEMBED_OK;
}

} /* namespace */

int main(void) {
    /* ---- ONNX side ------------------------------------------------------- */

    lembed_sparse_options_t onnx_opts = lembed_sparse_options_default();
    onnx_opts.model = LEMBED_SPARSE_SPLADE_PP_V1;
    onnx_opts.top_k = 0;
    onnx_opts.min_weight = 0.0f;

    lembed_sparse_embedding_ctx_t* onnx = nullptr;
    const lembed_status_t onnx_status =
        lembed_sparse_text_embedding_create(&onnx_opts, &onnx);
    if (onnx_status != LEMBED_OK) {
        fprintf(stderr, "skip: the ONNX SPLADE model is unavailable (%s)\n",
                lembed_status_message(onnx_status));
        printf("SKIP: ONNX sparse model unavailable\n");
        return 0;
    }

    /* ---- GGUF side ------------------------------------------------------- */

    /* Fetch first so the file can be inspected on its own: the criterion is
     * that the runtime accepts exactly what the inspector calls a SPLADE file. */
    char* gguf_path = nullptr;
    const lembed_status_t fetch_status =
        lembed_ensure_gguf_model(kGgufRepo, kGgufFilename, nullptr, 0, 0, &gguf_path);
    if (fetch_status != LEMBED_OK || !gguf_path) {
        fprintf(stderr, "FAIL: the SPLADE GGUF could not be downloaded (%s)\n",
                lembed_status_message(fetch_status));
        lembed_sparse_text_embedding_free(onnx);
        printf("FAIL: 0 passed, 1 failed\n");
        return 1;
    }

    lembed_gguf_desc_t desc;
    const lembed_status_t inspect_status =
        lembed_gguf_inspect(gguf_path, &desc);
    CHECK(inspect_status == LEMBED_OK, "the GGUF header can be inspected");
    CHECK((desc.capabilities & LEMBED_GGUF_CAP_SPLADE) != 0,
          "the inspector reports a SPLADE capability");
    CHECK(desc.sparse_formula == LEMBED_GGUF_SPARSE_FORMULA_SPLADE,
          "the inspector reports the SPLADE formula");

    lembed_sparse_options_t gguf_opts = onnx_opts;
    lembed_sparse_embedding_ctx_t* gguf = nullptr;
    const lembed_status_t gguf_status =
        lembed_sparse_text_embedding_create_from_gguf_path(gguf_path, &gguf_opts,
                                                            &gguf);
    lembed_free_string(gguf_path);
    if (gguf_status != LEMBED_OK) {
        fprintf(stderr, "FAIL: the SPLADE GGUF could not be loaded (%s: %s)\n",
                lembed_status_message(gguf_status), lembed_last_error());
        lembed_sparse_text_embedding_free(onnx);
        printf("FAIL: 0 passed, 1 failed\n");
        return 1;
    }

    CHECK(lembed_sparse_text_embedding_max_length(gguf) > 0,
          "the GGUF reports a usable sequence length");

    /* ---- Compare --------------------------------------------------------- */

    lembed_sparse_embeddings_t onnx_out;
    lembed_sparse_embeddings_t gguf_out;
    const bool onnx_ok = embed_all(onnx, &onnx_out);
    const bool gguf_ok = embed_all(gguf, &gguf_out);

    CHECK(onnx_ok, "the ONNX side embeds every text");
    CHECK(gguf_ok, "the GGUF side embeds every text");
    CHECK(onnx_out.count == kNumTexts, "the ONNX side returns one vector per text");
    CHECK(gguf_out.count == kNumTexts, "the GGUF side returns one vector per text");

    double worst = 2.0;
    double total = 0.0;
    int compared = 0;
    if (onnx_ok && gguf_ok && onnx_out.count == kNumTexts && gguf_out.count == kNumTexts) {
        for (int i = 0; i < kNumTexts; i++) {
            CHECK(gguf_out.items[i].length > 0, "the GGUF vector is not empty");
            CHECK(onnx_out.items[i].length > 0, "the ONNX vector is not empty");
            if (gguf_out.items[i].length == 0 || onnx_out.items[i].length == 0) continue;
            const double c = sparse_cosine(onnx_out.items[i], gguf_out.items[i]);
            if (c < worst) worst = c;
            total += c;
            compared++;
            CHECK(c >= kMinCosine, "GGUF and ONNX agree on this text");
        }
    }

    if (compared > 0) {
        printf("sparse GGUF vs ONNX: min cos %.4f, mean cos %.4f over %d texts "
               "(bar %.2f)\n", worst, total / compared, compared, kMinCosine);
    }

    lembed_sparse_embeddings_free(&onnx_out);
    lembed_sparse_embeddings_free(&gguf_out);
    lembed_sparse_text_embedding_free(onnx);
    lembed_sparse_text_embedding_free(gguf);

    printf("PASS: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}