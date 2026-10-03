/*
 * test_cpp_wrappers.c - Compile-and-resolve checks for the public C++ wrappers.
 *
 * This test exists because the C++ wrapper layer had no coverage at all, and
 * three defects lived there unnoticed for as long as it was unwritten:
 *
 *   1. provider.hpp included embedding_model.hpp and llama_provider.hpp before
 *      declaring EmbeddingProvider, and both derive from it. Whichever header a
 *      consumer included first lost: 'EmbeddingProvider' was never a complete
 *      type. No example, test or benchmark included cpp/embedding.hpp, so ctest
 *      stayed green.
 *   2. llama_provider.hpp and embedding_provider.hpp had no include guard, so
 *      including either one directly redefined the classes.
 *   3. embedding_pool.hpp read ctx->batch_size through the opaque
 *      lembed_text_embedding_t, which is an incomplete type by design.
 *
 * Defects 1 to 3 are compile errors, so building this file is most of the test.
 * The runtime assertions then pin the sparse registry resolution that
 * SparseEmbeddingOptions::model_path depends on.
 *
 * No model downloads required: nothing here loads a model.
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/cpp/embedding.hpp>
#include <libembedding/cpp/embedding_provider.hpp>
#include <libembedding/cpp/embedding_pool.hpp>

#include <stdio.h>
#include <string.h>

#include <string>
#include <type_traits>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

/* Every wrapper header must be usable as the first include a consumer picks.
 * The guards make the order irrelevant; the derived-class declarations make the
 * order matter. */
static_assert(sizeof(lembed::EmbeddingProvider) > 0, "EmbeddingProvider complete");
static_assert(sizeof(lembed::EmbeddingModel) > 0, "EmbeddingModel complete");
static_assert(sizeof(lembed::LlamaEmbeddingProvider) > 0, "LlamaEmbeddingProvider complete");
static_assert(sizeof(lembed::SparseEmbeddingModel) > 0, "SparseEmbeddingModel complete");
static_assert(sizeof(lembed::ImageEmbeddingModel) > 0, "ImageEmbeddingModel complete");
static_assert(sizeof(lembed::RerankerModel) > 0, "RerankerModel complete");
static_assert(sizeof(lembed::EmbeddingPool) > 0, "EmbeddingPool complete");

static_assert(std::is_abstract<lembed::EmbeddingProvider>::value,
              "EmbeddingProvider must stay abstract");
static_assert(std::is_base_of<lembed::EmbeddingProvider, lembed::EmbeddingModel>::value,
              "EmbeddingModel derives from EmbeddingProvider");
static_assert(std::is_base_of<lembed::EmbeddingProvider, lembed::LlamaEmbeddingProvider>::value,
              "LlamaEmbeddingProvider derives from EmbeddingProvider");

/* SparseEmbeddingOptions::model_path used to be dead: the constructor took the
 * model as a required argument and never read the field. It is now the default,
 * so an empty model must fall back to it. */
static_assert(std::is_default_constructible<lembed::SparseEmbeddingModel>::value,
              "SparseEmbeddingModel must be default-constructible");
static_assert(std::is_constructible<lembed::SparseEmbeddingModel, std::string>::value,
              "SparseEmbeddingModel(model) must still compile");

int main(void) {
    int passes = 0, failures = 0;

    /* Test: the sparse default is the canonical registry name.
     * It used to be "prithvida/SPLADE_PP_en_v1" -- misspelt org, wrong case --
     * which resolved only because the Python matcher falls back to the last path
     * segment. The C++ path had no such fallback, so it never resolved at all. */
    {
        const std::string dflt = lembed::SparseEmbeddingOptions().model_path;
        ASSERT(dflt == "prithivida/Splade_PP_en_v1",
               "default model_path is the canonical registry name");
    }

    /* Test: the sparse resolver accepts both the canonical name and the repo */
    {
        const int canonical = lembed_resolve_sparse_model("prithivida/Splade_PP_en_v1");
        ASSERT(canonical == LEMBED_SPARSE_SPLADE_PP_V1,
               "canonical sparse name resolves to the SPLADE++ entry");

        const int by_repo = lembed_resolve_sparse_model("Qdrant/Splade_PP_en_v1");
        ASSERT(by_repo == canonical, "repo form resolves to the same entry");

        const int unknown = lembed_resolve_sparse_model("definitely-not-a-model-xyz");
        ASSERT(unknown == -1, "unknown sparse model is rejected");

        /* The rejection message must name sparse, not text: the shared helper
         * used to hardcode "text" for every caller but the reranker. */
        const char* err = lembed_last_error();
        ASSERT(err != NULL && strstr(err, "sparse") != NULL,
               "rejection message mentions sparse");
        ASSERT(err != NULL && strstr(err, "lembed_list_sparse_models") != NULL,
               "rejection message points at the sparse lister");
    }

    /* Test: the text and reranker resolvers behave the same way */
    {
        ASSERT(lembed_resolve_text_model("BAAI/bge-small-en-v1.5") >= 0,
               "text repo resolves");
        ASSERT(lembed_resolve_text_model("nope-not-a-model-xyz") == -1,
               "unknown text model is rejected");
    }

    /* create_embedding_provider is deliberately not exercised here: it always
     * constructs a provider, and for a .gguf that means a real load attempt. This
     * is a no-network test. Its routing is a job for an integration test. */

    printf("%s: %d passed, %d failed\n",
           failures == 0 ? "PASS" : "FAIL", passes, failures);
    return failures == 0 ? 0 : 1;
}