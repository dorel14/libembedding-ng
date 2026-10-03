/*
 * test_llama_embedding.cpp - Standalone test for llama.cpp backend
 * Compiles directly against llama.cpp without full libembedding build.
 *
 * The C API is declared by the public headers below. This file used to repeat
 * those declarations locally so it could build without the library, but the
 * includes were still there: every type in the local block (lembed_embeddings_t,
 * lembed_status_t, lembed_text_options_t, LEMBED_OK) collided with the real
 * one and the target failed to compile. It never produced an executable, so
 * ctest reported it as "Not Run" while the other tests passed.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// The real C API headers: single source of truth for every type and prototype.
#include <libembedding/text_embedding.h>
#include <libembedding/error.h>
#include <libembedding/llamacpp_backend.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, lembed_last_error()); \
        failures++; \
    } else { \
        passes++; \
    } \
} while(0)

static int passes = 0, failures = 0;

static float cosine_sim(const float* a, const float* b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-8f) ? dot / denom : 0.0f;
}

static float l2_norm(const float* v, int dim) {
    float s = 0;
    for (int i = 0; i < dim; i++) s += v[i] * v[i];
    return sqrtf(s);
}

static int count_nan_inf(const float* data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (std::isnan(data[i]) || std::isinf(data[i])) count++;
    }
    return count;
}

static void test_basic(const char* gguf_path) {
    fprintf(stderr, "--- Test 1: Basic Functionality ---\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* ctx = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(gguf_path, &opts, &ctx);
    ASSERT(s == LEMBED_OK, "create llama embedder");
    if (s != LEMBED_OK) return;

    int dim = lembed_text_embedding_dim(ctx);
    ASSERT(dim == 384, "embedding dim is 384");

    const char* texts[] = {"Hello world", "Machine learning is great"};
    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(ctx, texts, 2, 0, &result);
    ASSERT(s == LEMBED_OK, "embed 2 texts");
    ASSERT(result.num_embeddings == 2, "got 2 embeddings");
    ASSERT(result.dim == 384, "result dim is 384");

    if (s == LEMBED_OK) {
        ASSERT(count_nan_inf(result.data, 2 * 384) == 0, "no NaN/Inf in embeddings");
        float norm0 = l2_norm(result.data, 384);
        float norm1 = l2_norm(result.data + 384, 384);
        ASSERT(fabsf(norm0 - 1.0f) < 0.01f, "embedding 0 L2 norm ~1.0");
        ASSERT(fabsf(norm1 - 1.0f) < 0.01f, "embedding 1 L2 norm ~1.0");
        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(ctx);
    fprintf(stderr, "  Basic: %d passed, %d failed\n\n", passes, failures);
}

static void test_edge_cases(const char* gguf_path) {
    fprintf(stderr, "--- Test 2: Edge Cases ---\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* ctx = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(gguf_path, &opts, &ctx);
    ASSERT(s == LEMBED_OK, "create embedder for edge cases");
    if (s != LEMBED_OK) return;

    const char* edge_texts[] = {
        "",
        "a",
        "   ",
        "\t",
        "\n",
        "12345 67890",
        "!@#$%^&*()",
        "café résumé naïve",
    };
    int n_edge = sizeof(edge_texts) / sizeof(edge_texts[0]);

    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(ctx, edge_texts, n_edge, 0, &result);
    ASSERT(s == LEMBED_OK, "embed edge case texts");
    ASSERT(result.num_embeddings == n_edge, "got correct number of embeddings");

    if (s == LEMBED_OK) {
        int nan_count = count_nan_inf(result.data, n_edge * 384);
        ASSERT(nan_count == 0, "no NaN/Inf in edge case embeddings");

        for (int i = 0; i < n_edge; i++) {
            float norm = l2_norm(result.data + (size_t)i * 384, 384);
            ASSERT(fabsf(norm - 1.0f) < 0.05f, "edge case embedding normalized");
        }
        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(ctx);
    fprintf(stderr, "  Edge cases: %d passed, %d failed\n\n", passes, failures);
}

static void test_user_corpus(const char* gguf_path) {
    fprintf(stderr, "--- Test 3: User Corpus ---\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* ctx = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(gguf_path, &opts, &ctx);
    ASSERT(s == LEMBED_OK, "create embedder for user corpus");
    if (s != LEMBED_OK) return;

    const char* corpus[] = {
        "Hello world.",
        "Bonjour le monde.",
        "Hallo Welt.",
        "Hola mundo.",
        "Ciao mondo.",
        "Olá mundo.",
        "Привет мир.",
        "こんにちは世界。",
        "안녕하세요 세계.",
        "你好世界。",
        "Machine learning transforms data into insights.",
        "L'intelligence artificielle transforme les données.",
        "Künstliche Intelligenz verändert die Welt.",
        "Climate change affects global weather patterns.",
        "Quantum computing promises revolution.",
        "The history of ancient Rome spans centuries.",
        "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing.",
        "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think.",
        "Les algorithmes d'apprentissage automatique peuvent identifier des motifs dans de grands ensembles de données automatiquement sans programmation explicite.",
        "Die künstliche Intelligenz hat bedeutende Fortschritte gemacht insbesondere in den Bereichen maschinelles Lernen und Verarbeitung natürlicher Sprache.",
    };
    int n_corpus = sizeof(corpus) / sizeof(corpus[0]);

    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(ctx, corpus, n_corpus, 0, &result);
    ASSERT(s == LEMBED_OK, "embed user corpus");
    ASSERT(result.num_embeddings == n_corpus, "got correct number of embeddings");

    if (s == LEMBED_OK) {
        int nan_count = count_nan_inf(result.data, n_corpus * 384);
        ASSERT(nan_count == 0, "no NaN/Inf in corpus embeddings");

        float sim_greeting = cosine_sim(result.data, result.data + 384, 384);
        float sim_different = cosine_sim(result.data, result.data + 16 * 384, 384);

        fprintf(stderr, "  Cosine (greeting vs greeting): %.4f\n", sim_greeting);
        fprintf(stderr, "  Cosine (greeting vs quantum):  %.4f\n", sim_different);

        ASSERT(sim_greeting > sim_different, "similar texts have higher similarity");
        ASSERT(sim_greeting > 0.3f, "greeting similarity > 0.3");

        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(ctx);
    fprintf(stderr, "  User corpus: %d passed, %d failed\n\n", passes, failures);
}

static void test_stress(const char* gguf_path) {
    fprintf(stderr, "--- Test 4: Stress Test ---\n");

    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* ctx = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(gguf_path, &opts, &ctx);
    ASSERT(s == LEMBED_OK, "create embedder for stress test");
    if (s != LEMBED_OK) return;

    int n_texts = 100;
    std::vector<std::string> texts;
    std::vector<const char*> text_ptrs;
    for (int i = 0; i < n_texts; i++) {
        texts.push_back("Document " + std::to_string(i) + " with content about topic " + std::to_string(i % 10));
        text_ptrs.push_back(texts.back().c_str());
    }

    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(ctx, text_ptrs.data(), n_texts, 0, &result);
    ASSERT(s == LEMBED_OK, "embed 100 texts");
    ASSERT(result.num_embeddings == n_texts, "got 100 embeddings");

    if (s == LEMBED_OK) {
        int nan_count = count_nan_inf(result.data, n_texts * 384);
        ASSERT(nan_count == 0, "no NaN/Inf in stress test");
        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(ctx);
    fprintf(stderr, "  Stress: %d passed, %d failed\n\n", passes, failures);
}

/* The GGUF fixture is not vendored. Resolve it from argv, then the environment,
 * then a relative default, so the test is not tied to one machine's home
 * directory. */
static bool file_readable(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    const char* env_path = getenv("LEMBED_TEST_GGUF");
    const char* gguf_path = (argc > 1) ? argv[1]
                       : (env_path && env_path[0]) ? env_path
                       : "./models/all-MiniLM-L6-v2-Q4_K_M.gguf";

    fprintf(stderr, "=== llama.cpp Backend Tests ===\n");
    fprintf(stderr, "Model: %s\n\n", gguf_path);

    if (!lembed_llama_backend_available()) {
        fprintf(stderr, "ERROR: llama.cpp backend not compiled in\n");
        return 1;
    }

    /* These are integration tests: they need a real GGUF model. Skip cleanly
     * when it is absent so ctest does not report a failure for a missing
     * fixture. */
    if (!file_readable(gguf_path)) {
        fprintf(stderr, "SKIP: GGUF model not found: %s\n", gguf_path);
        fprintf(stderr, "Pass a path as argv[1] or set LEMBED_TEST_GGUF to run them.\n");
        return 0;
    }

    test_basic(gguf_path);
    test_edge_cases(gguf_path);
    test_user_corpus(gguf_path);
    test_stress(gguf_path);

    fprintf(stderr, "=== TOTAL: %d passed, %d failed ===\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
