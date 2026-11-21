/*
 * test_backend_introspection.cpp - effective backend and provider (T9, D3).
 *
 * Hermetic part (always runs, no network):
 *   - every lembed_*_backend_name / lembed_*_effective_provider accessor
 *     returns NULL for a NULL context;
 *   - a provider other than CPU that this build cannot attach makes every
 *     ONNX create path fail with LEMBED_ERROR_UNSUPPORTED and a message naming
 *     the provider, instead of silently running on the CPU. The check runs
 *     before the model is parsed, so fake model files are enough;
 *   - the same fake files with the CPU provider fail differently (the model
 *     is invalid), which proves the error comes from the provider alone.
 *
 * Cache part (offline, skipped when the model is not already cached, never
 * downloads): bge-small-en-v1.5 (ONNX), the Snowflake-XS GGUF (llama.cpp) and
 * the Jina reranker GGUF (llama.cpp) report the runtime that serves them.
 *
 * Auteur: David Orel
 * Version: 1.14.0
 *
 * SPDX-License-Identifier: MIT
 */

/* Linux/macOS use the header-only implementation; Windows tests link to the DLL. */
#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>
#include <libembedding/detail/gguf_lookup.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

static int passes = 0, failures = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while (0)

static bool streq(const char* a, const char* b) {
    return a && b && std::strcmp(a, b) == 0;
}

static bool error_mentions(const char* needle) {
    const char* err = lembed_last_error();
    return err && std::strstr(err, needle) != nullptr;
}

static void write_file(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out << content;
}

/* A model directory whose files are not a real model: enough to reach the
 * session creation, where the provider is checked before anything is parsed. */
static fs::path make_fake_model_dir() {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path root = fs::temp_directory_path() / ("lembed_backend_" + std::to_string(tick));
    write_file(root / "model.onnx", "not an onnx model");
    write_file(root / "tokenizer.json", "{}");
    write_file(root / "config.json", "{\"hidden_size\": 8, \"max_position_embeddings\": 16}");
    return root;
}

static void test_null_contexts() {
    ASSERT(lembed_text_embedding_backend_name(nullptr) == nullptr, "text backend NULL");
    ASSERT(lembed_text_embedding_effective_provider(nullptr) == nullptr, "text provider NULL");
    ASSERT(lembed_sparse_text_embedding_backend_name(nullptr) == nullptr, "sparse backend NULL");
    ASSERT(lembed_sparse_text_embedding_effective_provider(nullptr) == nullptr,
           "sparse provider NULL");
    ASSERT(lembed_reranker_backend_name(nullptr) == nullptr, "reranker backend NULL");
    ASSERT(lembed_reranker_effective_provider(nullptr) == nullptr, "reranker provider NULL");
#ifndef LIBEMBEDDING_NO_IMAGE
    ASSERT(lembed_image_embedding_backend_name(nullptr) == nullptr, "image backend NULL");
    ASSERT(lembed_image_embedding_effective_provider(nullptr) == nullptr, "image provider NULL");
#endif
}

/* Providers no shipped build compiles in. CoreML is left out on Apple, where
 * ONNX Runtime may genuinely provide it. */
struct Unavailable { lembed_execution_provider_t provider; const char* name; };
static const Unavailable kUnavailable[] = {
    { LEMBED_PROVIDER_CUDA,     "CUDA" },
    { LEMBED_PROVIDER_DIRECTML, "DirectML" },
    { LEMBED_PROVIDER_TENSORRT, "TensorRT" },
#if !defined(__APPLE__)
    { LEMBED_PROVIDER_COREML,   "CoreML" },
#endif
    /* Not an ONNX Runtime provider at all. */
    { LEMBED_PROVIDER_LLAMACPP, "llama.cpp" },
};

static void test_explicit_provider_unavailable(const fs::path& dir) {
    const std::string path = dir.string();
    const std::string onnx = "not an onnx model";
    const std::string tok = "{}";

    for (const Unavailable& u : kUnavailable) {
        char label[160];

        /* create_custom (in-memory model) */
        lembed_user_defined_model_t udm;
        std::memset(&udm, 0, sizeof(udm));
        udm.onnx_data = (const unsigned char*)onnx.data();
        udm.onnx_data_size = onnx.size();
        udm.tokenizer_json = (const unsigned char*)tok.data();
        udm.tokenizer_json_size = tok.size();
        udm.dim = 8;
        lembed_text_embedding_t* text = nullptr;
        lembed_status_t s = lembed_text_embedding_create_custom(&udm, u.provider, 1, &text);
        std::snprintf(label, sizeof(label), "text create_custom %s -> UNSUPPORTED", u.name);
        ASSERT(s == LEMBED_ERROR_UNSUPPORTED && text == nullptr, label);
        std::snprintf(label, sizeof(label), "text create_custom %s: error names it", u.name);
        ASSERT(error_mentions(u.name), label);

        /* text create_from_path */
        lembed_text_options_t topts = lembed_text_options_default();
        topts.provider = u.provider;
        topts.dim = 8;
        text = nullptr;
        s = lembed_text_embedding_create_from_path(path.c_str(), &topts, &text);
        std::snprintf(label, sizeof(label), "text create_from_path %s -> UNSUPPORTED", u.name);
        ASSERT(s == LEMBED_ERROR_UNSUPPORTED && text == nullptr, label);

        /* sparse create_from_path */
        lembed_sparse_options_t sopts = lembed_sparse_options_default();
        sopts.provider = u.provider;
        lembed_sparse_embedding_ctx_t* sparse = nullptr;
        s = lembed_sparse_text_embedding_create_from_path(path.c_str(), &sopts, &sparse);
        std::snprintf(label, sizeof(label), "sparse create_from_path %s -> UNSUPPORTED", u.name);
        ASSERT(s == LEMBED_ERROR_UNSUPPORTED && sparse == nullptr, label);

        /* reranker create_from_path */
        lembed_reranker_options_t ropts = lembed_reranker_options_default();
        ropts.provider = u.provider;
        lembed_reranker_t* rr = nullptr;
        s = lembed_reranker_create_from_path(path.c_str(), &ropts, &rr);
        std::snprintf(label, sizeof(label), "reranker create_from_path %s -> UNSUPPORTED", u.name);
        ASSERT(s == LEMBED_ERROR_UNSUPPORTED && rr == nullptr, label);
        std::snprintf(label, sizeof(label), "reranker %s: error names it", u.name);
        ASSERT(error_mentions(u.name), label);

#ifndef LIBEMBEDDING_NO_IMAGE
        /* image create_from_path */
        lembed_image_options_t iopts = lembed_image_options_default();
        iopts.provider = u.provider;
        iopts.dim = 8;
        lembed_image_embedding_t* img = nullptr;
        s = lembed_image_embedding_create_from_path(path.c_str(), &iopts, &img);
        std::snprintf(label, sizeof(label), "image create_from_path %s -> UNSUPPORTED", u.name);
        ASSERT(s == LEMBED_ERROR_UNSUPPORTED && img == nullptr, label);
#endif
    }

    /* An out-of-range value is not silently treated as CPU either. */
    lembed_reranker_options_t ropts = lembed_reranker_options_default();
    ropts.provider = (lembed_execution_provider_t)42;
    lembed_reranker_t* rr = nullptr;
    lembed_status_t s = lembed_reranker_create_from_path(path.c_str(), &ropts, &rr);
    ASSERT(s == LEMBED_ERROR_UNSUPPORTED && rr == nullptr, "unknown provider -> UNSUPPORTED");
}

/* Same fake files, CPU provider: the failure is the invalid model, not the
 * provider. CPU (the default) never yields UNSUPPORTED. */
static void test_cpu_is_not_unsupported(const fs::path& dir) {
    const std::string path = dir.string();

    lembed_text_options_t topts = lembed_text_options_default();
    topts.dim = 8;
    lembed_text_embedding_t* text = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_path(path.c_str(), &topts, &text);
    ASSERT(s != LEMBED_OK && s != LEMBED_ERROR_UNSUPPORTED, "text CPU fake model: not UNSUPPORTED");

    lembed_reranker_options_t ropts = lembed_reranker_options_default();
    lembed_reranker_t* rr = nullptr;
    s = lembed_reranker_create_from_path(path.c_str(), &ropts, &rr);
    ASSERT(s != LEMBED_OK && s != LEMBED_ERROR_UNSUPPORTED,
           "reranker CPU fake model: not UNSUPPORTED");
}

/* The text `backend` option is honoured like the reranker's. Hermetic: nothing
 * is downloaded (offline + empty cache dir, or fake files). */
static void test_text_backend_option(const fs::path& dir) {
    const std::string path = dir.string();
    const std::string gguf = (dir / "nothing.gguf").string();

    /* ONNX forced: a .gguf path is refused, not silently run by llama.cpp. */
    lembed_text_options_t o = lembed_text_options_default();
    o.backend = LEMBED_BACKEND_ONNX;
    o.dim = 8;
    lembed_text_embedding_t* t = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_path(gguf.c_str(), &o, &t);
    ASSERT(s == LEMBED_ERROR_UNSUPPORTED && t == nullptr, "text ONNX + .gguf path -> UNSUPPORTED");
    ASSERT(error_mentions(".gguf"), "text ONNX + .gguf: error names the .gguf file");

    /* llama.cpp forced: a model directory is refused too. */
    o.backend = LEMBED_BACKEND_LLAMACPP;
    s = lembed_text_embedding_create_from_path(path.c_str(), &o, &t);
    ASSERT(s == LEMBED_ERROR_UNSUPPORTED && t == nullptr, "text LLAMACPP + directory -> UNSUPPORTED");

    /* AUTO unchanged: the .gguf path goes to llama.cpp (the file is missing, so
     * a llama error, not a refusal); the fake directory goes to ONNX. */
    o.backend = LEMBED_BACKEND_AUTO;
    s = lembed_text_embedding_create_from_path(gguf.c_str(), &o, &t);
    ASSERT(s != LEMBED_OK && s != LEMBED_ERROR_UNSUPPORTED, "text AUTO + .gguf: routed to llama.cpp");
    s = lembed_text_embedding_create_from_path(path.c_str(), &o, &t);
    ASSERT(s != LEMBED_OK && s != LEMBED_ERROR_UNSUPPORTED, "text AUTO + fake dir: routed to ONNX");
    o.backend = LEMBED_BACKEND_ONNX;
    s = lembed_text_embedding_create_from_path(path.c_str(), &o, &t);
    ASSERT(s != LEMBED_OK && s != LEMBED_ERROR_UNSUPPORTED, "text ONNX + fake dir: routed to ONNX");

    /* LLAMACPP on a registry model with no GGUF equivalent: clear error naming
     * the model, before any download. */
    const std::string empty_cache = (dir / "empty_cache").string();
    lembed_text_options_t r = lembed_text_options_default();
    r.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    r.backend = LEMBED_BACKEND_LLAMACPP;
    r.cache_dir = empty_cache.c_str();
    r.offline = 1;
    r.show_download_progress = 0;
    s = lembed_text_embedding_create(&r, &t);
    ASSERT(s == LEMBED_ERROR_UNSUPPORTED && t == nullptr, "text LLAMACPP, no GGUF equivalent -> UNSUPPORTED");
    ASSERT(error_mentions("bge-small-en-v1.5"), "text LLAMACPP error names the model");

    /* v2 takes the same route. */
    lembed_text_options_v2_t r2;
    std::memset(&r2, 0, sizeof(r2));
    r2.base = r;
    r2.quantization = LEMBED_QUANTIZATION_AUTO;
    s = lembed_text_embedding_create_v2(&r2, &t);
    ASSERT(s == LEMBED_ERROR_UNSUPPORTED && t == nullptr, "text v2 LLAMACPP, no GGUF equivalent -> UNSUPPORTED");

    /* AUTO, ONNX files missing and no GGUF equivalent: the ONNX download error
     * is the one reported (no behaviour change for models without GGUF). */
    r.backend = LEMBED_BACKEND_AUTO;
    s = lembed_text_embedding_create(&r, &t);
    ASSERT(s == LEMBED_ERROR_DOWNLOAD && t == nullptr, "text AUTO offline, no GGUF equivalent -> DOWNLOAD");

    /* LLAMACPP on a model that has a GGUF equivalent, offline, nothing cached:
     * the failure is the missing GGUF file, never UNSUPPORTED. */
    r.model = LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_XS;
    r.backend = LEMBED_BACKEND_LLAMACPP;
    s = lembed_text_embedding_create(&r, &t);
    ASSERT(s != LEMBED_OK && s != LEMBED_ERROR_UNSUPPORTED && t == nullptr,
           "text LLAMACPP with GGUF equivalent: resolved, fails only on the cache");
}

/* ---- Cache part: offline, skipped when the model is absent ---------------- */

static void test_cached_onnx_text() {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.offline = 1;
    opts.show_download_progress = 0;
    lembed_text_embedding_t* ctx = nullptr;
    if (lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
        printf("SKIP: bge-small-en-v1.5 not cached\n");
        return;
    }
    ASSERT(streq(lembed_text_embedding_backend_name(ctx), "onnxruntime"),
           "ONNX text model reports onnxruntime");
    ASSERT(streq(lembed_text_embedding_effective_provider(ctx), "CPU"),
           "ONNX text model on CPU reports CPU");
    lembed_text_embedding_free(ctx);
}

static void test_cached_gguf_text() {
    const lembed_gguf_model_info_t* info = lembed_find_gguf_model("Snowflake-XS-Q4");
    ASSERT(info != nullptr, "Snowflake-XS-Q4 is in the GGUF registry");
    if (!info) return;
    lembed_text_options_t opts = lembed_text_options_default();
    opts.offline = 1;
    opts.show_download_progress = 0;
    char* path = nullptr;
    if (lembed_resolve_gguf_path(info->gguf_url, nullptr, 1, &path) != LEMBED_OK || !path) {
        printf("SKIP: Snowflake-XS GGUF not cached\n");
        return;
    }
    lembed_text_embedding_t* ctx = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(path, &opts, &ctx);
    lembed_free_string(path);
    ASSERT(s == LEMBED_OK, "cached Snowflake-XS GGUF loads");
    if (s != LEMBED_OK) return;
    ASSERT(streq(lembed_text_embedding_backend_name(ctx), "llama.cpp"),
           "GGUF text model reports llama.cpp");
    ASSERT(streq(lembed_text_embedding_effective_provider(ctx), "CPU"),
           "GGUF text model without GPU layers reports CPU");
    lembed_text_embedding_free(ctx);
}

static void test_cached_gguf_reranker() {
    const lembed_gguf_model_info_t* info = lembed_find_gguf_model("Jina-Reranker-v1-Turbo-En");
    ASSERT(info != nullptr, "Jina reranker is in the GGUF registry");
    if (!info) return;
    char* path = nullptr;
    if (lembed_resolve_gguf_path(info->gguf_url, nullptr, 1, &path) != LEMBED_OK || !path) {
        printf("SKIP: Jina reranker GGUF not cached\n");
        return;
    }
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.offline = 1;
    opts.show_download_progress = 0;
    lembed_reranker_t* ctx = nullptr;
    lembed_status_t s = lembed_reranker_create_from_path(path, &opts, &ctx);
    lembed_free_string(path);
    if (s != LEMBED_OK) fprintf(stderr, "  reranker load: %d %s\n", (int)s, lembed_last_error());
    ASSERT(s == LEMBED_OK, "cached Jina reranker GGUF loads");
    if (s != LEMBED_OK) return;
    ASSERT(streq(lembed_reranker_backend_name(ctx), "llama.cpp"),
           "GGUF reranker reports llama.cpp");
    ASSERT(streq(lembed_reranker_effective_provider(ctx), "CPU"),
           "GGUF reranker reports CPU");
    const lembed_model_desc_v2_t* d2 = lembed_reranker_desc_v2(ctx);
    ASSERT(d2 && d2->base.name && d2->base.name[0], "GGUF reranker desc_v2 carries the name");
    lembed_reranker_free(ctx);
}

/* Registry-wide: every ONNX registry reranker that has a GGUF counterpart
 * (same HuggingFace model code) resolves to exactly that entry, and one without
 * counterpart resolves to nothing. No network. The pre-fix lookup
 * (display-name substring) is reported for comparison. */
static void test_gguf_lookup_rerankers() {
    const lembed_model_info_t* models = nullptr;
    int n = 0;
    ASSERT(lembed_list_reranker_models(&models, &n) == LEMBED_OK && n > 0, "reranker registry lists");
    const lembed_gguf_model_info_t* ggufs = nullptr;
    int gn = 0;
    ASSERT(lembed_list_gguf_models(&ggufs, &gn) == LEMBED_OK && gn > 0, "GGUF registry lists");
    if (!models || !ggufs) return;
    int with_counterpart = 0;
    for (int i = 0; i < n; i++) {
        const lembed_gguf_model_info_t* expected = nullptr;
        for (int j = 0; j < gn && !expected; j++) {
            if (lembed__gguf_ieq(ggufs[j].model_code, models[i].model_name) ||
                lembed__gguf_ieq(ggufs[j].model_code, models[i].model_code)) {
                expected = &ggufs[j];
            }
        }
        const lembed_gguf_model_info_t* got =
            lembed__gguf_for_model(models[i].model_name, models[i].model_code, 0);
        const lembed_gguf_model_info_t* old_hit = lembed_find_gguf_model(models[i].model_name);
        if (!old_hit) old_hit = lembed_find_gguf_model(models[i].model_code);
        printf("  reranker %-48s new=%-26s old=%s\n", models[i].model_name,
               got ? got->name : "(none)", old_hit ? old_hit->name : "(none)");
        if (expected) {
            with_counterpart++;
            ASSERT(got == expected, "reranker resolves to its GGUF counterpart");
            ASSERT(got && got->gguf_url && got->gguf_url[0], "resolved GGUF has a URL");
        } else {
            ASSERT(got == nullptr, "reranker without GGUF counterpart resolves to nothing");
        }
    }
    ASSERT(with_counterpart >= 3, "at least 3 registry rerankers have a GGUF counterpart");
}

/* Text path keeps its behaviour: a hit never changes the dimension and an
 * exact model_code match wins. */
static void test_gguf_lookup_text() {
    const lembed_model_info_t* models = nullptr;
    int n = 0;
    if (lembed_list_text_models(&models, &n) != LEMBED_OK || !models) return;
    for (int i = 0; i < n; i++) {
        const lembed_gguf_model_info_t* got =
            lembed__gguf_for_model(models[i].model_name, models[i].model_code, models[i].dim);
        if (got && got->dim > 0) ASSERT(got->dim == models[i].dim, "text GGUF lookup keeps the dimension");
    }
    const lembed_gguf_model_info_t* mini =
        lembed__gguf_for_model("sentence-transformers/all-MiniLM-L6-v2", "x", 384);
    ASSERT(mini && streq(mini->model_code, "sentence-transformers/all-MiniLM-L6-v2"),
           "MiniLM resolves by model code");
    ASSERT(lembed__gguf_for_model("sentence-transformers/all-MiniLM-L6-v2", "x", 768) == nullptr,
           "dimension mismatch rejects the entry");
}

int main(void) {
    /* Unbuffered: a crash must not swallow the progress lines (ctest only
     * shows what reached the pipe). */
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    printf("stage: null contexts\n");
    test_null_contexts();

    printf("stage: fake model dir\n");
    fs::path dir = make_fake_model_dir();
    printf("stage: explicit provider unavailable\n");
    test_explicit_provider_unavailable(dir);
    printf("stage: CPU is not unsupported\n");
    test_cpu_is_not_unsupported(dir);
    printf("stage: text backend option\n");
    test_text_backend_option(dir);
    std::error_code ec;
    fs::remove_all(dir, ec);

    printf("stage: cached ONNX text\n");
    test_cached_onnx_text();
    printf("stage: cached GGUF text\n");
    test_cached_gguf_text();
    printf("stage: cached GGUF reranker\n");
    test_cached_gguf_reranker();

    printf("stage: GGUF lookup\n");
    test_gguf_lookup_rerankers();
    test_gguf_lookup_text();

    printf("test_backend_introspection: %d passed, %d failed\n", passes, failures);
    return failures == 0 ? 0 : 1;
}
