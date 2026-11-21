/*
 * libembedding - test_lifecycle_memory
 * Error-path cleanup of every create entry point, and large-batch memory.
 *
 * Two modes, one executable:
 *
 *   test_lifecycle_memory                 error paths (default, fast, no network)
 *   test_lifecycle_memory --large-batch   10 000 texts through bge-small and a
 *                                         bound on the resident-set growth
 *
 * Error paths. Each create entry point (text, sparse ONNX + GGUF, reranker ONNX
 * + GGUF, image; _create, _create_from_path, _create_custom) is made to fail at
 * every stage a test can reach: invalid arguments, unknown model, empty cache
 * offline, missing files, garbage ONNX, garbage tokenizer.json after a session
 * that did load, garbage GGUF. The assertions here are only "it failed, and the
 * out pointer stayed NULL". Whether anything leaked is decided by the sanitizer:
 * the .github/workflows/asan.yml job builds this file with -fsanitize=address
 * and LeakSanitizer, which reports every allocation still reachable-by-nobody
 * at exit, ONNX Runtime and ggml included. Without the sanitizer the test
 * still pins the status codes and the NULL out pointer.
 *
 * The session that does load is a hand-written 100-byte ONNX graph (one
 * Identity node), so the "session loaded, tokenizer refused" stage is reachable
 * without any downloaded model.
 *
 * Large batch. Needs bge-small in the local cache and skips (exit 77) without
 * it. No timing: only the result shape and the growth of the resident set.
 *
 * Auteur: David Orel
 * Version: 1.14.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#include <stdlib.h>
#endif
#include <libembedding/libembedding.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <process.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; \
        fprintf(stderr, "FAIL (line %d): %s | last error: %s\n", __LINE__, msg, \
                lembed_last_error()); } \
} while (0)

/* ------------------------------------------------------------------------- */
/* Files                                                                      */
/* ------------------------------------------------------------------------- */

static void write_file(const fs::path& p, const std::string& bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), (std::streamsize)bytes.size());
}

/* Hand-encoded protobuf: ModelProto { ir_version 8, opset 13, graph with one
 * Identity node from int64 "input_ids" to "sentence_embedding" }. ONNX Runtime
 * loads it; it is never run. */
static std::string pb_len(unsigned tag, const std::string& payload) {
    std::string s;
    s.push_back((char)((tag << 3) | 2));
    size_t n = payload.size();
    while (n >= 0x80) { s.push_back((char)((n & 0x7F) | 0x80)); n >>= 7; }
    s.push_back((char)n);
    return s + payload;
}
static std::string pb_varint(unsigned tag, unsigned v) {
    std::string s;
    s.push_back((char)(tag << 3));
    s.push_back((char)v);
    return s;
}
static std::string pb_value_info(const char* name) {
    std::string dim = pb_len(2, "b");                       /* dim_param */
    std::string shape = pb_len(1, dim);                     /* TensorShapeProto.dim */
    std::string tensor = pb_varint(1, 7) + pb_len(2, shape); /* int64, shape */
    std::string type = pb_len(1, tensor);                   /* TypeProto.tensor_type */
    return pb_len(1, name) + pb_len(2, type);
}
static std::string tiny_onnx() {
    std::string node = pb_len(1, "input_ids") + pb_len(2, "sentence_embedding") +
                       pb_len(4, "Identity");
    std::string graph = pb_len(1, node) + pb_len(2, "g") +
                        pb_len(11, pb_value_info("input_ids")) +
                        pb_len(12, pb_value_info("sentence_embedding"));
    return pb_varint(1, 8) + pb_len(7, graph) + pb_len(8, pb_varint(2, 13));
}

/* 7-bit noise only: vendored ggml (gguf.cpp:478-481) calls isprint() on a
 * signed char of the magic, which trips the Debug UCRT assertion
 * "c >= -1 && c <= 255" for high-bit bytes. Harmless in Release and outside
 * this test's scope (third_party), so the fixture stays ASCII. */
static std::string garbage_bytes(size_t n) {
    std::string s(n, '\0');
    unsigned x = 0x9E3779B9u;
    for (size_t i = 0; i < n; i++) { x = x * 1664525u + 1013904223u; s[i] = (char)((x >> 24) & 0x7F); }
    return s;
}

/* "GGUF", version 3, zero tensors, zero key/values: a header and nothing else. */
static std::string truncated_gguf() {
    std::string s = "GGUF";
    s.append("\x03\x00\x00\x00", 4);
    s.append(16, '\0');
    return s;
}

/* ------------------------------------------------------------------------- */
/* Model directories                                                          */
/* ------------------------------------------------------------------------- */

struct Dirs {
    fs::path root;
    fs::path missing, empty_cache;
    fs::path garbage_onnx;      /* model.onnx = noise, tokenizer.json = "{}" */
    fs::path tiny_no_tok;       /* valid ONNX, no tokenizer.json */
    fs::path tiny_bad_tok;      /* valid ONNX, tokenizer.json = not JSON */
    fs::path tiny_empty_tok;    /* valid ONNX, tokenizer.json empty */
    fs::path tiny_empty_obj;    /* valid ONNX, tokenizer.json = "{}" */
    fs::path tiny_no_dim;       /* valid ONNX, no config.json */
    fs::path tiny_ok_dim;       /* valid ONNX + dim, for the image positive path */
    fs::path gguf_garbage, gguf_truncated, gguf_empty, gguf_missing;
};

static std::string str(const fs::path& p) { return p.generic_string(); }

static Dirs make_dirs() {
    Dirs d;
    d.root = fs::temp_directory_path() /
             ("lembed_t35_" + std::to_string((long long)
#ifdef _WIN32
             _getpid()
#else
             getpid()
#endif
             ));
    fs::remove_all(d.root);
    fs::create_directories(d.root);
    const std::string onnx = tiny_onnx();

    d.missing = d.root / "does_not_exist";
    d.empty_cache = d.root / "empty_cache";
    fs::create_directories(d.empty_cache);

    d.garbage_onnx = d.root / "garbage_onnx";
    write_file(d.garbage_onnx / "model.onnx", garbage_bytes(4096));
    write_file(d.garbage_onnx / "tokenizer.json", "{}");

    d.tiny_no_tok = d.root / "tiny_no_tok";
    write_file(d.tiny_no_tok / "model.onnx", onnx);

    d.tiny_bad_tok = d.root / "tiny_bad_tok";
    write_file(d.tiny_bad_tok / "model.onnx", onnx);
    write_file(d.tiny_bad_tok / "tokenizer.json", "this is { not json");

    d.tiny_empty_tok = d.root / "tiny_empty_tok";
    write_file(d.tiny_empty_tok / "model.onnx", onnx);
    write_file(d.tiny_empty_tok / "tokenizer.json", "");

    d.tiny_empty_obj = d.root / "tiny_empty_obj";
    write_file(d.tiny_empty_obj / "model.onnx", onnx);
    write_file(d.tiny_empty_obj / "tokenizer.json", "{}");

    d.tiny_no_dim = d.tiny_empty_obj;
    d.tiny_ok_dim = d.root / "tiny_ok_dim";
    write_file(d.tiny_ok_dim / "model.onnx", onnx);
    write_file(d.tiny_ok_dim / "config.json", "{\"hidden_size\": 8}");

    d.gguf_garbage = d.root / "garbage.gguf";
    write_file(d.gguf_garbage, garbage_bytes(8192));
    d.gguf_truncated = d.root / "truncated.gguf";
    write_file(d.gguf_truncated, truncated_gguf());
    d.gguf_empty = d.root / "empty.gguf";
    write_file(d.gguf_empty, "");
    d.gguf_missing = d.root / "missing.gguf";
    return d;
}

/* ------------------------------------------------------------------------- */
/* Error-path matrix                                                          */
/* ------------------------------------------------------------------------- */

/* Pointers that must stay NULL on failure. A leftover non-NULL pointer would
 * be freed by a caller that checks the status, so the contract is both. */
template <typename Ctx, typename Free>
static void expect_failed(lembed_status_t s, Ctx* ctx, Free freefn, const char* what) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s fails (status %d)", what, (int)s);
    CHECK(s != LEMBED_OK, buf);
    snprintf(buf, sizeof(buf), "%s leaves out == NULL", what);
    CHECK(ctx == nullptr, buf);
    if (ctx) freefn(ctx);
}

static lembed_text_options_t text_opts() {
    lembed_text_options_t o = lembed_text_options_default();
    o.dim = 8;
    o.num_threads = 1;
    return o;
}
static lembed_sparse_options_t sparse_opts() {
    lembed_sparse_options_t o = lembed_sparse_options_default();
    o.num_threads = 1;
    return o;
}
static lembed_reranker_options_t rerank_opts() {
    lembed_reranker_options_t o = lembed_reranker_options_default();
    o.num_threads = 1;
    return o;
}
static lembed_image_options_t image_opts() {
    lembed_image_options_t o = lembed_image_options_default();
    o.num_threads = 1;
    return o;
}

/* Each macro declares `c` so the call expression can write into &c. */
#define FAIL_TEXT(fn_args, what) do { \
    lembed_text_embedding_t* c = nullptr; \
    lembed_status_t s = fn_args; \
    expect_failed(s, c, lembed_text_embedding_free, what); } while (0)
#define FAIL_SPARSE(fn_args, what) do { \
    lembed_sparse_embedding_ctx_t* c = nullptr; \
    lembed_status_t s = fn_args; \
    expect_failed(s, c, lembed_sparse_text_embedding_free, what); } while (0)
#define FAIL_RERANK(fn_args, what) do { \
    lembed_reranker_t* c = nullptr; \
    lembed_status_t s = fn_args; \
    expect_failed(s, c, lembed_reranker_free, what); } while (0)
#define FAIL_IMAGE(fn_args, what) do { \
    lembed_image_embedding_t* c = nullptr; \
    lembed_status_t s = fn_args; \
    expect_failed(s, c, lembed_image_embedding_free, what); } while (0)

static void test_invalid_arguments(const Dirs& d) {
    printf("[invalid arguments and options]\n");
    lembed_text_options_t t = text_opts();
    lembed_sparse_options_t sp = sparse_opts();
    lembed_reranker_options_t rr = rerank_opts();
    lembed_image_options_t im = image_opts();
    const std::string dir = str(d.tiny_empty_obj);

    /* NULL pieces */
    FAIL_TEXT(lembed_text_embedding_create(nullptr, &c), "text create NULL options");
    FAIL_TEXT(lembed_text_embedding_create_from_path(nullptr, &t, &c), "text from_path NULL path");
    FAIL_TEXT(lembed_text_embedding_create_from_path(dir.c_str(), nullptr, &c), "text from_path NULL options");
    FAIL_SPARSE(lembed_sparse_text_embedding_create(nullptr, &c), "sparse create NULL options");
    FAIL_SPARSE(lembed_sparse_text_embedding_create_from_path("", &sp, &c), "sparse from_path empty path");
    FAIL_RERANK(lembed_reranker_create(nullptr, &c), "reranker create NULL options");
    FAIL_RERANK(lembed_reranker_create_from_path(dir.c_str(), nullptr, &c), "reranker from_path NULL options");
    FAIL_IMAGE(lembed_image_embedding_create(nullptr, &c), "image create NULL options");
    FAIL_IMAGE(lembed_image_embedding_create_from_path("", &im, &c), "image from_path empty path");

    /* Backend outside the enum */
    for (int bad : {-1, 99}) {
        lembed_text_options_t o = t;
        o.backend = (lembed_backend_t)bad;
        FAIL_TEXT(lembed_text_embedding_create(&o, &c), "text create bad backend");
        lembed_reranker_options_t r = rr;
        r.backend = (lembed_backend_t)bad;
        FAIL_RERANK(lembed_reranker_create(&r, &c), "reranker create bad backend");
    }

    /* Model id outside the registry, then a registry model that is not in an
     * empty cache and must not be downloaded (offline). */
    for (int bad : {-1, (int)LEMBED_TEXT_MODEL_COUNT, 9999}) {
        lembed_text_options_t o = t;
        o.model = (lembed_text_model_t)bad;
        o.offline = 1;
        FAIL_TEXT(lembed_text_embedding_create(&o, &c), "text create unknown model");
    }
    for (int bad : {-1, (int)LEMBED_SPARSE_MODEL_COUNT}) {
        lembed_sparse_options_t o = sp;
        o.model = (lembed_sparse_model_t)bad;
        o.offline = 1;
        FAIL_SPARSE(lembed_sparse_text_embedding_create(&o, &c), "sparse create unknown model");
    }
    for (int bad : {-1, (int)LEMBED_RERANKER_MODEL_COUNT}) {
        lembed_reranker_options_t o = rr;
        o.model = (lembed_reranker_model_t)bad;
        o.offline = 1;
        FAIL_RERANK(lembed_reranker_create(&o, &c), "reranker create unknown model (onnx)");
        o.backend = LEMBED_BACKEND_AUTO;
        FAIL_RERANK(lembed_reranker_create(&o, &c), "reranker create unknown model (auto)");
        o.backend = LEMBED_BACKEND_LLAMACPP;
        FAIL_RERANK(lembed_reranker_create(&o, &c), "reranker create unknown model (llama)");
    }
    for (int bad : {-1, (int)LEMBED_IMAGE_MODEL_COUNT}) {
        lembed_image_options_t o = im;
        o.model = (lembed_image_model_t)bad;
        o.offline = 1;
        FAIL_IMAGE(lembed_image_embedding_create(&o, &c), "image create unknown model");
    }

    const std::string cache = str(d.empty_cache);
    {
        lembed_text_options_t o = t;
        o.offline = 1; o.cache_dir = cache.c_str();
        FAIL_TEXT(lembed_text_embedding_create(&o, &c), "text create offline, empty cache");
        lembed_text_options_v2_t o2 = lembed_text_options_default_v2();
        o2.base = o;
        lembed_text_embedding_t* c2 = nullptr;
        expect_failed(lembed_text_embedding_create_v2(&o2, &c2), c2,
                      lembed_text_embedding_free, "text create_v2 offline, empty cache");
    }
    {
        lembed_sparse_options_t o = sp;
        o.offline = 1; o.cache_dir = cache.c_str();
        FAIL_SPARSE(lembed_sparse_text_embedding_create(&o, &c), "sparse create offline, empty cache");
    }
    {
        lembed_reranker_options_t o = rr;
        o.offline = 1; o.cache_dir = cache.c_str();
        FAIL_RERANK(lembed_reranker_create(&o, &c), "reranker create offline, empty cache");
        o.backend = LEMBED_BACKEND_LLAMACPP;
        FAIL_RERANK(lembed_reranker_create(&o, &c), "reranker create (llama) offline, empty cache");
    }
    {
        lembed_image_options_t o = im;
        o.offline = 1; o.cache_dir = cache.c_str();
        FAIL_IMAGE(lembed_image_embedding_create(&o, &c), "image create offline, empty cache");
    }

    /* create_custom */
    {
        lembed_user_defined_model_t m;
        std::memset(&m, 0, sizeof(m));
        FAIL_TEXT(lembed_text_embedding_create_custom(&m, LEMBED_PROVIDER_CPU, 1, &c),
                  "text create_custom empty model");
        const std::string noise = garbage_bytes(2048);
        const unsigned char tok[] = "{}";
        m.onnx_data = (const unsigned char*)noise.data();
        m.onnx_data_size = noise.size();
        m.tokenizer_json = tok;
        m.tokenizer_json_size = 2;
        m.dim = 8;
        FAIL_TEXT(lembed_text_embedding_create_custom(&m, LEMBED_PROVIDER_CPU, 1, &c),
                  "text create_custom garbage ONNX");
        m.onnx_data_size = 0;
        FAIL_TEXT(lembed_text_embedding_create_custom(&m, LEMBED_PROVIDER_CPU, 1, &c),
                  "text create_custom zero-size ONNX");

        /* A session that loads, then a tokenizer that does not. */
        const std::string onnx = tiny_onnx();
        const std::string bad_tok = "this is { not json";
        m.onnx_data = (const unsigned char*)onnx.data();
        m.onnx_data_size = onnx.size();
        m.tokenizer_json = (const unsigned char*)bad_tok.data();
        m.tokenizer_json_size = bad_tok.size();
        FAIL_TEXT(lembed_text_embedding_create_custom(&m, LEMBED_PROVIDER_CPU, 1, &c),
                  "text create_custom valid ONNX, garbage tokenizer");
        m.tokenizer_json_size = 0;
        FAIL_TEXT(lembed_text_embedding_create_custom(&m, LEMBED_PROVIDER_CPU, 1, &c),
                  "text create_custom valid ONNX, empty tokenizer");
    }
}

static void test_missing_and_garbage_files(const Dirs& d) {
    printf("[missing files, garbage ONNX, garbage tokenizer]\n");
    lembed_text_options_t t = text_opts();
    lembed_sparse_options_t sp = sparse_opts();
    lembed_reranker_options_t rr = rerank_opts();
    lembed_image_options_t im = image_opts();

    const fs::path bad_dirs[] = { d.missing, d.garbage_onnx, d.tiny_no_tok,
                                  d.tiny_bad_tok, d.tiny_empty_tok };
    /* A tokenizer.json of "{}" is valid JSON with an empty vocabulary; the
     * loader accepts it, so it is not an error case. */
    const char* names[] = { "missing dir", "garbage ONNX", "ONNX without tokenizer.json",
                            "valid ONNX, garbage tokenizer", "valid ONNX, empty tokenizer" };
    for (int i = 0; i < 5; i++) {
        const std::string p = str(bad_dirs[i]);
        char what[160];
        snprintf(what, sizeof(what), "text from_path: %s", names[i]);
        FAIL_TEXT(lembed_text_embedding_create_from_path(p.c_str(), &t, &c), what);
        snprintf(what, sizeof(what), "sparse from_path: %s", names[i]);
        FAIL_SPARSE(lembed_sparse_text_embedding_create_from_path(p.c_str(), &sp, &c), what);
        snprintf(what, sizeof(what), "reranker from_path: %s", names[i]);
        FAIL_RERANK(lembed_reranker_create_from_path(p.c_str(), &rr, &c), what);
        if (i < 2) { /* the image loader reads no tokenizer */
            lembed_image_options_t o = im;
            o.dim = 8;
            snprintf(what, sizeof(what), "image from_path: %s", names[i]);
            FAIL_IMAGE(lembed_image_embedding_create_from_path(p.c_str(), &o, &c), what);
        }
    }

    /* Local text model with a cache enabled: the LRU cache must not leak when the
     * session refuses the file after it was allocated. */
    {
        lembed_text_options_t o = t;
        o.cache_size = 64;
        const std::string p = str(d.garbage_onnx);
        FAIL_TEXT(lembed_text_embedding_create_from_path(p.c_str(), &o, &c),
                  "text from_path garbage ONNX with LRU cache");
        lembed_reranker_options_t r = rr;
        r.cache_size = 64;
        FAIL_RERANK(lembed_reranker_create_from_path(p.c_str(), &r, &c),
                    "reranker from_path garbage ONNX with LRU cache");
        const std::string q = str(d.tiny_bad_tok);
        FAIL_RERANK(lembed_reranker_create_from_path(q.c_str(), &r, &c),
                    "reranker from_path garbage tokenizer with LRU cache");
    }

    /* No dimension anywhere: refused before any session exists. */
    {
        lembed_text_options_t o = lembed_text_options_default();
        o.dim = 0;
        const std::string p = str(d.tiny_no_dim);
        FAIL_TEXT(lembed_text_embedding_create_from_path(p.c_str(), &o, &c),
                  "text from_path without dimension");
        lembed_image_options_t i = im;
        i.dim = 0;
        FAIL_IMAGE(lembed_image_embedding_create_from_path(p.c_str(), &i, &c),
                   "image from_path without dimension");
    }
}

static void test_garbage_gguf(const Dirs& d) {
    printf("[garbage GGUF]\n");
    lembed_text_options_t t = text_opts();
    lembed_sparse_options_t sp = sparse_opts();
    lembed_reranker_options_t rr = rerank_opts();
    const fs::path files[] = { d.gguf_garbage, d.gguf_truncated, d.gguf_empty, d.gguf_missing };
    const char* names[] = { "noise", "header only", "empty file", "missing file" };
    for (int i = 0; i < 4; i++) {
        const std::string p = str(files[i]);
        char what[160];
        for (int workers = 0; workers < 2; workers++) {
            lembed_text_options_t o = t;
            o.auto_workers = workers;
            snprintf(what, sizeof(what), "text from_path gguf: %s (auto_workers=%d)", names[i], workers);
            FAIL_TEXT(lembed_text_embedding_create_from_path(p.c_str(), &o, &c), what);
            snprintf(what, sizeof(what), "text from_gguf_path: %s (auto_workers=%d)", names[i], workers);
            FAIL_TEXT(lembed_text_embedding_create_from_gguf_path(p.c_str(), &o, &c), what);
        }
        lembed_text_options_t oc = t;
        oc.cache_size = 64;
        snprintf(what, sizeof(what), "text from_gguf_path with cache: %s", names[i]);
        FAIL_TEXT(lembed_text_embedding_create_from_gguf_path(p.c_str(), &oc, &c), what);

        snprintf(what, sizeof(what), "sparse from_path gguf: %s", names[i]);
        FAIL_SPARSE(lembed_sparse_text_embedding_create_from_path(p.c_str(), &sp, &c), what);
        snprintf(what, sizeof(what), "sparse from_gguf_path: %s", names[i]);
        FAIL_SPARSE(lembed_sparse_text_embedding_create_from_gguf_path(p.c_str(), &sp, &c), what);

        snprintf(what, sizeof(what), "reranker from_path gguf: %s", names[i]);
        FAIL_RERANK(lembed_reranker_create_from_path(p.c_str(), &rr, &c), what);
        snprintf(what, sizeof(what), "reranker from_gguf_path: %s", names[i]);
        FAIL_RERANK(lembed_reranker_create_from_gguf_path(p.c_str(), &rr, &c), what);
    }
}

/* The one create that succeeds without a model: build, introspect, free, many
 * times. A leak per cycle adds up under the sanitizer, and a session that is
 * not released would keep growing the resident set. */
static void test_create_free_cycles(const Dirs& d) {
    printf("[create/free cycles]\n");
    lembed_image_options_t im = image_opts();
    im.dim = 8;
    const std::string p = str(d.tiny_ok_dim);
    bool all_ok = true;
    for (int i = 0; i < 25; i++) {
        lembed_image_embedding_t* c = nullptr;
        lembed_status_t s = lembed_image_embedding_create_from_path(p.c_str(), &im, &c);
        if (s != LEMBED_OK || !c) { all_ok = false; break; }
        if (lembed_image_embedding_dim(c) != 8) all_ok = false;
        lembed_image_embedding_free(c);
    }
    CHECK(all_ok, "image create_from_path + free x25 (tiny ONNX)");
    lembed_image_embedding_free(nullptr); /* free(NULL) is a no-op */
    lembed_text_embedding_free(nullptr);
    lembed_sparse_text_embedding_free(nullptr);
    lembed_reranker_free(nullptr);
    CHECK(true, "free(NULL) is a no-op for every context type");
}

/* The same failures many times: a leak of even a few bytes per call is found by
 * LeakSanitizer after one iteration, but a loop makes the growth visible to a
 * human reading a non-sanitized run too. */
static void test_repeated_failures(const Dirs& d) {
    printf("[repeated failures]\n");
    lembed_text_options_t t = text_opts();
    const std::string a = str(d.garbage_onnx);
    const std::string b = str(d.tiny_bad_tok);
    bool ok = true;
    for (int i = 0; i < 50; i++) {
        lembed_text_embedding_t* c = nullptr;
        if (lembed_text_embedding_create_from_path(a.c_str(), &t, &c) == LEMBED_OK) ok = false;
        if (c) { lembed_text_embedding_free(c); ok = false; }
        if (lembed_text_embedding_create_from_path(b.c_str(), &t, &c) == LEMBED_OK) ok = false;
        if (c) { lembed_text_embedding_free(c); ok = false; }
    }
    CHECK(ok, "50 failed text from_path loads, none returns a context");
}

/* ------------------------------------------------------------------------- */
/* Large batch                                                                */
/* ------------------------------------------------------------------------- */

static size_t current_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
    return (size_t)pmc.WorkingSetSize;
#elif defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &n) != KERN_SUCCESS)
        return 0;
    return (size_t)info.resident_size;
#else
    std::ifstream f("/proc/self/statm");
    size_t pages = 0, resident = 0;
    if (!(f >> pages >> resident)) return 0;
    return resident * (size_t)sysconf(_SC_PAGESIZE);
#endif
}

/* Polls the *current* resident set. A high-water mark would also include the
 * model load that precedes the measured call, which is not what is asked. */
class RssSampler {
public:
    RssSampler() : stop_(false), peak_(current_rss_bytes()), th_([this] {
        while (!stop_.load()) {
            size_t v = current_rss_bytes();
            size_t p = peak_.load();
            while (v > p && !peak_.compare_exchange_weak(p, v)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }) {}
    size_t stop() {
        stop_.store(true);
        if (th_.joinable()) th_.join();
        return peak_.load();
    }
    ~RssSampler() { stop(); }
private:
    std::atomic<bool> stop_;
    std::atomic<size_t> peak_;
    std::thread th_;
};

static int run_large_batch() {
    printf("[large batch: 10000 texts]\n");
    lembed_text_options_t o = lembed_text_options_default();
    o.offline = 1; /* never download in a test */
    char* dir = nullptr;
    lembed_status_t s = lembed_ensure_text_model(o.model, o.cache_dir, 0, 1, &dir);
    if (s != LEMBED_OK) {
        printf("SKIP: default text model is not in the local cache\n");
        return 77;
    }
    lembed_free_string(dir);

    lembed_text_embedding_t* ctx = nullptr;
    s = lembed_text_embedding_create(&o, &ctx);
    CHECK(s == LEMBED_OK && ctx, "text model loads from the cache");
    if (s != LEMBED_OK || !ctx) return 1;
    const int dim = lembed_text_embedding_dim(ctx);

    const int N = 10000;
    std::vector<std::string> store;
    store.reserve(N);
    for (int i = 0; i < N; i++) {
        store.push_back("short text number " + std::to_string(i) + " about topic " +
                        std::to_string(i % 97));
    }
    std::vector<const char*> texts(N);
    for (int i = 0; i < N; i++) texts[i] = store[i].c_str();

    /* Warm-up: ORT arena, tokenizer tables and thread pool reach steady state
     * on a small call, so the baseline is not the cold-start footprint. */
    {
        lembed_embeddings_t w;
        std::memset(&w, 0, sizeof(w));
        s = lembed_text_embedding_embed(ctx, texts.data(), 256, 0, &w);
        CHECK(s == LEMBED_OK, "warm-up embed");
        lembed_embeddings_free(&w);
    }

    const size_t output_bytes = (size_t)N * (size_t)dim * sizeof(float);
    const size_t baseline = current_rss_bytes();

    lembed_embeddings_t r;
    std::memset(&r, 0, sizeof(r));
    size_t peak;
    {
        RssSampler sampler;
        s = lembed_text_embedding_embed(ctx, texts.data(), N, 0, &r);
        peak = sampler.stop();
    }
    CHECK(s == LEMBED_OK, "embed 10000 texts");
    CHECK(r.num_embeddings == N, "result has 10000 rows");
    CHECK(r.dim == dim && dim > 0, "result dimension matches the model");

    bool finite = true, nonzero = true;
    if (r.data) {
        for (int i = 0; i < N && finite && nonzero; i++) {
            const float* v = r.data + (size_t)i * dim;
            double n2 = 0;
            for (int j = 0; j < dim; j++) {
                if (!std::isfinite(v[j])) finite = false;
                n2 += (double)v[j] * v[j];
            }
            if (n2 <= 0.0) nonzero = false;
        }
    }
    CHECK(r.data && finite, "every component is finite");
    CHECK(r.data && nonzero, "no row is all zero");

    /* Bound: the caller-visible result is output_bytes. Allow it four times
     * over for the result, the sorted index, the token buffers and the length
     * buckets alive at once, plus a fixed 64 MiB for allocator and ORT arena
     * slack. Independent of the model: it scales with N * dim. */
    const size_t growth = peak > baseline ? peak - baseline : 0;
    const size_t bound = 4 * output_bytes + ((size_t)64 << 20);
    printf("  output %.1f MiB, baseline %.1f MiB, peak growth %.1f MiB, bound %.1f MiB\n",
           output_bytes / 1048576.0, baseline / 1048576.0, growth / 1048576.0,
           bound / 1048576.0);
    CHECK(baseline > 0, "resident set size is readable on this platform");
    CHECK(growth < bound, "peak resident-set growth stays below the bound");

    /* A second identical call after the first result is freed must settle back
     * to the same footprint: a result or batch buffer leaked per call would add
     * output_bytes each time. */
    lembed_embeddings_free(&r);
    const size_t after_first = current_rss_bytes();
    {
        lembed_embeddings_t r2;
        std::memset(&r2, 0, sizeof(r2));
        s = lembed_text_embedding_embed(ctx, texts.data(), N, 0, &r2);
        CHECK(s == LEMBED_OK && r2.num_embeddings == N, "second 10000-text embed");
        lembed_embeddings_free(&r2);
    }
    const size_t after_second = current_rss_bytes();
    const size_t drift = after_second > after_first ? after_second - after_first : 0;
    printf("  drift after a second call: %.1f MiB (limit %.1f MiB)\n",
           drift / 1048576.0, output_bytes / 2 / 1048576.0);
    CHECK(drift < output_bytes / 2, "no per-call growth after the result is freed");

    lembed_text_embedding_free(ctx);
    return g_fail == 0 ? 0 : 1;
}

/* ------------------------------------------------------------------------- */

int main(int argc, char** argv) {
    /* Unbuffered, so the last line survives a crash or a hang in CI logs. */
    setvbuf(stdout, nullptr, _IONBF, 0);
#if defined(_MSC_VER) && defined(_DEBUG)
    /* No modal dialog: CRT assertions go to stderr. */
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG);
#endif
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--large-batch") == 0) {
            int rc = run_large_batch();
            printf("\n%d passed, %d failed\n", g_pass, g_fail);
            return rc;
        }
    }

    Dirs d = make_dirs();
    test_invalid_arguments(d);
    test_missing_and_garbage_files(d);
    test_garbage_gguf(d);
    test_create_free_cycles(d);
    test_repeated_failures(d);
    std::error_code ec;
    fs::remove_all(d.root, ec);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
