/*
 * libembedding - tests/test_concurrency.cpp
 * Multi-thread tests of the thread-safety contract (docs/en/advanced_usage.md,
 * "Thread safety"). Meant to run under ThreadSanitizer (.github/workflows/
 * tsan.yml) but they also assert results, so they are worth running without it.
 *
 * Contract under test:
 *   - one context shared between threads is safe for text, sparse, reranker and
 *     image contexts (calls are serialised per context; image runs in parallel);
 *   - TextEmbeddingPool, the LRU cache, the autotune cache and
 *     lembed_last_error() are safe from any thread.
 * Every shared-context test also checks the results are identical to a
 * single-thread reference and that the stats counters lost no update.
 *
 * Models come from the local cache; a test whose model is missing is skipped
 * (see conc_support.h for LEMBED_TEST_ALLOW_DOWNLOAD). Small models only.
 *
 * Auteur: David Orel
 * Version: 1.14.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>
#include <libembedding/cpp/embedding_pool.hpp>
#include <libembedding/detail/autotune_cache.hpp>

#include "conc_support.h"

#include <cmath>
#include <filesystem>
#include <fstream>

static const int kThreads = 4;
static const int kIterations = 4;
static const float kTol = 1e-5f;

static const char* const kTexts[] = {
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings map text to vectors.",
    "A short one.",
    "Thread safety means a shared object keeps its invariants under concurrent use, "
    "which is what these tests check, one context shared by several threads.",
    "Paris is the capital of France.",
    "Reranking orders documents by relevance to a query.",
};
static const int kNumTexts = (int)(sizeof(kTexts) / sizeof(kTexts[0]));

static std::string g_tmp_root;

static void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static bool close_vec(const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++)
        if (std::fabs(a[i] - b[i]) > kTol) return false;
    return true;
}

/* -------------------------------------------------------------------------
 * Dense text (ONNX and GGUF): one shared context
 * ------------------------------------------------------------------------- */

static void run_text_shared(const char* label, lembed_text_embedding_t* ctx, bool cached) {
    lembed_embeddings_t ref = {0};
    if (lembed_text_embedding_embed(ctx, kTexts, kNumTexts, 0, &ref) != LEMBED_OK) {
        CONC_CHECK(false, "reference embed");
        return;
    }
    const int dim = ref.dim;
    std::atomic<int> mismatches{0};

    conc_run(kThreads, [&](int) {
        for (int i = 0; i < kIterations; i++) {
            lembed_embeddings_t out = {0};
            if (lembed_text_embedding_embed(ctx, kTexts, kNumTexts, 0, &out) != LEMBED_OK) {
                mismatches++;
                continue;
            }
            if (out.dim != dim || !close_vec(out.data, ref.data, dim * kNumTexts)) mismatches++;
            lembed_embeddings_free(&out);
            /* Readers of the counters run next to the writers. */
            lembed_stats_v2_t st;
            lembed_text_embedding_stats_v2(ctx, &st);
            (void)lembed_text_embedding_dim(ctx);
            (void)lembed_text_embedding_model_name(ctx);
        }
    });
    CONC_CHECK(mismatches.load() == 0, label);

    lembed_stats_t st;
    lembed_text_embedding_stats(ctx, &st);
    const uint64_t expected = (uint64_t)kNumTexts * (1 + kThreads * kIterations);
    CONC_CHECK(st.texts_embedded == expected, "text stats lost no update");
    if (cached) {
        lembed_stats_v2_t st2;
        lembed_text_embedding_stats_v2(ctx, &st2);
        CONC_CHECK(st2.cache_hits + st2.cache_misses == expected,
                   "text cache hit+miss counters lost no update");
    }
    lembed_embeddings_free(&ref);
}

static void test_text_onnx(bool with_cache) {
    const char* name = with_cache ? "text ONNX shared context + LRU cache" : "text ONNX shared context";
    lembed_text_options_t opts = lembed_text_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 1;
    opts.offline = conc_may_download("text") ? 0 : 1;
    opts.cache_size = with_cache ? 64 : 0;
    lembed_text_embedding_t* ctx = nullptr;
    if (lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    run_text_shared(name, ctx, with_cache);
    lembed_text_embedding_free(ctx);
}

static void test_text_gguf() {
    const char* name = "text GGUF shared context";
    char* path = nullptr;
    if (lembed_resolve_gguf_path("MiniLM-L6-Q4", nullptr,
                                 conc_may_download("text_gguf") ? 0 : 1, &path) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    lembed_text_options_t opts = lembed_text_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 1;
    opts.offline = 1;
    lembed_text_embedding_t* ctx = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(path, &opts, &ctx);
    lembed_free_string(path);
    if (s != LEMBED_OK) {
        conc_skip(name, lembed_last_error());
        return;
    }
    run_text_shared(name, ctx, false);
    lembed_text_embedding_free(ctx);
}

/* -------------------------------------------------------------------------
 * Reranker (ONNX, and GGUF when LEMBED_TEST_RERANKER_GGUF names a file)
 * ------------------------------------------------------------------------- */

static void run_reranker_shared(const char* label, lembed_reranker_t* ctx, bool cached) {
    const char* query = "What is the capital of France?";
    lembed_rerank_results_t ref = {0};
    if (lembed_reranker_rerank(ctx, query, kTexts, kNumTexts, 0, &ref) != LEMBED_OK) {
        CONC_CHECK(false, "reference rerank");
        return;
    }
    std::atomic<int> mismatches{0};
    conc_run(kThreads, [&](int) {
        for (int i = 0; i < kIterations; i++) {
            lembed_rerank_results_t out = {0};
            if (lembed_reranker_rerank(ctx, query, kTexts, kNumTexts, 0, &out) != LEMBED_OK ||
                out.count != ref.count) {
                mismatches++;
                lembed_rerank_results_free(&out);
                continue;
            }
            for (int j = 0; j < out.count; j++) {
                if (out.items[j].index != ref.items[j].index ||
                    std::fabs(out.items[j].score - ref.items[j].score) > 1e-4f)
                    mismatches++;
            }
            lembed_rerank_results_free(&out);
            lembed_stats_v2_t st;
            lembed_reranker_stats_v2(ctx, &st);
        }
    });
    CONC_CHECK(mismatches.load() == 0, label);

    lembed_stats_t st;
    lembed_reranker_stats(ctx, &st);
    const uint64_t expected = (uint64_t)kNumTexts * (1 + kThreads * kIterations);
    CONC_CHECK(st.texts_embedded == expected, "reranker stats lost no update");
    if (cached) {
        lembed_stats_v2_t st2;
        lembed_reranker_stats_v2(ctx, &st2);
        CONC_CHECK(st2.cache_hits + st2.cache_misses == expected,
                   "reranker cache hit+miss counters lost no update");
    }
    lembed_rerank_results_free(&ref);
}

static void test_reranker_onnx(bool with_cache) {
    const char* name = with_cache ? "reranker ONNX shared context + cache" : "reranker ONNX shared context";
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 1;
    opts.offline = conc_may_download("reranker") ? 0 : 1;
    opts.cache_size = with_cache ? 64 : 0;
    lembed_reranker_t* ctx = nullptr;
    if (lembed_reranker_create(&opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    run_reranker_shared(name, ctx, with_cache);
    lembed_reranker_free(ctx);
}

static void test_reranker_gguf() {
    const char* name = "reranker GGUF shared context";
    const char* path = std::getenv("LEMBED_TEST_RERANKER_GGUF");
    if (!path || !path[0] || !std::filesystem::exists(path)) {
        conc_skip(name, "set LEMBED_TEST_RERANKER_GGUF to a reranker .gguf");
        return;
    }
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 1;
    opts.offline = 1;
    lembed_reranker_t* ctx = nullptr;
    if (lembed_reranker_create_from_gguf_path(path, &opts, &ctx) != LEMBED_OK) {
        conc_skip(name, lembed_last_error());
        return;
    }
    run_reranker_shared(name, ctx, false);
    lembed_reranker_free(ctx);
}

/* -------------------------------------------------------------------------
 * Sparse (ONNX and GGUF)
 * ------------------------------------------------------------------------- */

static bool same_sparse(const lembed_sparse_embeddings_t& a, const lembed_sparse_embeddings_t& b) {
    if (a.count != b.count) return false;
    for (int i = 0; i < a.count; i++) {
        if (a.items[i].length != b.items[i].length) return false;
        for (int j = 0; j < a.items[i].length; j++) {
            if (a.items[i].indices[j] != b.items[i].indices[j]) return false;
            if (std::fabs(a.items[i].values[j] - b.items[i].values[j]) > 1e-4f) return false;
        }
    }
    return true;
}

static void run_sparse_shared(const char* label, lembed_sparse_embedding_ctx_t* ctx,
                              const lembed_sparse_options_t* opts) {
    lembed_sparse_embeddings_t ref = {0};
    if (lembed_sparse_text_embedding_embed(ctx, kTexts, kNumTexts, 0, opts, &ref) != LEMBED_OK) {
        CONC_CHECK(false, "reference sparse embed");
        return;
    }
    std::atomic<int> mismatches{0};
    conc_run(kThreads, [&](int) {
        for (int i = 0; i < kIterations; i++) {
            lembed_sparse_embeddings_t out = {0};
            if (lembed_sparse_text_embedding_embed(ctx, kTexts, kNumTexts, 0, opts, &out) != LEMBED_OK) {
                mismatches++;
                continue;
            }
            if (!same_sparse(ref, out)) mismatches++;
            lembed_sparse_embeddings_free(&out);
            lembed_stats_v2_t st;
            lembed_sparse_text_embedding_stats_v2(ctx, &st);
        }
    });
    CONC_CHECK(mismatches.load() == 0, label);

    lembed_stats_t st;
    lembed_sparse_text_embedding_stats(ctx, &st);
    CONC_CHECK(st.texts_embedded == (uint64_t)kNumTexts * (1 + kThreads * kIterations),
               "sparse stats lost no update");
    lembed_sparse_embeddings_free(&ref);
}

static void test_sparse_onnx() {
    const char* name = "sparse ONNX shared context";
    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 1;
    opts.offline = conc_may_download("sparse") ? 0 : 1;
    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    run_sparse_shared(name, ctx, &opts);
    lembed_sparse_text_embedding_free(ctx);
}

static void test_sparse_gguf() {
    const char* name = "sparse GGUF shared context";
    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 2;
    opts.workers = 2;
    opts.offline = conc_may_download("sparse_gguf") ? 0 : 1;
    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create_from_gguf_model(
            "cstr/splade-pp-en-v1-GGUF", "splade-pp-en-v1-q8_0.gguf", &opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    run_sparse_shared(name, ctx, &opts);
    lembed_sparse_text_embedding_free(ctx);
}

/* -------------------------------------------------------------------------
 * Image: one shared context, calls run in parallel
 * ------------------------------------------------------------------------- */

static void test_image() {
    const char* name = "image shared context";
#ifdef LIBEMBEDDING_NO_IMAGE
    conc_skip(name, "built without image support");
    return;
#else
    lembed_image_options_t opts = lembed_image_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 1;
    opts.offline = conc_may_download("image") ? 0 : 1;
    lembed_image_embedding_t* ctx = nullptr;
    if (lembed_image_embedding_create(&opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    /* A 16x16 binary PPM, decoded by stb_image. */
    std::string ppm = "P6\n16 16\n255\n";
    for (int i = 0; i < 16 * 16; i++) {
        ppm.push_back((char)(i * 7));
        ppm.push_back((char)(i * 3));
        ppm.push_back((char)(255 - i));
    }
    const unsigned char* data[1] = {(const unsigned char*)ppm.data()};
    const int sizes[1] = {(int)ppm.size()};

    lembed_embeddings_t ref = {0};
    if (lembed_image_embedding_embed_bytes(ctx, data, sizes, 1, 0, &ref) != LEMBED_OK) {
        CONC_CHECK(false, "reference image embed");
        lembed_image_embedding_free(ctx);
        return;
    }
    std::atomic<int> mismatches{0};
    conc_run(kThreads, [&](int) {
        for (int i = 0; i < kIterations; i++) {
            lembed_embeddings_t out = {0};
            if (lembed_image_embedding_embed_bytes(ctx, data, sizes, 1, 0, &out) != LEMBED_OK) {
                mismatches++;
                continue;
            }
            if (out.dim != ref.dim || !close_vec(out.data, ref.data, ref.dim)) mismatches++;
            lembed_embeddings_free(&out);
            lembed_stats_t st;
            lembed_image_embedding_stats(ctx, &st);
        }
    });
    CONC_CHECK(mismatches.load() == 0, name);
    lembed_stats_t st;
    lembed_image_embedding_stats(ctx, &st);
    CONC_CHECK(st.texts_embedded == (uint64_t)(1 + kThreads * kIterations),
               "image stats lost no update");
    lembed_embeddings_free(&ref);
    lembed_image_embedding_free(ctx);
#endif
}

/* -------------------------------------------------------------------------
 * TextEmbeddingPool shared between caller threads
 * ------------------------------------------------------------------------- */

static void test_pool() {
    const char* name = "EmbeddingPool shared between caller threads";
    lembed::PoolOptions po;
    po.model_path = "BAAI/bge-small-en-v1.5";
    po.num_workers = 2;
    po.threads_per_worker = 1;
    po.offline = !conc_may_download("text");
    po.show_download_progress = false;
    std::unique_ptr<lembed::EmbeddingPool> pool;
    try {
        pool.reset(new lembed::EmbeddingPool(po));
    } catch (const std::exception& e) {
        conc_skip(name, e.what());
        return;
    }
    const std::vector<std::string> texts(kTexts, kTexts + kNumTexts);
    const auto ref = pool->embed(texts, 0);
    std::atomic<int> mismatches{0};
    conc_run(kThreads, [&](int) {
        for (int i = 0; i < kIterations; i++) {
            const auto out = pool->embed(texts, 0);
            if (out.size() != ref.size()) { mismatches++; continue; }
            for (size_t j = 0; j < out.size(); j++)
                if (out[j].size() != ref[j].size() ||
                    !close_vec(out[j].data(), ref[j].data(), (int)ref[j].size()))
                    mismatches++;
        }
    });
    CONC_CHECK(mismatches.load() == 0, name);
}

/* -------------------------------------------------------------------------
 * LRU cache (C API): concurrent put / get_copy / clear / size
 * ------------------------------------------------------------------------- */

static void test_lru_cache() {
    lembed_cache_config_t cfg = lembed_cache_config_default();
    cfg.capacity = 32;
    lembed_cache_t* cache = lembed_cache_create(&cfg);
    CONC_CHECK(cache != nullptr, "cache create");
    if (!cache) return;
    const int dim = 8;
    std::atomic<int> torn{0};
    conc_run(kThreads * 2, [&](int t) {
        float vec[8], got[8];
        for (int i = 0; i < 2000; i++) {
            const std::string key = "k" + std::to_string((i + t) % 48);
            const float tag = (float)((i + t) % 48);
            for (int d = 0; d < dim; d++) vec[d] = tag;
            if (t % 4 == 3 && i % 500 == 499) lembed_cache_clear(cache);
            lembed_cache_put(cache, key.c_str(), vec, dim);
            int out_dim = 0;
            if (lembed_cache_get_copy(cache, key.c_str(), got, dim, &out_dim) == 1) {
                /* An entry is always one writer's vector: never a mix. */
                for (int d = 1; d < dim; d++)
                    if (got[d] != got[0]) torn++;
            }
            (void)lembed_cache_size(cache);
        }
    });
    CONC_CHECK(torn.load() == 0, "LRU cache never returns a torn vector");
    CONC_CHECK(lembed_cache_size(cache) <= lembed_cache_capacity(cache), "LRU cache stays within capacity");
    lembed_cache_free(cache);
}

/* -------------------------------------------------------------------------
 * Autotune cache: same entry written by many threads, legacy cache saved from
 * many threads under distinct keys
 * ------------------------------------------------------------------------- */

static void test_autotune_cache() {
    using namespace lembed::detail;
    namespace fs = std::filesystem;
    const fs::path root = fs::path(g_tmp_root) / "autotune";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    /* The tests below point the cache at a scratch directory; the original
     * values are put back at the end so later tests keep finding their models. */
    const char* old_la = std::getenv("LOCALAPPDATA");
    const char* old_home = std::getenv("HOME");
    const std::string saved_la = old_la ? old_la : "";
    const std::string saved_home = old_home ? old_home : "";
    const bool had_la = old_la != nullptr, had_home = old_home != nullptr;
    set_env("LOCALAPPDATA", root.string());
    set_env("HOME", root.string());

    /* 1. One entry, written and read by every thread at once. */
    autotune_cache_identity id;
    id.model = "test/concurrency";
    id.variant = "default";
    id.corpus = "c0";
    id.objective = (int)LEMBED_OBJECTIVE_BALANCED;
    id.mode = (int)LEMBED_AUTOTUNE_QUICK;
    std::atomic<int> bad_reads{0};
    std::atomic<int> failed_writes{0};
    conc_run(kThreads * 2, [&](int t) {
        for (int i = 0; i < 100; i++) {
            int threads = 4, batch = 32;
            std::vector<cache_field> w = {cache_int("threads", threads), cache_int("batch_size", batch)};
            if (!write_cache_entry("conc", id, w)) failed_writes++;
            int r_threads = -1, r_batch = -1;
            std::vector<cache_field> r = {cache_int("threads", r_threads), cache_int("batch_size", r_batch)};
            /* A read may miss (never) or hit: it must not see a torn document. */
            if (read_cache_entry("conc", id, r) && (r_threads != 4 || r_batch != 32)) bad_reads++;
            (void)t;
        }
    });
    CONC_CHECK(failed_writes.load() == 0, "autotune entry writes all succeed under contention");
    CONC_CHECK(bad_reads.load() == 0, "autotune entry is never read torn");

    /* 2. Legacy single-file cache: a read-modify-write per save. */
    lembed_cache_hardware_info_t hw;
    lembed_cache_detect_hardware(&hw);
    lembed_cache_software_info_t sw;
    lembed_cache_detect_software(&sw);
    lembed_tune_cache_clear();
    const int kModels = kThreads * 4;
    conc_run(kThreads, [&](int t) {
        for (int m = t; m < kModels; m += kThreads) {
            lembed_tune_cache_entry_t e;
            std::memset(&e, 0, sizeof(e));
            e.hardware = hw;
            e.software = sw;
            std::snprintf(e.model.model_id, sizeof(e.model.model_id), "conc-model-%d", m);
            e.model.dim = 8;
            std::snprintf(e.backend, sizeof(e.backend), "onnx");
            e.num_configs = 1;
            e.configs[0].num_threads = m + 1;
            lembed_tune_cache_save(&e);
        }
    });
    int found = 0;
    for (int m = 0; m < kModels; m++) {
        lembed_tune_cache_entry_t q;
        std::memset(&q, 0, sizeof(q));
        q.hardware = hw;
        q.software = sw;
        std::snprintf(q.model.model_id, sizeof(q.model.model_id), "conc-model-%d", m);
        q.model.dim = 8;
        std::snprintf(q.backend, sizeof(q.backend), "onnx");
        lembed_tune_cache_entry_t out;
        if (lembed_tune_cache_load(&q.hardware, &q.software, &q.model, q.backend, &out) == LEMBED_OK) found++;
    }
    CONC_CHECK(found == kModels, "legacy tune cache lost no concurrent save");
    lembed_tune_cache_clear();

    /* An unset variable is restored as unset where the platform allows it. */
#ifdef _WIN32
    _putenv_s("LOCALAPPDATA", had_la ? saved_la.c_str() : "");
    _putenv_s("HOME", had_home ? saved_home.c_str() : "");
#else
    if (had_la) setenv("LOCALAPPDATA", saved_la.c_str(), 1); else unsetenv("LOCALAPPDATA");
    if (had_home) setenv("HOME", saved_home.c_str(), 1); else unsetenv("HOME");
#endif
}

/* -------------------------------------------------------------------------
 * Downloader: concurrent resolution of an already cached file (no network)
 * ------------------------------------------------------------------------- */

static void test_downloader_resolve() {
    const char* name = "downloader resolve from several threads";
    char* probe = nullptr;
    if (lembed_resolve_gguf_path("MiniLM-L6-Q4", nullptr,
                                 conc_may_download("text_gguf") ? 0 : 1, &probe) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    const std::string expected = probe;
    lembed_free_string(probe);
    std::atomic<int> bad{0};
    conc_run(kThreads, [&](int) {
        for (int i = 0; i < 50; i++) {
            char* p = nullptr;
            if (lembed_resolve_gguf_path("MiniLM-L6-Q4", nullptr, 1, &p) != LEMBED_OK) { bad++; continue; }
            if (expected != p) bad++;
            lembed_free_string(p);
        }
    });
    CONC_CHECK(bad.load() == 0, name);
}

/* -------------------------------------------------------------------------
 * lembed_last_error(): thread-local, one message per thread
 * ------------------------------------------------------------------------- */

static void test_last_error_thread_local() {
    std::atomic<int> bad{0};
    conc_run(kThreads, [&](int t) {
        const std::string path = "/nonexistent/libembedding-conc-" + std::to_string(t) + "-dir";
        for (int i = 0; i < 50; i++) {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.show_download_progress = 0;
            opts.offline = 1;
            lembed_text_embedding_t* ctx = nullptr;
            if (lembed_text_embedding_create_from_path(path.c_str(), &opts, &ctx) == LEMBED_OK) {
                lembed_text_embedding_free(ctx);
                bad++;
                continue;
            }
            /* The message belongs to this thread and mentions this thread's path
             * or, when the loader does not echo the path, is at least not empty
             * and not another thread's. */
            const std::string msg = lembed_last_error();
            if (msg.empty()) {
                bad++;
                std::fprintf(stderr, "last_error empty after a failed create (thread %d)\n", t);
            }
            for (int other = 0; other < kThreads; other++) {
                if (other == t) continue;
                if (msg.find("conc-" + std::to_string(other) + "-dir") != std::string::npos) bad++;
            }
        }
    });
    CONC_CHECK(bad.load() == 0, "lembed_last_error() is per thread");

    /* A thread that never failed sees an empty message, whatever the others did. */
    std::string seen = "x";
    std::thread([&]() { seen = lembed_last_error(); }).join();
    CONC_CHECK(seen.empty(), "a fresh thread starts with an empty lembed_last_error()");
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0); /* progress stays visible if a test hangs */
    g_tmp_root = (std::filesystem::temp_directory_path() / "lembed_test_concurrency").string();
    std::error_code ec;
    std::filesystem::create_directories(g_tmp_root, ec);

#define RUN(call) do { std::printf("RUN  %s\n", #call); call; } while (0)
    RUN(test_lru_cache());
    RUN(test_autotune_cache());
    RUN(test_last_error_thread_local());
    /* Model tests load real models, which a Debug build makes slow (minutes),
     * so they run only on request: LEMBED_TEST_MODELS=1 (cached models) or
     * LEMBED_TEST_ALLOW_DOWNLOAD (CI). The default suite stays fast. */
#define RUN_MODEL(call) do { \
        if (conc_models_enabled()) RUN(call); \
        else conc_skip(#call, "set LEMBED_TEST_MODELS=1 to run the model tests"); \
    } while (0)
    RUN_MODEL(test_downloader_resolve());
    RUN_MODEL(test_text_onnx(false));
    RUN_MODEL(test_text_onnx(true));
    RUN_MODEL(test_text_gguf());
    RUN_MODEL(test_reranker_onnx(false));
    RUN_MODEL(test_reranker_onnx(true));
    RUN_MODEL(test_reranker_gguf());
    RUN_MODEL(test_sparse_onnx());
    RUN_MODEL(test_sparse_gguf());
    RUN_MODEL(test_image());
    RUN_MODEL(test_pool());
#undef RUN_MODEL
#undef RUN

    std::filesystem::remove_all(g_tmp_root, ec);
    std::printf("test_concurrency: %d checks, %d failed, %d skipped\n",
                g_checks.load(), g_failures.load(), g_skipped.load());
    return g_failures.load() == 0 ? 0 : 1;
}
