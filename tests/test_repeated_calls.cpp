/*
 * libembedding - tests/test_repeated_calls.cpp
 * Memory stability over 10 000 repeated calls (embed, rerank, sparse): the
 * process RSS is sampled while the calls run and its growth is bounded.
 *
 * Why a bound and not "equal": the allocator and ONNX Runtime's arena keep
 * freed memory, so RSS is not exact. The model is warmed up first with every
 * input shape the loop uses, which settles the arena. After that, a real leak
 * grows linearly with the call count, so the bound is chosen from the smallest
 * leak worth catching: one result buffer per call. A dense result here is
 * 2 texts x 384 floats = 3 KB, so 10 000 calls leak 30 MB if the buffer is
 * never freed; a rerank result (2 x 8 B) leaks only 0.16 MB that way, but the
 * per-call tokenizer batches and cache keys it also allocates are kilobytes.
 * The threshold is 16 MB of growth between the end of warm-up and the end of
 * the run: above allocator noise (measured: 0 to 1 MB over 200 calls), and
 * below a leaked dense result buffer (30 MB) by a factor of 1.9. The reranker
 * is the weaker check; its bound catches leaks of about 1.6 KB per call and up.
 *
 * It is slow (minutes), so it is not part of the default suite: build it with
 * -DLIBEMBEDDING_STABILITY_TESTS=ON; .github/workflows/tsan.yml runs it.
 * Dense and rerank run 10 000 calls each, the cached rerank 2 000 and the
 * sparse model 500 (its 500 MB model makes each call dear). A smaller
 * LEMBED_STABILITY_ITERS gives a smoke run: the threshold scales with the call
 * count (floor 4 MB). Models are loaded from the cache and
 * the test skips when one is missing (LEMBED_TEST_ALLOW_DOWNLOAD, see
 * conc_support.h).
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

#include "conc_support.h"

#include <algorithm>
#include <cstdint>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

/* Resident set size in bytes, 0 when unavailable. */
static size_t current_rss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.WorkingSetSize;
    return 0;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long total = 0, resident = 0;
    const int n = std::fscanf(f, "%lu %lu", &total, &resident);
    std::fclose(f);
    return n == 2 ? (size_t)resident * 4096u : 0;
#endif
}

static const char* const kTexts[] = {
    "A short one.",
    "Stability tests call the same API many times and watch the resident memory, "
    "because a leak of a few kilobytes per call is invisible in any single call.",
};
static const int kNumTexts = (int)(sizeof(kTexts) / sizeof(kTexts[0]));

/* Calls `one_call` iters times, sampling RSS every iters/20 calls. Growth is
 * measured between the end of warm-up and the median of the last 3 samples
 * (a single sample can sit on an allocator trim or a page-in). */
template <typename Fn>
static void check_bounded(const char* label, int warmup, int iters, Fn one_call) {
    for (int i = 0; i < warmup; i++) one_call();
    const size_t base = current_rss();
    if (base == 0) {
        conc_skip(label, "RSS not available");
        return;
    }
    std::vector<size_t> samples;
    const int step = std::max(1, iters / 20);
    for (int i = 1; i <= iters; i++) {
        one_call();
        if (i % step == 0) samples.push_back(current_rss());
    }
    std::vector<size_t> tail(samples.end() - 3, samples.end());
    std::sort(tail.begin(), tail.end());
    const double growth_mb = ((double)tail[1] - (double)base) / (1024.0 * 1024.0);
    /* 16 MB at 10 000 calls, proportional below (floor 4 MB). */
    const double limit_mb = std::max(4.0, 16.0 * (double)iters / 10000.0);
    std::printf("%s: %d calls, RSS %.1f MB -> %.1f MB (growth %.1f MB, limit %.1f MB)\n",
                label, iters, (double)base / 1048576.0, (double)tail[1] / 1048576.0,
                growth_mb, limit_mb);
    CONC_CHECK(growth_mb <= limit_mb, label);
}

static void test_dense(int iters) {
    const char* name = "dense embed, repeated calls";
    lembed_text_options_t opts = lembed_text_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 2;
    opts.offline = conc_may_download("text") ? 0 : 1;
    lembed_text_embedding_t* ctx = nullptr;
    if (lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    check_bounded(name, 200, iters, [&]() {
        lembed_embeddings_t out = {0};
        if (lembed_text_embedding_embed(ctx, kTexts, kNumTexts, 0, &out) != LEMBED_OK) {
            g_failures++;
            return;
        }
        lembed_embeddings_free(&out);
    });
    lembed_text_embedding_free(ctx);
}

static void test_reranker(int iters, bool with_cache) {
    const char* name = with_cache ? "rerank with LRU cache, repeated calls" : "rerank, repeated calls";
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 2;
    opts.offline = conc_may_download("reranker") ? 0 : 1;
    opts.cache_size = with_cache ? 256 : 0;
    lembed_reranker_t* ctx = nullptr;
    if (lembed_reranker_create(&opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    /* With the cache on, vary the query so the cache keeps evicting: a fixed
     * query would turn every call into six hits and stop exercising the model. */
    int n = 0;
    check_bounded(name, 200, iters, [&]() {
        const std::string query = with_cache ? "query number " + std::to_string(n++ % 512)
                                             : std::string("What is the capital of France?");
        lembed_rerank_results_t out = {0};
        if (lembed_reranker_rerank(ctx, query.c_str(), kTexts, kNumTexts, 0, &out) != LEMBED_OK) {
            g_failures++;
            return;
        }
        lembed_rerank_results_free(&out);
    });
    lembed_reranker_free(ctx);
}

static void test_sparse(int iters) {
    const char* name = "sparse embed, repeated calls";
    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.show_download_progress = 0;
    opts.num_threads = 2;
    opts.offline = conc_may_download("sparse") ? 0 : 1;
    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
        conc_skip(name, "model not cached");
        return;
    }
    check_bounded(name, 100, iters, [&]() {
        lembed_sparse_embeddings_t out = {0};
        if (lembed_sparse_text_embedding_embed(ctx, kTexts, kNumTexts, 0, &opts, &out) != LEMBED_OK) {
            g_failures++;
            return;
        }
        lembed_sparse_embeddings_free(&out);
    });
    lembed_sparse_text_embedding_free(ctx);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0); /* progress stays visible in CI logs */
    const int iters = conc_env_int("LEMBED_STABILITY_ITERS", 10000);
    test_dense(iters);
    test_reranker(iters, false);
    test_reranker(std::max(1, iters / 5), true);
    /* The sparse ONNX model is 500 MB: only when cached or allowed. */
    test_sparse(std::max(1, iters / 20));
    std::printf("test_repeated_calls: %d checks, %d failed, %d skipped\n",
                g_checks.load(), g_failures.load(), g_skipped.load());
    return g_failures.load() == 0 ? 0 : 1;
}
