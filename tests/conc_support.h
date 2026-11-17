/*
 * libembedding - tests/conc_support.h
 * Helpers shared by test_concurrency.cpp and test_repeated_calls.cpp: model
 * gating (skip when a model is not cached), a tiny check macro, and a start
 * barrier so that worker threads begin their loop together.
 *
 * Auteur: David Orel
 * Version: 1.14.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LEMBED_TESTS_CONC_SUPPORT_H
#define LEMBED_TESTS_CONC_SUPPORT_H

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/* LEMBED_TEST_ALLOW_DOWNLOAD is a comma-separated list of model tags (text,
 * text_gguf, reranker, reranker_gguf, sparse, sparse_gguf, image) or "all".
 * A model whose tag is listed may be downloaded; any other model is loaded from
 * the cache only (offline), and its tests are skipped when it is not cached.
 * A local run therefore never touches the network. */
static inline bool conc_may_download(const char* tag) {
    const char* env = std::getenv("LEMBED_TEST_ALLOW_DOWNLOAD");
    if (!env || !env[0]) return false;
    if (std::strcmp(env, "all") == 0) return true;
    const std::string list = std::string(",") + env + ",";
    return list.find(std::string(",") + tag + ",") != std::string::npos;
}

/* Model tests are opt-in: LEMBED_TEST_MODELS=1 (use cached models) or any
 * LEMBED_TEST_ALLOW_DOWNLOAD value. */
static inline bool conc_models_enabled() {
    const char* a = std::getenv("LEMBED_TEST_MODELS");
    const char* b = std::getenv("LEMBED_TEST_ALLOW_DOWNLOAD");
    return (a && a[0] && a[0] != '0') || (b && b[0]);
}

static inline int conc_env_int(const char* name, int fallback) {
    const char* env = std::getenv(name);
    if (!env || !env[0]) return fallback;
    const int v = std::atoi(env);
    return v > 0 ? v : fallback;
}

static std::atomic<int> g_checks{0};
static std::atomic<int> g_failures{0};
static std::atomic<int> g_skipped{0};

#define CONC_CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        g_failures++; \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
    } \
} while (0)

static inline void conc_skip(const char* what, const char* why) {
    g_skipped++;
    std::printf("SKIP: %s (%s)\n", what, why);
}

/* Runs fn(thread_index) on n threads released together by a barrier. */
template <typename Fn>
static void conc_run(int n, Fn fn) {
    std::mutex m;
    std::condition_variable cv;
    bool go = false;
    std::vector<std::thread> threads;
    for (int t = 0; t < n; t++) {
        threads.emplace_back([&, t]() {
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [&]() { return go; });
            }
            fn(t);
        });
    }
    {
        std::lock_guard<std::mutex> lock(m);
        go = true;
    }
    cv.notify_all();
    for (auto& th : threads) th.join();
}

#endif /* LEMBED_TESTS_CONC_SUPPORT_H */
