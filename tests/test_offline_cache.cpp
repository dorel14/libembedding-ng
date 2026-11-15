/*
 * test_offline_cache.cpp - offline mode never leaves a model folder behind.
 *
 * Hermetic: LIBEMBEDDING_CACHE_DIR points to a temporary directory, no network.
 * Before the fix, ensure_model() created models--<repo> before checking the
 * offline flag, so every uncached model (each candidate of
 * lembed_auto_select_model(), whose benchmark sessions are offline) left an
 * empty folder in the cache.
 *
 * Auteur: David Orel
 * Version: 1.13.0
 *
 * SPDX-License-Identifier: MIT
 */

/* Linux/macOS use the header-only implementation; Windows tests link to the DLL. */
#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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

static void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

/* Number of models--* entries directly under the cache root. */
static int count_model_dirs(const fs::path& root) {
    int n = 0;
    std::error_code ec;
    if (!fs::exists(root, ec)) return 0;
    for (const auto& e : fs::directory_iterator(root, ec)) {
        if (e.path().filename().string().rfind("models--", 0) == 0) n++;
    }
    return n;
}

int main(void) {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path root = fs::temp_directory_path() / ("lembed_offline_" + std::to_string(tick));
    fs::create_directories(root / "home");
    const fs::path cache = root / "models";
    fs::create_directories(cache);
    set_env("LIBEMBEDDING_CACHE_DIR", cache.string());

    /* 1. Offline creation of an uncached model fails and creates no folder. */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;
        opts.offline = 1;
        opts.cache_dir = nullptr; /* default: LIBEMBEDDING_CACHE_DIR */
        lembed_text_embedding_t* ctx = nullptr;
        lembed_status_t s = lembed_text_embedding_create(&opts, &ctx);
        ASSERT(s != LEMBED_OK, "offline create of an uncached model fails");
        ASSERT(ctx == nullptr, "no context returned");
        ASSERT(count_model_dirs(cache) == 0, "offline create leaves no models--* folder");
    }

    /* 2. Explicit cache_dir behaves the same. */
    {
        fs::path other = root / "explicit";
        std::string other_str = other.string();
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
        opts.offline = 1;
        opts.cache_dir = other_str.c_str();
        lembed_text_embedding_t* ctx = nullptr;
        ASSERT(lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK,
               "offline create with explicit cache_dir fails");
        ASSERT(count_model_dirs(other) == 0, "explicit cache_dir: no models--* folder");
    }

    /* 3. Auto model selection on an empty cache: every candidate is skipped
     *    and none leaves a folder. The status is not asserted: a tuning cache
     *    on the machine could in theory make a candidate win.
     *    lembed_auto_select_model() is only built into the shared library, so
     *    the header-only build (Linux/macOS) cannot call it here;
     *    python/tests/test_auto_select_offline.py covers it on every OS. */
#ifdef _WIN32
    {
        lembed_model_selection_t sel;
        (void)lembed_auto_select_model("balanced", &sel);
        ASSERT(count_model_dirs(cache) == 0, "auto_select_model leaves no models--* folder");
    }
#endif

    std::error_code ec;
    fs::remove_all(root, ec);
    printf("test_offline_cache: %d passed, %d failed\n", passes, failures);
    return failures == 0 ? 0 : 1;
}
