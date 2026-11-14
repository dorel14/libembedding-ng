/*
 * test_cache_cleanup.cpp - lembed_cleanup_model_cache[_except] on a fake cache.
 *
 * Hermetic: a temporary directory stands in for the model cache, with model
 * directories laid out as the downloader writes them (models--<org>-<repo>).
 * Checks the counts and byte totals, that dry runs delete nothing, that
 * non-model entries survive, and that a model kept by its usual name (the
 * original model, e.g. BAAI/bge-small-en-v1.5) protects the directories of the
 * repositories actually downloaded for it (Qdrant/..., Xenova/...).
 *
 * No model downloads required.
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

static void write_file(const fs::path& p, size_t bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out << std::string(bytes, 'x');
}

/* Downloaded repositories of BAAI/bge-small-en-v1.5 in the text registry
 * (model_code), plus an unrelated model, a tuning cache file and a non-model
 * directory that cleanup must never touch. */
static fs::path make_cache() {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path root = fs::temp_directory_path() / ("lembed_cleanup_" + std::to_string(tick));
    /* Layout of detail/downloader_impl.hpp: models-- + repo with '/' -> '-'. */
    write_file(root / "models--Qdrant-bge-small-en-v1.5-onnx-Q" / "model.onnx", 100);
    write_file(root / "models--Xenova-bge-small-en-v1.5" / "onnx" / "model.onnx", 50);
    write_file(root / "models--org-other" / "model.onnx", 30);
    write_file(root / "tune_cache.json", 7);
    write_file(root / "notes" / "keep.txt", 3);
    return root;
}

static bool present(const fs::path& root, const char* name) {
    return fs::exists(root / name);
}

int main(void) {
    /* 1. Keep by downloaded repository, dry run: report only. */
    {
        fs::path root = make_cache();
        const char* keep[] = { "Qdrant/bge-small-en-v1.5-onnx-Q", nullptr };
        size_t n = 99; uint64_t bytes = 99;
        lembed_status_t s = lembed_cleanup_model_cache(root.string().c_str(), keep, 1, &n, &bytes);
        ASSERT(s == LEMBED_OK, "dry run returns OK");
        ASSERT(n == 2, "dry run counts the two other model directories");
        ASSERT(bytes == 80, "dry run sums their bytes (50 + 30)");
        ASSERT(present(root, "models--Xenova-bge-small-en-v1.5"), "dry run deletes nothing");
        ASSERT(present(root, "models--org-other"), "dry run deletes nothing (2)");
        fs::remove_all(root);
    }

    /* 2. Keep by the original model name: every repository downloaded for it
     *    is protected, only the unrelated model goes. */
    {
        fs::path root = make_cache();
        const char* keep[] = { "BAAI/bge-small-en-v1.5", nullptr };
        size_t n = 0; uint64_t bytes = 0;
        lembed_status_t s = lembed_cleanup_model_cache(root.string().c_str(), keep, 0, &n, &bytes);
        ASSERT(s == LEMBED_OK, "cleanup returns OK");
        ASSERT(n == 1, "only the unrelated model is deleted");
        ASSERT(bytes == 30, "freed bytes are those of the unrelated model");
        ASSERT(present(root, "models--Qdrant-bge-small-en-v1.5-onnx-Q"), "Qdrant copy kept");
        ASSERT(present(root, "models--Xenova-bge-small-en-v1.5"), "Xenova copy kept");
        ASSERT(!present(root, "models--org-other"), "unrelated model deleted");
        ASSERT(present(root, "tune_cache.json"), "tuning cache file kept");
        ASSERT(present(root, "notes/keep.txt"), "non-model directory kept");
        fs::remove_all(root);
    }

    /* 3. The _except variant, keyed by a downloaded repository. */
    {
        fs::path root = make_cache();
        size_t n = 0; uint64_t bytes = 0;
        lembed_status_t s = lembed_cleanup_model_cache_except(
            root.string().c_str(), "org/other", 0, &n, &bytes);
        ASSERT(s == LEMBED_OK, "_except returns OK");
        ASSERT(n == 2 && bytes == 150, "_except deletes the two bge copies (100 + 50)");
        ASSERT(present(root, "models--org-other"), "_except keeps its model");
        ASSERT(present(root, "tune_cache.json"), "_except keeps non-model files");
        fs::remove_all(root);
    }

    /* 4. Edge cases. */
    {
        const char* keep[] = { nullptr };
        size_t n = 99; uint64_t bytes = 99;
        fs::path missing = fs::temp_directory_path() / "lembed_cleanup_does_not_exist";
        ASSERT(lembed_cleanup_model_cache(missing.string().c_str(), keep, 0, &n, &bytes) == LEMBED_OK,
               "missing cache directory is not an error");
        ASSERT(n == 0 && bytes == 0, "missing cache directory reports zero");
        ASSERT(lembed_cleanup_model_cache(missing.string().c_str(), nullptr, 0, &n, &bytes)
                   == LEMBED_ERROR_INVALID_ARGUMENT,
               "NULL keep list is rejected");
        ASSERT(lembed_cleanup_model_cache_except(missing.string().c_str(), "", 0, &n, &bytes)
                   == LEMBED_ERROR_INVALID_ARGUMENT,
               "empty active model is rejected");
    }

    printf("test_cache_cleanup: %d passed, %d failed\n", passes, failures);
    return failures == 0 ? 0 : 1;
}
