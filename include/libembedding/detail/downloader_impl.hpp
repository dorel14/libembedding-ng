/*
 * libembedding - detail/downloader_impl.hpp
 * Model downloading from HuggingFace Hub using libcurl
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_DOWNLOADER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_DOWNLOADER_IMPL_HPP

#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>

namespace lembed { namespace detail {

/* Get default cache directory (works in both download and no-download modes) */
inline std::string get_cache_dir(const char* override_dir = nullptr) {
    if (override_dir && override_dir[0]) return override_dir;

    const char* env = std::getenv("LIBEMBEDDING_CACHE_DIR");
    if (env && env[0]) return env;

    env = std::getenv("FASTEMBED_CACHE_DIR");
    if (env && env[0]) return env;

    /* Default: ~/.cache/libembedding */
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (!home) home = "/tmp";

    return std::string(home) + "/.cache/libembedding";
}

/* Check if file exists */
inline bool file_exists(const std::string& path) {
    std::error_code ec;
    bool exists = std::filesystem::exists(path, ec);
    bool regular = std::filesystem::is_regular_file(path, ec);
    return exists && regular;
}

/* Read an entire file into a string. Throws std::runtime_error on failure. */
inline std::string read_file_to_string(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::string blob;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        blob.append(buf, n);
    }
    fclose(f);
    return blob;
}

/* Recursively create directories */
inline bool mkdirs(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec && std::filesystem::is_directory(path, ec) && !ec;
}

/* Sanitize repo name for directory */
inline std::string repo_to_dirname(const std::string& repo_code) {
    std::string repo_dir = repo_code;
    for (auto& c : repo_dir) {
        if (c == '/') c = '-';
    }
    return repo_dir;
}

#ifndef LIBEMBEDDING_NO_DOWNLOAD

#include <curl/curl.h>

/* Correction pour le support de S_ISREG sous Windows (MSVC) */
#if defined(_WIN32) || defined(WIN32)
#include <sys/stat.h>
#if !defined(S_ISREG) && defined(_S_IFREG)
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#endif

/* Progress callback type */
typedef void (*download_progress_fn)(float fraction, void* userdata);

/* Get HuggingFace endpoint */
inline std::string get_hf_endpoint() {
    const char* env = std::getenv("HF_ENDPOINT");
    if (env && env[0]) return env;
    return "https://huggingface.co";
}

/* curl write callback */
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    FILE* fp = (FILE*)userp;
    const size_t bytes = size * nmemb;
    return fwrite(contents, 1, bytes, fp);
}

/* curl progress callback */
struct ProgressData {
    download_progress_fn fn;
    void* userdata;
};

static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                              curl_off_t, curl_off_t) {
    ProgressData* pd = (ProgressData*)clientp;
    if (pd && pd->fn && dltotal > 0) {
        float frac = (float)dlnow / (float)dltotal;
        pd->fn(frac, pd->userdata);
    }
    return 0;
}

/* Download a single file from HuggingFace */
inline bool download_hf_file(const std::string& repo, const std::string& filename,
                              const std::string& dest_path,
                              download_progress_fn progress_fn = nullptr,
                              void* progress_userdata = nullptr) {
    static std::atomic<unsigned long long> temp_sequence(0);
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    const auto sequence = temp_sequence.fetch_add(1, std::memory_order_relaxed);

    /* Construct URL */
    std::string url = get_hf_endpoint() + "/" + repo + "/resolve/main/" + filename;

    /* Ensure destination directory exists */
    const std::string parent = std::filesystem::path(dest_path).parent_path().string();
    if (!parent.empty()) {
        if (!mkdirs(parent)) return false;
    }

    /* Open temp file */
    std::string tmp_path = dest_path + ".tmp-" + std::to_string(tick) + "-" +
                           std::to_string(thread_id) + "-" +
                           std::to_string(sequence);
    FILE* fp = fopen(tmp_path.c_str(), "wb");
    if (!fp) return false;

    CURL* curl = curl_easy_init();
    if (!curl) {
        fclose(fp);
        remove(tmp_path.c_str());
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libembedding/" LIBEMBEDDING_VERSION_STRING);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    ProgressData pd = { progress_fn, progress_userdata };
    if (progress_fn) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &pd);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);

    if (res != CURLE_OK) {
        remove(tmp_path.c_str());
        return false;
    }

    /* Publish atomically; another downloader may have completed this file. */
    if (rename(tmp_path.c_str(), dest_path.c_str()) != 0) {
        if (file_exists(dest_path)) {
            remove(tmp_path.c_str());
            return true;
        }
        remove(tmp_path.c_str());
        return false;
    }

    return true;
}

/* Check if a filename is an optional tokenizer/config file */
inline bool is_optional_file(const std::string& filename) {
    return (filename == "tokenizer.json" ||
            filename == "tokenizer_config.json" ||
            filename == "special_tokens_map.json" ||
            filename == "config.json");
}

/*
 * Ensure all model files are present in cache.
 * Returns the cache directory path containing the model files.
 */
inline std::string ensure_model(const std::string& repo_code,
                                const std::string& model_file,
                                const char* const* additional_files,
                                const std::string& cache_dir,
                                bool show_progress,
                                bool offline = false) {
    /* Build model directory: cache_dir/models--repo_with_dashes/ */
    std::string repo_dir = repo_to_dirname(repo_code);
    std::string model_dir = cache_dir + "/models--" + repo_dir;
    if (!mkdirs(model_dir)) {
        throw std::runtime_error("Cannot create model cache directory: " + model_dir);
    }

    /* Files to download */
    std::vector<std::string> files;
    files.push_back(model_file);
    files.push_back("tokenizer.json");
    files.push_back("config.json");
    files.push_back("tokenizer_config.json");
    files.push_back("special_tokens_map.json");

    /* Additional files (ONNX data, etc.) */
    if (additional_files) {
        for (int i = 0; additional_files[i]; i++) {
            files.push_back(additional_files[i]);
        }
    }

    /* Default progress callback: print to stderr */
    download_progress_fn pfn = nullptr;
    if (show_progress) {
        pfn = [](float frac, void*) {
            fprintf(stderr, "\rDownloading... %.1f%%", frac * 100.0f);
            if (frac >= 1.0f) fprintf(stderr, "\n");
        };
    }

    /* Download each file if not cached */
    for (const auto& f : files) {
        std::string dest = model_dir + "/" + f;
        if (file_exists(dest)) continue;

        /* In offline mode, never attempt download */
        if (offline) {
            if (!is_optional_file(f)) {
                throw std::runtime_error(
                    "Model file not in cache (offline mode): " + dest);
            }
            continue;
        }

        /* tokenizer/config files are optional */
        bool ok = download_hf_file(repo_code, f, dest, pfn, nullptr);
        if (!ok && !is_optional_file(f)) {
            throw std::runtime_error("Failed to download: " + repo_code + "/" + f);
        }
    }

    return model_dir;
}

#else /* LIBEMBEDDING_NO_DOWNLOAD */

/*
 * Stub implementations for offline mode (no libcurl).
 * Only supports loading from existing cache.
 */

inline std::string ensure_model(const std::string& repo_code,
                                const std::string& model_file,
                                const char* const* /*additional_files*/,
                                const std::string& cache_dir,
                                bool /*show_progress*/,
                                bool /*offline*/ = false) {
    std::string repo_dir = repo_to_dirname(repo_code);
    std::string model_dir = cache_dir + "/models--" + repo_dir;
    std::string full_path = model_dir + "/" + model_file;
    if (!file_exists(full_path)) {
        throw std::runtime_error("Model not in cache and downloading is disabled: " + full_path);
    }
    return model_dir;
}

#endif /* LIBEMBEDDING_NO_DOWNLOAD */

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_DOWNLOADER_IMPL_HPP */
