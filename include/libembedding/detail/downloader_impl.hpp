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
#ifdef __APPLE__
#include <TargetConditionals.h>  /* TARGET_OS_IPHONE */
#endif

/* iOS (device and simulator), where the app sandbox and the certificate store
 * differ from every other platform. */
#if defined(__APPLE__) && defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
#define LEMBED_DETAIL_IOS 1
#else
#define LEMBED_DETAIL_IOS 0
#endif

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
    if (!home || !home[0]) home = std::getenv("USERPROFILE");
#if LEMBED_DETAIL_IOS
    /* HOME is the app container, whose root an app may not write to; caches
     * belong in Library/Caches, which the system may purge when space runs
     * low (a purged model is downloaded again). */
    if (home && home[0]) return std::string(home) + "/Library/Caches/libembedding";
#endif
    if (home && home[0]) return std::string(home) + "/.cache/libembedding";

    /* No home directory. Android apps have none and no /tmp either, but the
     * runtime (and Python on Android) points TMPDIR at the app's cache dir. */
    const char* tmp = std::getenv("TMPDIR");
    if (tmp && tmp[0]) return std::string(tmp) + "/libembedding";
#if defined(__ANDROID__)
    return "/data/local/tmp/libembedding";
#else
    return "/tmp/.cache/libembedding";
#endif
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
#ifdef LIBEMBEDDING_EMBEDDED_CA_BUNDLE
/* Generated at configure time from LIBEMBEDDING_CA_BUNDLE_EMBED: defines
 * lembed_embedded_ca_bundle[], the PEM bundle as bytes. */
#include "lembed_embedded_ca_bundle.h"
#endif

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

/* CA certificates for TLS on Linux and Android.
 *
 * libcurl looks for CA certificates at a path fixed when it was built. The
 * Linux wheels bundle the libcurl of their build image: AlmaLinux's points to
 * /etc/pki/tls/certs/ca-bundle.crt, which Debian and Ubuntu do not have, so
 * every HTTPS download failed there (measured on the aarch64 wheel). Instead,
 * honour SSL_CERT_FILE / CURL_CA_BUNDLE, then use the first bundle that exists
 * on this machine. Windows and macOS keep libcurl's own default.
 *
 * iOS exposes no certificate file and its native verifier (SecTrust) is not
 * available with mbedTLS, so the iOS build embeds Mozilla's CA bundle
 * (LIBEMBEDDING_EMBEDDED_CA_BUNDLE, see CMakeLists.txt). SSL_CERT_FILE /
 * CURL_CA_BUNDLE still take precedence there. */
struct CaSource {
    std::string file;   /* for CURLOPT_CAINFO, empty if none */
    std::string dir;    /* for CURLOPT_CAPATH, empty if none */
};

inline CaSource find_ca_source() {
    CaSource ca;
#if defined(__linux__) || defined(__ANDROID__) || LEMBED_DETAIL_IOS
    for (const char* var : {"SSL_CERT_FILE", "CURL_CA_BUNDLE"}) {
        const char* v = std::getenv(var);
        if (v && v[0] && file_exists(v)) {
            ca.file = v;
            return ca;
        }
    }
#endif
#if defined(__linux__) || defined(__ANDROID__)
    static const char* const kBundles[] = {
        "/etc/ssl/certs/ca-certificates.crt",                 /* Debian, Ubuntu, Alpine */
        "/etc/pki/tls/certs/ca-bundle.crt",                   /* RHEL, Fedora, AlmaLinux */
        "/etc/ssl/ca-bundle.pem",                             /* openSUSE */
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  /* RHEL 7+ */
        "/etc/ssl/cert.pem",                                  /* Alpine, others */
    };
    for (const char* path : kBundles) {
        if (file_exists(path)) {
            ca.file = path;
            return ca;
        }
    }
    /* Android ships hashed certificates in a directory, not a bundle file. */
    std::error_code ec;
    if (std::filesystem::is_directory("/system/etc/security/cacerts", ec)) {
        ca.dir = "/system/etc/security/cacerts";
    }
#endif
    return ca;
}

/* Resolved once: the bundle does not move while the process runs. */
inline const CaSource& ca_source() {
    static const CaSource ca = find_ca_source();
    return ca;
}

/* Download a single file from HuggingFace.
 * error_out, when not null, receives libcurl's reason for a failure. */
inline bool download_hf_file(const std::string& repo, const std::string& filename,
                              const std::string& dest_path,
                              download_progress_fn progress_fn = nullptr,
                              void* progress_userdata = nullptr,
                              std::string* error_out = nullptr) {
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

    const CaSource& ca = ca_source();
    if (!ca.file.empty()) curl_easy_setopt(curl, CURLOPT_CAINFO, ca.file.c_str());
    if (!ca.dir.empty()) curl_easy_setopt(curl, CURLOPT_CAPATH, ca.dir.c_str());
#ifdef LIBEMBEDDING_EMBEDDED_CA_BUNDLE
    if (ca.file.empty() && ca.dir.empty()) {
        struct curl_blob blob;
        blob.data = (void*)lembed_embedded_ca_bundle;
        blob.len = sizeof(lembed_embedded_ca_bundle);
        blob.flags = CURL_BLOB_NOCOPY;  /* static storage, outlives the transfer */
        curl_easy_setopt(curl, CURLOPT_CAINFO_BLOB, &blob);
    }
#endif

    char errbuf[CURL_ERROR_SIZE];
    errbuf[0] = '\0';
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);

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
        if (error_out) {
            *error_out = curl_easy_strerror(res);
            if (errbuf[0]) *error_out += std::string(": ") + errbuf;
        }
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
        std::string reason;
        bool ok = download_hf_file(repo_code, f, dest, pfn, nullptr, &reason);
        if (!ok && !is_optional_file(f)) {
            throw std::runtime_error("Failed to download: " + repo_code + "/" + f +
                                     (reason.empty() ? "" : " (" + reason + ")"));
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
