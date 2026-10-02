/*
 * libembedding - detail/autotune_cache.hpp
 * Auto-tuning cache system
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_CACHE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_CACHE_HPP

#include "libembedding/autotuner.h"

/* get_cache_key() reads the ONNX Runtime version so a cache entry cannot
 * outlive the runtime it was measured on. */
#include <onnxruntime_c_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lembed { namespace detail {

/* Get number of logical CPU cores */
inline int cpu_logical_cores() {
    static int cores = []() {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (int)si.dwNumberOfProcessors;
#else
        return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    }();
    return cores;
}

/* Get physical CPU cores */
inline int cpu_physical_cores() {
    int logical = cpu_logical_cores();
    int physical = logical;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (len > 0) {
        auto* buf = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(len);
        if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len)) {
            int n_phys = 0;
            uint8_t* ptr = (uint8_t*)buf;
            uint8_t* end = ptr + len;
            while (ptr < end) {
                auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
                if (info->Relationship == RelationProcessorCore) {
                    n_phys++;
                }
                ptr += info->Size;
            }
            if (n_phys > 0) physical = n_phys;
        }
        free(buf);
    }
#else
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        int max_phys = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "cpu cores", 9) == 0) {
                int n = 0;
                if (sscanf(line, "cpu cores : %d", &n) == 1 && n > max_phys)
                    max_phys = n;
            }
        }
        if (max_phys > 0) physical = max_phys;
        fclose(f);
    }
#endif
    return physical;
}

/* Get CPU brand string */
inline std::string cpu_brand_string() {
    char cpu_brand[64] = "unknown";
#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD size = sizeof(cpu_brand);
        RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
                        (LPBYTE)cpu_brand, &size);
        RegCloseKey(hKey);
    }
#else
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    strncpy(cpu_brand, colon + 2, sizeof(cpu_brand) - 1);
                    cpu_brand[sizeof(cpu_brand) - 1] = '\0';
                    char* nl = strchr(cpu_brand, '\n');
                    if (nl) *nl = '\0';
                }
                break;
            }
        }
        fclose(fp);
    }
#endif
    /* Sanitize */
    for (int i = 0; cpu_brand[i]; i++) {
        char c = cpu_brand[i];
        if (c == ' ' || c == '(' || c == ')' || c == '@' || c == '.' || c == '/')
            cpu_brand[i] = '_';
    }
    return std::string(cpu_brand);
}

/* =========================================================================
 * Cache system for autotune results
 * ========================================================================= */

inline std::string autotune_cache_dir() {
#ifdef _WIN32
    const char* local_appdata = std::getenv("LOCALAPPDATA");
    if (local_appdata) {
        return std::string(local_appdata) + "\\libembedding\\autotune";
    }
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) {
        return std::string(userprofile) + "\\AppData\\Local\\libembedding\\autotune";
    }
#else
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.cache/libembedding/autotune";
    }
#endif
    return "./libembedding_autotune_cache";
}

/* Make a model name safe to embed in a file name.
 * Shared with clear_autotune_cache(), which matches entries on this form. */
inline std::string sanitize_for_key(const char* model_name) {
    std::string safe_model = model_name ? model_name : "";
    for (auto& c : safe_model) {
        if (c == '/' || c == '\\' || c == ':' || c == ' ')
            c = '_';
    }
    return safe_model;
}

/* Build cache key from model_name + hardware + versions */
inline std::string get_cache_key(const char* model_name) {
    std::string brand = cpu_brand_string();
    int logical = cpu_logical_cores();
    int physical = cpu_physical_cores();

    std::string safe_model = sanitize_for_key(model_name);

    /* ORT version (major.minor only) */
    const OrtApiBase* ort_base = OrtGetApiBase();
    const char* ort_ver = ort_base->GetVersionString();
    int ort_major = 0, ort_minor = 0;
    if (ort_ver) {
        sscanf(ort_ver, "%d.%d", &ort_major, &ort_minor);
    }

    std::ostringstream key;
    key << logical << "x" << physical << "_" << brand << "_" << safe_model
        << "_ort" << ort_major << "." << ort_minor
        << "_v" << LIBEMBEDDING_VERSION_MAJOR << "." << LIBEMBEDDING_VERSION_MINOR;
    return key.str();
}

/* Get cache file path for a model.
 * `variant` distinguishes tuning requests that share a model but differ by
 * corpus, objective or mode; pass "" for the single default tuning. */
inline std::string get_cache_path(const char* model_name, const std::string& subdir = "",
                                  const std::string& variant = "") {
    std::string dir = autotune_cache_dir();
    if (!subdir.empty()) dir += "/" + subdir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string key = model_name ? model_name : "";
    if (!variant.empty()) key += "_" + variant;
    return dir + "/" + get_cache_key(key.c_str()) + ".json";
}

/* =========================================================================
 * Statistics
 * ========================================================================= */

/* Linear-interpolation percentile over a pre-sorted vector.
 * Defined for every sample count (0, 1, n) and never reads out of bounds,
 * unlike an index computed as (size_t)(pct * n) which overflows the vector for
 * small n and silently returns the maximum for the rest. */
inline double percentile_sorted(const std::vector<double>& sorted, double pct) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double rank = pct * (double)(sorted.size() - 1);
    double lo_f = std::floor(rank);
    double hi_f = std::ceil(rank);
    if (lo_f == hi_f) return sorted[(size_t)lo_f];
    double frac = rank - lo_f;
    return sorted[(size_t)lo_f] + (sorted[(size_t)hi_f] - sorted[(size_t)lo_f]) * frac;
}

/* =========================================================================
 * Corpus fingerprint
 *
 * A tuning result is only valid for the corpus it was measured on. The
 * fingerprint goes into the cache identity so two corpora can never share an
 * entry, which would otherwise silently return the first corpus' tuning for
 * the second one.
 * ========================================================================= */

inline uint64_t fnv1a_update(uint64_t h, const char* data, size_t len) {
    for (size_t j = 0; j < len; j++) {
        h ^= (uint64_t)(unsigned char)data[j];
        h *= 1099511628211ULL;
    }
    return h;
}

inline std::string hash64_to_hex(uint64_t h) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

inline std::string corpus_fingerprint(const char* const* texts, int n_texts) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n_texts; i++) {
        const char* t = (texts && texts[i]) ? texts[i] : "";
        size_t len = std::strlen(t);
        /* Mix the length first so {"ab","c"} and {"a","bc"} differ. */
        for (int b = 0; b < 8; b++) {
            h ^= (uint64_t)((len >> (8 * b)) & 0xFFu);
            h *= 1099511628211ULL;
        }
        h = fnv1a_update(h, t, len);
    }
    return hash64_to_hex(h);
}

inline std::string corpus_fingerprint(const std::vector<std::string>& corpus) {
    uint64_t h = 1469598103934665603ULL;
    for (const auto& t : corpus) {
        for (int b = 0; b < 8; b++) {
            h ^= (uint64_t)((t.size() >> (8 * b)) & 0xFFu);
            h *= 1099511628211ULL;
        }
        h = fnv1a_update(h, t.data(), t.size());
    }
    return hash64_to_hex(h);
}

/* =========================================================================
 * Cache entry identity
 *
 * Every entry records what it was tuned for. Reading re-checks all of it, so a
 * stale or foreign entry is a miss instead of a wrong answer.
 * ========================================================================= */

struct autotune_cache_identity {
    std::string model;
    std::string variant;   /* "" for the default tuning, "custom" for a corpus */
    std::string corpus;    /* "" or corpus_fingerprint() */
    int objective = -1;    /* -1 when the tuner has no objective */
    int mode = -1;         /* -1 when the tuner has no mode */
};

/* Short, collision-resistant file-name suffix for an identity. Keeps the path
 * well under MAX_PATH on Windows while remaining stable. */
inline std::string cache_variant_key(const char* variant, const std::string& corpus,
                                     int objective, int mode) {
    std::string identity = std::string(variant ? variant : "default") + "|" + corpus +
                           "|" + std::to_string(objective) + "|" + std::to_string(mode);
    return hash64_to_hex(fnv1a_update(1469598103934665603ULL, identity.data(), identity.size()))
        .substr(0, 12);
}

/* =========================================================================
 * Minimal, fault-tolerant JSON accessors
 *
 * The cache format is one "key": value per line. Every accessor returns false
 * on malformed input instead of throwing or yielding a sentinel, so a corrupt
 * file degrades to a cache miss.
 * ========================================================================= */

/* Trim whitespace and quotes from JSON values */
inline void trim_json_value(std::string& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '"' || s.front() == ','))
        s.erase(s.begin());
    while (!s.empty() && (s.back() == ' ' || s.back() == '"' || s.back() == ','))
        s.pop_back();
}

/* Locate the value that follows "key" on a single-line JSON entry. */
inline bool json_value_span(const std::string& line, const char* key,
                            size_t& begin, size_t& end) {
    std::string search = std::string("\"") + key + "\"";
    size_t pos = line.find(search);
    if (pos == std::string::npos) return false;
    size_t colon = line.find(':', pos + search.size());
    if (colon == std::string::npos) return false;
    begin = colon + 1;
    end = line.size();
    return true;
}

inline bool json_string(const std::string& line, const char* key, std::string& out) {
    size_t begin = 0, end = 0;
    if (!json_value_span(line, key, begin, end)) return false;
    size_t open = line.find('"', begin);
    if (open == std::string::npos || open >= end) return false;
    size_t close = line.find('"', open + 1);
    if (close == std::string::npos || close > end) return false;
    out = line.substr(open + 1, close - open - 1);
    return true;
}

inline bool json_number(const std::string& line, const char* key, double& out) {
    size_t begin = 0, end = 0;
    if (!json_value_span(line, key, begin, end)) return false;
    std::string val = line.substr(begin, end - begin);
    trim_json_value(val);
    if (val.empty()) return false;
    try {
        double d = std::stod(val);
        if (!std::isfinite(d)) return false;
        out = d;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

inline bool json_number(const std::string& line, const char* key, int& out) {
    double d = 0.0;
    if (!json_number(line, key, d)) return false;
    if (d < (double)std::numeric_limits<int>::min() ||
        d > (double)std::numeric_limits<int>::max())
        return false;
    out = (int)d;
    return true;
}

inline bool json_number_in_file(const std::string& path, const char* key, double& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (json_number(line, key, out)) return true;
    }
    return false;
}

inline bool json_string_in_file(const std::string& path, const char* key, std::string& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (json_string(line, key, out)) return true;
    }
    return false;
}

/* =========================================================================
 * Atomic file write
 *
 * Truncate-then-write leaves a partial document if the process dies, and a
 * partial document is read back as a half-filled result. Writing to a
 * temporary file and renaming makes the swap indivisible.
 * ========================================================================= */

inline bool write_file_atomic(const std::string& path, const std::string& content) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return false;
        f << content;
        f.flush();
        if (!f.good()) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        /* Some filesystems refuse to replace an existing file; retry once. */
        std::error_code rm;
        std::filesystem::remove(path, rm);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
    if (ec) {
        std::error_code rm;
        std::filesystem::remove(tmp, rm);
        return false;
    }
    return true;
}

/* Write the identity block that opens every cache entry. */
inline void write_cache_identity(std::ostream& os, const autotune_cache_identity& id) {
    os << "  \"model\": \"" << id.model << "\",\n";
    os << "  \"variant\": \"" << id.variant << "\",\n";
    os << "  \"corpus\": \"" << id.corpus << "\",\n";
    os << "  \"objective\": " << id.objective << ",\n";
    os << "  \"mode\": " << id.mode << ",\n";
}

/* An entry is usable only when every identity field is present and matches.
 * A missing field means an entry written by an older layout: treat it as a
 * miss rather than accepting partially initialized memory. */
inline bool cache_identity_matches(const std::string& path, const autotune_cache_identity& want) {
    std::string model, variant, corpus;
    double objective = 0.0, mode = 0.0;
    if (!json_string_in_file(path, "model", model)) return false;
    if (!json_string_in_file(path, "variant", variant)) return false;
    if (!json_string_in_file(path, "corpus", corpus)) return false;
    if (!json_number_in_file(path, "objective", objective)) return false;
    if (!json_number_in_file(path, "mode", mode)) return false;
    return model == want.model && variant == want.variant && corpus == want.corpus &&
           (int)objective == want.objective && (int)mode == want.mode;
}

/* =========================================================================
 * Cache lifecycle
 * ========================================================================= */

/* Clear cache for a model (or all if model_name is nullptr).
 *
 * Entries are matched on the "model" field stored inside the file, because the
 * file name is a hardware-dependent hash that cannot be reconstructed from the
 * model name. Entries written by the legacy layout (no "model" field) are
 * matched on the sanitized model name embedded in their file name, delimited on
 * both sides so that "org/model" cannot match "org/model-v2". */
inline void clear_autotune_cache(const char* model_name, const std::string& subdir = "") {
    std::string dir = autotune_cache_dir();
    if (!subdir.empty()) dir += "/" + subdir;

    if (!model_name) {
        std::error_code ec;
        if (std::filesystem::exists(dir, ec))
            std::filesystem::remove_all(dir, ec);
        return;
    }

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return;

    std::string safe_model = sanitize_for_key(model_name);
    std::string legacy_token = "_" + safe_model + "_";

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        std::error_code fec;
        if (!entry.is_regular_file(fec)) continue;

        std::string path = entry.path().string();
        bool is_cache_file = (path.size() > 5 && path.compare(path.size() - 5, 5, ".json") == 0) ||
                             (path.size() > 9 && path.compare(path.size() - 9, 9, ".json.tmp") == 0);
        if (!is_cache_file) continue;

        bool match = path.find(legacy_token) != std::string::npos;
        if (!match) {
            std::string stored_model;
            if (json_string_in_file(path, "model", stored_model) && stored_model == model_name)
                match = true;
        }
        if (match) {
            std::error_code rm;
            std::filesystem::remove(path, rm);
        }
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_CACHE_HPP */
