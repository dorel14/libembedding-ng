/*
 * libembedding - detail/autotune_cache_impl.hpp
 * Autotuning cache with full hardware+software+model fingerprint
 *
 * Auteur: David Orel
 * Version: 1.9.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_AUTOTUNE_CACHE_IMPL_HPP
#define LIBEMBEDDING_AUTOTUNE_CACHE_IMPL_HPP

#include "../autotune_cache.h"
#include "../config.h"
#include "cJSON.h"
#include "autotune_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#endif

namespace lembed {
namespace detail {

static std::string cache_file() {
    return autotune_cache_dir() + "/tune_cache.json";
}

static std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return ""; }
    long size = ftell(f);
    if (size < 0) { fclose(f); return ""; }
    std::rewind(f);
    std::string content((size_t)size, '\0');
    /* A short read (file truncated between the seeks, I/O error) would leave
     * NUL bytes in the middle: the caller parses it with cJSON_Parse(), which
     * stops at the first NUL and would silently read a truncated cache. Treat
     * it as "no cache" instead. */
    if (size > 0 && fread(&content[0], 1, (size_t)size, f) != (size_t)size) {
        fclose(f);
        return "";
    }
    fclose(f);
    return content;
}

static bool write_file(const std::string& path, const std::string& content) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(content.data(), 1, content.size(), f);
    fclose(f);
    return true;
}

static cJSON* load_cache() {
    std::string content = read_file(cache_file());
    if (content.empty()) return cJSON_CreateObject();
    cJSON* root = cJSON_Parse(content.c_str());
    return root ? root : cJSON_CreateObject();
}

static bool save_cache(cJSON* root) {
    char* json = cJSON_Print(root);
    if (!json) return false;
    bool ok = write_file(cache_file(), json);
    free(json);
    return ok;
}

} /* namespace detail */
} /* namespace lembed */

#ifdef LIBEMBEDDING_IMPLEMENTATION

const char* lembed_tune_cache_path(void) {
    static std::string path = lembed::detail::cache_file();
    return path.c_str();
}

lembed_status_t lembed_cache_detect_hardware(lembed_cache_hardware_info_t* hw) {
    if (!hw) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(hw, 0, sizeof(*hw));
#ifdef _WIN32
    char brand[0x40] = {0};
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0x80000002);
    memcpy(brand, cpuInfo, sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000003);
    memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000004);
    memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
    strncpy(hw->cpu_name, brand, sizeof(hw->cpu_name) - 1);
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    hw->logical_cores = (int)si.dwNumberOfProcessors;
    hw->physical_cores = hw->logical_cores > 1 ? hw->logical_cores / 2 : 1;
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) hw->ram_mb = (int)(ms.ullTotalPhys / (1024*1024));
    strncpy(hw->os_name, "Windows", sizeof(hw->os_name) - 1);
    /* Detect features */
    __cpuid(cpuInfo, 1);
    if (cpuInfo[2] & (1 << 28)) strncpy(hw->features, "AVX", sizeof(hw->features) - 1);
    __cpuid(cpuInfo, 7);
    if (cpuInfo[1] & (1 << 5)) {
        if (hw->features[0]) strncat(hw->features, ",AVX2", sizeof(hw->features) - strlen(hw->features) - 1);
        else strncpy(hw->features, "AVX2", sizeof(hw->features) - 1);
    }
#else
    strncpy(hw->cpu_name, "unknown", sizeof(hw->cpu_name) - 1);
    hw->physical_cores = 1;
    hw->logical_cores = 1;
    strncpy(hw->os_name, "unknown", sizeof(hw->os_name) - 1);
#endif
    return LEMBED_OK;
}

lembed_status_t lembed_cache_detect_software(lembed_cache_software_info_t* sw) {
    if (!sw) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(sw, 0, sizeof(*sw));
    strncpy(sw->libembedding, LIBEMBEDDING_VERSION_STRING, sizeof(sw->libembedding) - 1);
    strncpy(sw->llama_cpp, "b5434", sizeof(sw->llama_cpp) - 1); /* TODO: get from llama.cpp */
    return LEMBED_OK;
}

void lembed_tune_cache_key(const lembed_cache_hardware_info_t* hw,
                           const lembed_cache_software_info_t* sw,
                           const lembed_cache_model_info_t* model,
                           const char* backend,
                           char* key_out) {
    if (!key_out) return;
    /* The fingerprint fields can hold up to 415 bytes in total (127 + 63 + 31
     * + 31 + 127 + 31 plus the separators), which does not fit in a fixed key
     * buffer: a truncated key would make two different fingerprints collide and
     * hand back a tuning result measured on another CPU or model. The key is
     * therefore a 64-bit FNV-1a hash of the whole fingerprint, rendered as
     * 16 hex characters. The readable fields are still stored in the cache
     * entry ("cpu", "os", "libembedding", "llama_cpp", "model", "backend"). */
    std::string fp;
    fp.reserve(512);
    auto append = [&fp](const char* value, char separator) {
        fp.append(value ? value : "unknown");
        fp.push_back(separator);
    };
    append(hw ? hw->cpu_name : nullptr, '|');
    append(hw ? hw->os_name : nullptr, '|');
    append(sw ? sw->libembedding : nullptr, '|');
    append(sw ? sw->llama_cpp : nullptr, '|');
    append(model ? model->model_id : nullptr, '|');
    /* The last field is terminated by a NUL, which cannot appear inside the
     * fixed-size fingerprint fields, so no two different fingerprints can build
     * the same string. */
    append(backend, '\0');

    uint64_t hash = 1469598103934665603ULL; /* FNV-1a 64-bit offset basis */
    for (unsigned char c : fp) {
        hash ^= (uint64_t)c;
        hash *= 1099511628211ULL; /* FNV-1a 64-bit prime */
    }
    /* %llx rather than PRIx64: <cinttypes> pulls in the whole inttypes family
     * and does not compile cleanly with MSVC here. */
    snprintf(key_out, LEMBED_TUNE_CACHE_KEY_SIZE, "%016llx",
             (unsigned long long)hash);
}

void lembed_tune_cache_add_config(lembed_tune_cache_entry_t* entry,
                                   const lembed_tune_config_result_t* config) {
    if (!entry || !config || entry->num_configs >= 16) return;
    entry->configs[entry->num_configs++] = *config;
}

void lembed_tune_cache_set_best(lembed_tune_cache_entry_t* entry, int idx) {
    if (!entry || idx < 0 || idx >= entry->num_configs) return;
    entry->best_idx = idx;
}

lembed_status_t lembed_tune_cache_load(
    const lembed_cache_hardware_info_t* hw,
    const lembed_cache_software_info_t* sw,
    const lembed_cache_model_info_t* model,
    const char* backend,
    lembed_tune_cache_entry_t* entry) {
    if (!hw || !sw || !model || !backend || !entry) return LEMBED_ERROR_INVALID_ARGUMENT;
    char key[LEMBED_TUNE_CACHE_KEY_SIZE];
    lembed_tune_cache_key(hw, sw, model, backend, key);
    cJSON* root = lembed::detail::load_cache();
    cJSON* item = cJSON_GetObjectItem(root, key);
    if (!item) { cJSON_Delete(root); return LEMBED_ERROR_CACHE_MISS; }
    memset(entry, 0, sizeof(*entry));
    entry->cache_schema_version = LEMBED_TUNE_CACHE_SCHEMA_VERSION;
    cJSON* v;
    if ((v = cJSON_GetObjectItem(item, "schema_version"))) entry->cache_schema_version = v->valueint;
    /* Load best config */
    if ((v = cJSON_GetObjectItem(item, "best_idx"))) entry->best_idx = v->valueint;
    cJSON* configs = cJSON_GetObjectItem(item, "configs");
    if (configs) {
        int n = cJSON_GetArraySize(configs);
        entry->num_configs = 0;
        for (int i = 0; i < n && i < 16; i++) {
            cJSON* c = cJSON_GetArrayItem(configs, i);
            lembed_tune_config_result_t cr = {0};
            if ((v = cJSON_GetObjectItem(c, "sessions"))) cr.num_sessions = v->valueint;
            if ((v = cJSON_GetObjectItem(c, "threads"))) cr.num_threads = v->valueint;
            if ((v = cJSON_GetObjectItem(c, "batch_size"))) cr.batch_size = v->valueint;
            if ((v = cJSON_GetObjectItem(c, "throughput"))) cr.throughput_docs_sec = (float)v->valuedouble;
            if ((v = cJSON_GetObjectItem(c, "latency_p50"))) cr.latency_p50_ms = (float)v->valuedouble;
            if ((v = cJSON_GetObjectItem(c, "latency_p95"))) cr.latency_p95_ms = (float)v->valuedouble;
            lembed_tune_cache_add_config(entry, &cr);
        }
    }
    cJSON_Delete(root);
    return LEMBED_OK;
}

lembed_status_t lembed_tune_cache_save(const lembed_tune_cache_entry_t* entry) {
    if (!entry) return LEMBED_ERROR_INVALID_ARGUMENT;
    lembed_cache_hardware_info_t hw = entry->hardware;
    lembed_cache_software_info_t sw = entry->software;
    lembed_cache_model_info_t model = entry->model;
    char key[LEMBED_TUNE_CACHE_KEY_SIZE];
    lembed_tune_cache_key(&hw, &sw, &model, entry->backend, key);
    cJSON* root = lembed::detail::load_cache();
    cJSON* item = cJSON_CreateObject();
    cJSON_AddNumberToObject(item, "schema_version", LEMBED_TUNE_CACHE_SCHEMA_VERSION);
    cJSON_AddStringToObject(item, "model", model.model_id);
    cJSON_AddStringToObject(item, "backend", entry->backend);
    cJSON_AddNumberToObject(item, "dim", model.dim);
    cJSON_AddNumberToObject(item, "best_idx", entry->best_idx);
    /* Store all configs */
    cJSON* configs = cJSON_CreateArray();
    for (int i = 0; i < entry->num_configs; i++) {
        cJSON* c = cJSON_CreateObject();
        cJSON_AddNumberToObject(c, "sessions", entry->configs[i].num_sessions);
        cJSON_AddNumberToObject(c, "threads", entry->configs[i].num_threads);
        cJSON_AddNumberToObject(c, "batch_size", entry->configs[i].batch_size);
        cJSON_AddNumberToObject(c, "throughput", entry->configs[i].throughput_docs_sec);
        cJSON_AddNumberToObject(c, "latency_p50", entry->configs[i].latency_p50_ms);
        cJSON_AddNumberToObject(c, "latency_p95", entry->configs[i].latency_p95_ms);
        cJSON_AddItemToArray(configs, c);
    }
    cJSON_AddItemToObject(item, "configs", configs);
    /* Fingerprints for debugging */
    cJSON_AddStringToObject(item, "cpu", hw.cpu_name);
    cJSON_AddStringToObject(item, "os", hw.os_name);
    cJSON_AddStringToObject(item, "libembedding", sw.libembedding);
    cJSON_AddStringToObject(item, "llama_cpp", sw.llama_cpp);
    cJSON_AddStringToObject(item, "quantization", model.quantization);
    cJSON_AddItemToObject(root, key, item);
    bool ok = lembed::detail::save_cache(root);
    cJSON_Delete(root);
    return ok ? LEMBED_OK : LEMBED_ERROR_IO;
}

lembed_status_t lembed_tune_cache_clear(void) {
    std::filesystem::remove(lembed::detail::cache_file());
    return LEMBED_OK;
}

#endif /* LIBEMBEDDING_IMPLEMENTATION */
#endif /* LIBEMBEDDING_AUTOTUNE_CACHE_IMPL_HPP */