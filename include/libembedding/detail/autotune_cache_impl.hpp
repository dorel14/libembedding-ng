/*
 * libembedding - detail/autotune_cache_impl.hpp
 * Autotuning cache with full hardware+software+model fingerprint
 *
 * Auteur: David Orel
 * Version: 1.14.0
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

#include <initializer_list>
#include <set>

#include "ggml.h"   /* ggml_version(): the software fingerprint */

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#include <sys/utsname.h>
#endif
#ifdef __APPLE__
#include <TargetConditionals.h>  /* TARGET_OS_IPHONE */
/* Declared here rather than through <sys/sysctl.h>: that header pulls in
 * <sys/proc.h>, which failed to compile in the macOS and iOS builds of
 * libembedding_shlib.cpp ("field has incomplete type 'struct itimerval'"),
 * this header being included after others that change the feature macros.
 * Signature from sysctlbyname(3), stable since Mac OS X 10.0. */
extern "C" int sysctlbyname(const char* name, void* oldp, size_t* oldlenp,
                            void* newp, size_t newlen);
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

/* Hardware detection helpers. The fingerprint keys the tuning cache, so a
 * wrong value is not cosmetic: before these, every Linux and macOS machine
 * reported "unknown", 1 core and 0 MB, and a result tuned on one CPU was
 * served to any other. */

static void copy_field(char* dst, size_t size, const std::string& value) {
    if (size == 0) return;
    const size_t n = value.size() < size - 1 ? value.size() : size - 1;
    memcpy(dst, value.data(), n);
    dst[n] = '\0';
}

static std::string trim_copy(const std::string& s) {
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

static void append_feature(std::string& list, const char* name) {
    if (!list.empty()) list += ",";
    list += name;
}

#if defined(__linux__)
/* First "key : value" line of /proc/cpuinfo whose key is one of `keys`. */
static std::string proc_cpuinfo_value(std::initializer_list<const char*> keys) {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim_copy(line.substr(0, colon));
        for (const char* k : keys) {
            if (key == k) return trim_copy(line.substr(colon + 1));
        }
    }
    return "";
}

/* Physical cores = distinct sibling sets among the online CPUs. Works on x86
 * (SMT pairs share a set) and on ARM (one CPU per set); falls back to the
 * logical count when sysfs is not readable (some sandboxes, old Android). */
static int linux_physical_cores(int logical) {
    std::set<std::string> cores;
    for (int cpu = 0; cpu < logical; cpu++) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream in(base + "core_cpus_list");
        if (!in) in.open(base + "thread_siblings_list");
        std::string siblings;
        if (!in || !std::getline(in, siblings)) return logical;
        cores.insert(trim_copy(siblings));
    }
    return cores.empty() ? logical : (int)cores.size();
}
#endif

#if defined(__APPLE__)
static std::string sysctl_string(const char* name) {
    size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return "";
    std::string value(size, '\0');
    if (sysctlbyname(name, &value[0], &size, nullptr, 0) != 0) return "";
    value.resize(strnlen(value.c_str(), size));
    return value;
}

static long long sysctl_int(const char* name) {
    long long value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return 0;
    /* hw.physicalcpu and friends are 32-bit: read the low part only. */
    if (size == sizeof(int)) {
        int v32 = 0;
        memcpy(&v32, &value, sizeof(v32));
        return v32;
    }
    return value;
}
#endif

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
    using namespace lembed::detail;
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
    copy_field(hw->cpu_name, sizeof(hw->cpu_name), trim_copy(brand));
    /* All processor groups (more than 64 logical CPUs), and physical cores
     * counted rather than guessed: logical/2 was wrong without SMT and on
     * hybrid CPUs, whose E-cores have no second thread. */
    hw->logical_cores = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<char> buf(len);
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
            int cores = 0;
            for (DWORD off = 0; off < len;) {
                auto* entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
                if (entry->Relationship == RelationProcessorCore) cores++;
                off += entry->Size;
            }
            hw->physical_cores = cores;
        }
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) hw->ram_mb = (int)(ms.ullTotalPhys / (1024*1024));
    strncpy(hw->os_name, "Windows", sizeof(hw->os_name) - 1);
    /* Detect features */
    std::string features;
    __cpuid(cpuInfo, 1);
    if (cpuInfo[2] & (1 << 28)) append_feature(features, "AVX");
    if (cpuInfo[2] & (1 << 12)) append_feature(features, "FMA");
    __cpuid(cpuInfo, 7);
    if (cpuInfo[1] & (1 << 5)) append_feature(features, "AVX2");
    if (cpuInfo[1] & (1 << 16)) append_feature(features, "AVX512F");
    copy_field(hw->features, sizeof(hw->features), features);
#elif defined(__linux__)
    /* x86 "model name"; ARM kernels give no name, Raspberry Pi OS adds "Model". */
    std::string name = proc_cpuinfo_value({"model name", "Model", "Hardware", "Processor", "cpu model"});
    std::string features;
    const std::string flags = " " + proc_cpuinfo_value({"flags", "Features"}) + " ";
    auto has = [&flags](const char* f) { return flags.find(std::string(" ") + f + " ") != std::string::npos; };
    if (has("avx")) append_feature(features, "AVX");
    if (has("fma")) append_feature(features, "FMA");
    if (has("avx2")) append_feature(features, "AVX2");
    if (has("avx512f")) append_feature(features, "AVX512F");
    if (has("neon") || has("asimd")) append_feature(features, "NEON");
    if (has("asimddp")) append_feature(features, "DOTPROD");
    if (has("i8mm")) append_feature(features, "I8MM");
    if (has("sve")) append_feature(features, "SVE");
    copy_field(hw->features, sizeof(hw->features), features);

    long online = sysconf(_SC_NPROCESSORS_ONLN);
    hw->logical_cores = online > 0 ? (int)online : 1;
    hw->physical_cores = linux_physical_cores(hw->logical_cores);
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        hw->ram_mb = (int)((unsigned long long)pages * (unsigned long long)page_size / (1024ULL * 1024ULL));
    }
    struct utsname un;
    const bool have_uname = uname(&un) == 0;
    if (name.empty() && have_uname) name = un.machine;
    copy_field(hw->cpu_name, sizeof(hw->cpu_name), name.empty() ? "unknown" : name);
#if defined(__ANDROID__)
    std::string os = "Android";
#else
    std::string os = "Linux";
#endif
    if (have_uname) os += std::string(" ") + un.release;
    copy_field(hw->os_name, sizeof(hw->os_name), os);
#elif defined(__APPLE__)
    std::string name = sysctl_string("machdep.cpu.brand_string");  /* Intel and Apple Silicon */
    if (name.empty()) name = sysctl_string("hw.machine");           /* iOS: e.g. "iPhone15,2" */
    copy_field(hw->cpu_name, sizeof(hw->cpu_name), name.empty() ? "unknown" : name);
    hw->logical_cores = (int)sysctl_int("hw.logicalcpu");
    hw->physical_cores = (int)sysctl_int("hw.physicalcpu");
    hw->ram_mb = (int)(sysctl_int("hw.memsize") / (1024LL * 1024LL));
#if TARGET_OS_IPHONE
    std::string os = "iOS";
#else
    std::string os = "macOS";
#endif
    const std::string release = sysctl_string("kern.osproductversion");
    if (!release.empty()) os += " " + release;
    copy_field(hw->os_name, sizeof(hw->os_name), os);
    std::string features;
#if defined(__aarch64__) || defined(__arm64__)
    append_feature(features, "NEON");
    if (sysctl_int("hw.optional.arm.FEAT_DotProd")) append_feature(features, "DOTPROD");
    if (sysctl_int("hw.optional.arm.FEAT_I8MM")) append_feature(features, "I8MM");
#else
    if (sysctl_int("hw.optional.avx1_0")) append_feature(features, "AVX");
    if (sysctl_int("hw.optional.fma")) append_feature(features, "FMA");
    if (sysctl_int("hw.optional.avx2_0")) append_feature(features, "AVX2");
    if (sysctl_int("hw.optional.avx512f")) append_feature(features, "AVX512F");
#endif
    copy_field(hw->features, sizeof(hw->features), features);
#else
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    hw->logical_cores = online > 0 ? (int)online : 1;
    struct utsname un;
    if (uname(&un) == 0) {
        copy_field(hw->cpu_name, sizeof(hw->cpu_name), un.machine);
        copy_field(hw->os_name, sizeof(hw->os_name), un.sysname);
    } else {
        copy_field(hw->cpu_name, sizeof(hw->cpu_name), "unknown");
        copy_field(hw->os_name, sizeof(hw->os_name), "unknown");
    }
#endif
    if (hw->logical_cores < 1) hw->logical_cores = 1;
    if (hw->physical_cores < 1 || hw->physical_cores > hw->logical_cores) {
        hw->physical_cores = hw->logical_cores;
    }
    return LEMBED_OK;
}

lembed_status_t lembed_cache_detect_software(lembed_cache_software_info_t* sw) {
    if (!sw) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(sw, 0, sizeof(*sw));
    strncpy(sw->libembedding, LIBEMBEDDING_VERSION_STRING, sizeof(sw->libembedding) - 1);
    /* The bundled snapshot is identified by its ggml version (llama.cpp
     * v0.6.0 ships ggml 0.26.0): a bump changes the kernels, so it must
     * invalidate tuning results. The previous constant ("b5434") never moved. */
    const char* ggml = ggml_version();
    lembed::detail::copy_field(sw->llama_cpp, sizeof(sw->llama_cpp),
                               std::string("ggml ") + (ggml ? ggml : "unknown"));
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