/*
 * libembedding - detail/worker_autotune_impl.hpp
 * Auto-tuning implementation for llama.cpp workers/sessions
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_WORKER_AUTOTUNE_IMPL_HPP
#define LIBEMBEDDING_DETAIL_WORKER_AUTOTUNE_IMPL_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <random>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lembed { namespace detail {

/* =========================================================================
 * Hardware detection
 * ========================================================================= */

static int detect_physical_cores() {
#if defined(_WIN32) || defined(_WIN64)
    DWORD len = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len)) {
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<char> buf(len);
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
                int cores = 0;
                DWORD offset = 0;
                while (offset < len) {
                    if (info->Relationship == RelationProcessorCore) {
                        cores++;
                    }
                    offset += info->Size;
                    info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                        reinterpret_cast<char*>(info) + info->Size);
                }
                if (cores > 0) return cores;
            }
        }
    }
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return (int)sysInfo.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

static int detect_logical_cores() {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return (int)sysInfo.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_CONF);
#endif
}

/* =========================================================================
 * Benchmark helpers
 * ========================================================================= */

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* =========================================================================
 * Public API
 * ========================================================================= */

inline lembed_worker_config_t detect_optimal_workers() {
    lembed_worker_config_t cfg;
    cfg.physical_cores = detect_physical_cores();
    cfg.logical_cores = detect_logical_cores();

    /* Heuristic: for small BERT embedding models on CPU,
       sessions = min(physical_cores * 2, 8) is usually optimal.
       threads per session = 1.
    */
    cfg.optimal_workers = std::min(cfg.physical_cores * 2, 8);
    cfg.optimal_threads = 1;

    return cfg;
}

inline int recommended_workers_for_model(const char* model_path) {
    (void)model_path;

    /* Placeholder: future implementation can inspect model size
       and adjust worker count accordingly. For now, use hardware heuristic. */
    lembed_worker_config_t cfg = detect_optimal_workers();
    return cfg.optimal_workers;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_WORKER_AUTOTUNE_IMPL_HPP */
