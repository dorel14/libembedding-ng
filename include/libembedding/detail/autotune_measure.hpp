/*
 * libembedding - detail/autotune_measure.hpp
 * Measurement protocol shared by every auto-tuner.
 *
 * One place decides how a configuration is timed, so the four tuners cannot
 * drift apart again (they used to mix wall-clock totals, per-document averages
 * and P50s, and two of them reported invented memory figures):
 *
 *   - warmup calls are run and discarded (first-call allocations, JIT-like
 *     kernel selection in ORT, page faults on the weights);
 *   - every timed call is measured individually with a monotonic clock;
 *   - the result is the P50 and P95 of those calls, plus the throughput derived
 *     from the P50, which is robust to one slow outlier;
 *   - memory is the growth of the resident set across "create + warmup", an
 *     estimate of what the configuration costs, not an absolute peak.
 *
 * Nothing here knows about models: the caller supplies a callable.
 *
 * Auteur: David Orel
 * Version: 1.12.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_MEASURE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_MEASURE_HPP

#include "libembedding/autotuner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
/* The project's guarded windows.h + psapi.h include (WIN32_LEAN_AND_MEAN),
 * so this header can precede curl/winsock without redefinition errors. */
#  include "win_psapi_init.hpp"
#elif defined(__APPLE__)
#  include <mach/mach.h>
#else
#  include <unistd.h>
#endif

namespace lembed { namespace detail {

/* Outcome of measuring one configuration.
 *
 * `valid` is false when the configuration could not be created or when every
 * timed call failed: such a trial must never be selected, whatever its
 * (meaningless) numbers say. */
struct autotune_measurement {
    bool valid = false;
    double p50_ms = 0.0;           /* median duration of one timed call */
    double p95_ms = 0.0;           /* tail duration of one timed call */
    double throughput = 0.0;       /* items per second, from the P50 */
    double memory_mb = 0.0;        /* RSS growth across create + warmup */
    int items_per_call = 0;        /* items processed by one timed call */
    int calls_ok = 0;              /* timed calls that succeeded */
};

/* Linear-interpolation percentile over a pre-sorted sample.
 * Defined for every sample count (0, 1, n) and never reads out of bounds,
 * unlike an index computed as (size_t)(pct * n), which overflows for small n
 * and silently returns the maximum for the rest. */
inline double percentile_sorted(const std::vector<double>& sorted, double pct) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    if (pct <= 0.0) return sorted.front();
    if (pct >= 1.0) return sorted.back();
    double rank = pct * (double)(sorted.size() - 1);
    size_t lo = (size_t)std::floor(rank);
    size_t hi = (size_t)std::ceil(rank);
    double frac = rank - (double)lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

/* Current resident set size of this process, in MB; 0 when unavailable.
 * Same call as unified_benchmark_impl.hpp, so both link the same way. */
inline double autotune_current_rss_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS)
        return (double)info.resident_size / (1024.0 * 1024.0);
    return 0.0;
#else
    long pages = 0, resident = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0.0;
    int n = std::fscanf(f, "%ld %ld", &pages, &resident);
    std::fclose(f);
    if (n != 2) return 0.0;
    return (double)resident * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
#endif
}

/* Iteration counts of the protocol. QUICK keeps a tuning short; FULL trades
 * time for tighter percentiles.
 *
 * `prune_above_ms` implements racing: when the first TWO timed calls of a
 * configuration both exceed it, the remaining timed calls are skipped. The grid
 * search sets it to a multiple of the best P50 seen so far, so a configuration
 * that cannot win costs two calls instead of `iters`. Two samples, not one, so
 * a single transient spike (a thread pool spinning up on the first point of a
 * new session) cannot eliminate a configuration. 0 = off. */
struct autotune_protocol {
    int warmup = 1;
    int iters = 3;
    double prune_above_ms = 0.0;
};

inline autotune_protocol autotune_protocol_for(lembed_autotune_mode_t mode) {
    autotune_protocol p;
    if (mode == LEMBED_AUTOTUNE_FULL) {
        p.warmup = 2;
        p.iters = 10;
    }
    return p;
}

/* Times `call` (a callable returning bool) according to `proto`.
 *
 * `items_per_call` converts the P50 into a throughput. `rss_before_mb` is the
 * resident set sampled by the caller *before* it created the model, so the
 * memory figure covers the session and its first-call allocations. Pass a
 * negative value to skip the memory estimate. */
template <class Call>
autotune_measurement autotune_measure_calls(const autotune_protocol& proto,
                                            int items_per_call,
                                            double rss_before_mb,
                                            Call&& call) {
    using clock = std::chrono::steady_clock;
    autotune_measurement m;
    m.items_per_call = items_per_call;

    for (int i = 0; i < proto.warmup; i++) (void)call();
    if (rss_before_mb >= 0.0) {
        double after = autotune_current_rss_mb();
        m.memory_mb = (after > rss_before_mb) ? after - rss_before_mb : 0.0;
    }

    std::vector<double> times;
    times.reserve(proto.iters > 0 ? (size_t)proto.iters : 0);
    for (int i = 0; i < proto.iters; i++) {
        auto t0 = clock::now();
        bool ok = call();
        auto t1 = clock::now();
        if (!ok) continue;
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
        if (times.size() == 2 && proto.prune_above_ms > 0.0 &&
            std::min(times[0], times[1]) > proto.prune_above_ms)
            break;
    }

    m.calls_ok = (int)times.size();
    if (times.empty()) return m; /* valid stays false */

    std::sort(times.begin(), times.end());
    m.p50_ms = percentile_sorted(times, 0.50);
    m.p95_ms = percentile_sorted(times, 0.95);
    m.throughput = (m.p50_ms > 0.0) ? items_per_call * 1000.0 / m.p50_ms : 0.0;
    m.valid = m.p50_ms > 0.0;
    return m;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_MEASURE_HPP */
