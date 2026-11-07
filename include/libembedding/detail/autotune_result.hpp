/*
 * libembedding - detail/autotune_result.hpp
 * lembed_unified_tuning_result_t as the tuners' common currency.
 *
 * Every tuner fills a unified result; the task-specific public structs
 * (lembed_tuning_result_t, lembed_reranker_tuning_result_t, ...) are produced
 * by small conversions in the legacy facades. This keeps one cache layout, one
 * way to copy a measurement into a result, and lets every task report P95.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_RESULT_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_RESULT_HPP

#include "libembedding/autotuner.h"
#include "libembedding/error.h"
#include "autotune_cache.hpp"
#include "autotune_measure.hpp"

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* =========================================================================
 * Error reporting (tuners are compiled with LIBEMBEDDING_IMPLEMENTATION)
 * ========================================================================= */

/* Records `msg` for lembed_last_error() and returns `status`, so every error
 * path of a tuner explains itself instead of leaving a stale message. */
inline lembed_status_t autotune_fail(lembed_status_t status, const std::string& msg) {
    set_error(msg);
    return status;
}

/* "no configuration could run", carrying the last error a model creation or an
 * inference call left behind (copied first: set_error overwrites it). */
inline lembed_status_t autotune_fail_no_config(const char* label, const std::string& model) {
    std::string cause = lembed_last_error();
    std::string msg = std::string(label) + ": no configuration of '" + model + "' could run";
    if (!cause.empty()) msg += " (last error: " + cause + ")";
    return autotune_fail(LEMBED_ERROR_ONNX_RUNTIME, msg);
}

/* Runs `body` (returning lembed_status_t) and converts every exception into a
 * status + message: nothing may unwind through the C ABI. The handlers only use
 * fixed-size buffers, so they cannot throw themselves. */
template <class Body>
lembed_status_t autotune_guard(const char* label, Body&& body) noexcept {
    char msg[512];
    /* Start clean: an error message left by an earlier, unrelated call must not
     * be reported as the cause of this one (autotune_fail_no_config quotes it). */
    set_error("");
    try {
        return body();
    } catch (const std::bad_alloc&) {
        std::snprintf(msg, sizeof(msg), "%s: out of memory", label);
        set_error(msg);
        return LEMBED_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        std::snprintf(msg, sizeof(msg), "%s: %s", label, e.what());
        set_error(msg);
        return LEMBED_ERROR_ONNX_RUNTIME;
    } catch (...) {
        std::snprintf(msg, sizeof(msg), "%s: unknown exception", label);
        set_error(msg);
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

inline lembed_unified_tuning_result_t make_unified_result(lembed_task_t task) {
    lembed_unified_tuning_result_t r;
    std::memset(&r, 0, sizeof(r));
    r.task = task;
    return r;
}

/* Copies the performance figures of the winning measurement. Latencies are the
 * duration of one tuning call; tasks that report per-item latency rescale. */
inline void set_result_metrics(lembed_unified_tuning_result_t& r, const autotune_measurement& m) {
    r.throughput_docs_sec = m.throughput;
    r.latency_ms = m.p50_ms;
    r.p95_latency_ms = m.p95_ms;
    r.memory_mb = m.memory_mb;
}

/* Cache fields of a result: the common figures plus the configuration knobs of
 * r.task. Names match the entries earlier releases wrote, so a field list is
 * also the documentation of the on-disk layout. r.task must be set. */
inline std::vector<cache_field> unified_cache_fields(lembed_unified_tuning_result_t& r) {
    std::vector<cache_field> f = {
        cache_int("threads", r.threads),
        cache_int("batch_size", r.batch_size),
        cache_double("throughput_docs_sec", r.throughput_docs_sec),
        cache_double("latency_ms", r.latency_ms),
        cache_double("p95_latency_ms", r.p95_latency_ms),
        cache_double("memory_mb", r.memory_mb),
    };
    switch (r.task) {
        case LEMBED_TASK_EMBEDDING:
            f.push_back(cache_int("workers", r.workers));
            break;
        case LEMBED_TASK_RERANKING:
            f.push_back(cache_int("max_tokens", r.max_tokens));
            break;
        case LEMBED_TASK_SPARSE:
            f.push_back(cache_int("top_k", r.top_k));
            f.push_back(cache_float("min_weight", r.min_weight));
            f.push_back(cache_int("storage_format", r.storage_format));
            break;
        case LEMBED_TASK_IMAGE:
        default:
            break;
    }
    return f;
}

/* Cache read/write of a unified result for its task's sub-directory. */
inline bool read_unified_cache(const char* subdir, const autotune_cache_identity& id,
                               lembed_unified_tuning_result_t& out) {
    lembed_unified_tuning_result_t tmp = make_unified_result(out.task);
    if (!read_cache_entry(subdir, id, unified_cache_fields(tmp))) return false;
    out = tmp;
    return true;
}

inline void write_unified_cache(const char* subdir, const autotune_cache_identity& id,
                                const lembed_unified_tuning_result_t& r) {
    lembed_unified_tuning_result_t copy = r;
    write_cache_entry(subdir, id, unified_cache_fields(copy));
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_RESULT_HPP */
