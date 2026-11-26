/*
 * libembedding - detail/autotune_bench_sparse.hpp
 * Sparse (SPLADE-style) embedding auto-tuner
 *
 * Tunes the performance knobs of a sparse model: threads x batch_size.
 *
 * top_k, min_weight and storage_format are deliberately NOT tuned. They change
 * the vectors that come out (how many terms survive, in which layout), so
 * choosing them for speed would silently trade recall for throughput; their
 * post-processing cost is also negligible next to the transformer itself. The
 * result reports the values the caller would get by default, unchanged.
 *
 * Auteur: David Orel
 * Version: 1.16.6
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP

#include "libembedding/autotuner.h"
#include "enum_check_autotune.hpp"
#include "libembedding/sparse_text_embedding.h"
#include "libembedding/model_registry.h"
#include "libembedding/gguf_registry.h"
#include "libembedding/downloader.h"
#include "sparse_batching.hpp"
#include "autotune_cache.hpp"
#include "autotune_corpus.hpp"
#include "autotune_result.hpp"
#include "autotune_search.hpp"
#include "autotune_session.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace lembed { namespace detail {

/* =========================================================================
 * Configuration space
 * ========================================================================= */

struct sparse_tune_config {
    int threads;
    int batch;
};

inline std::string describe_sparse_config(const sparse_tune_config& c) {
    return "threads=" + std::to_string(c.threads) + " batch=" + std::to_string(c.batch);
}

/* Sparse models are BERT-base sized: single-threaded runs are never
 * competitive, so the grid starts at half the physical cores. */
inline std::vector<sparse_tune_config> sparse_tune_grid(lembed_autotune_mode_t mode) {
    int physical = std::max(1, cpu_physical_cores());
    int logical = std::max(physical, cpu_logical_cores());
    std::vector<int> threads = {physical, logical};
    if (mode == LEMBED_AUTOTUNE_FULL) threads.push_back(std::max(1, physical / 2));
    std::sort(threads.begin(), threads.end());
    threads.erase(std::unique(threads.begin(), threads.end()), threads.end());

    std::vector<int> batches = (mode == LEMBED_AUTOTUNE_QUICK) ? std::vector<int>{4, 16}
                                                               : std::vector<int>{4, 8, 16, 32};
    std::vector<sparse_tune_config> grid;
    for (int t : threads)
        for (int b : batches) grid.push_back({t, b});
    return grid;
}

inline int sparse_tune_corpus_size(lembed_autotune_mode_t mode) {
    return (mode == LEMBED_AUTOTUNE_QUICK) ? 16 : 64;
}

/* =========================================================================
 * Measuring one configuration
 * ========================================================================= */

/* The batch size is a per-call argument: one session per thread count. */
using sparse_session_cache = autotune_session_cache<int, lembed_sparse_embedding_ctx_t>;

inline lembed_sparse_options_t sparse_tune_options(lembed_sparse_model_t model, int threads) {
    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.model = model;
    opts.num_threads = threads;
    opts.offline = 1;
    opts.show_download_progress = 0;
    return opts;
}

inline autotune_measurement bench_sparse_config(lembed_sparse_model_t model,
                                                const std::vector<std::string>& texts,
                                                const sparse_tune_config& c,
                                                const autotune_protocol& proto,
                                                sparse_session_cache& sessions) {
    autotune_measurement failed;
    if (texts.empty()) return failed;

    lembed_sparse_options_t opts = sparse_tune_options(model, c.threads);
    lembed_sparse_embedding_ctx_t* ctx =
        sessions.get(c.threads, [&]() -> lembed_sparse_embedding_ctx_t* {
            lembed_sparse_embedding_ctx_t* created = nullptr;
            return lembed_sparse_text_embedding_create(&opts, &created) == LEMBED_OK ? created
                                                                                    : nullptr;
        });
    if (!ctx) return failed;

    std::vector<const char*> c_texts = corpus_c_strings(texts);
    const int n = (int)c_texts.size();
    return autotune_measure_calls(proto, n, sessions.rss_before_mb(), [&]() {
        lembed_sparse_embeddings_t r = {0};
        bool ok = lembed_sparse_text_embedding_embed(ctx, c_texts.data(), n, c.batch, &opts, &r) ==
                  LEMBED_OK;
        if (ok) lembed_sparse_embeddings_free(&r);
        return ok;
    });
}

/* =========================================================================
 * Cache
 * ========================================================================= */

inline const char* sparse_cache_subdir() { return "sparse"; }

/* The sparse GGUF registry entry named `name` (exact, case-insensitive), or
 * nullptr. Checked before the ONNX registry, as SparseTextEmbedding does, so the
 * "...-GGUF" name is never taken for the ONNX model of the same family. */
inline const lembed_gguf_model_info_t* sparse_gguf_entry(const char* name) {
    return name ? lembed_find_sparse_gguf_model(name) : nullptr;
}

/* A path ending in ".gguf", which create_from_path() routes to the GGUF runtime. */
inline bool sparse_is_gguf_path(const char* name) {
    if (!name) return false;
    std::string s(name);
    if (s.size() < 5) return false;
    std::string ext = s.substr(s.size() - 5);
    for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
    return ext == ".gguf";
}

/* Cache identity of a GGUF target: registry entries by name, files by absolute
 * path with forward slashes (the entry is a JSON string). */
inline std::string sparse_gguf_model_id(const char* name) {
    if (const lembed_gguf_model_info_t* e = sparse_gguf_entry(name)) {
        std::string id = std::string("gguf:") + e->name;
        for (char& ch : id) ch = (char)std::tolower((unsigned char)ch);
        return id;
    }
    std::error_code ec;
    std::filesystem::path p = std::filesystem::absolute(std::filesystem::path(name), ec);
    /* The id ends up in the cache file name: file name + a hash of the full path,
     * not the path itself, which can exceed MAX_PATH. */
    std::string full = ec ? std::string(name) : p.generic_string();
    return "gguf:" + std::filesystem::path(full).filename().string() + "_" +
           hash64_to_hex(fnv1a_update(1469598103934665603ULL, full.data(), full.size()))
               .substr(0, 12);
}

/* Canonical registry code, so both name forms share entries and purges. */
inline std::string canonical_sparse_id(const char* model) {
    if (!model) return std::string();
    if (sparse_gguf_entry(model) || sparse_is_gguf_path(model)) return sparse_gguf_model_id(model);
    int idx = lembed_resolve_sparse_model(model);
    if (idx < 0) return std::string(model);
    return std::string(lembed__sparse_models[idx].model_code);
}

/* =========================================================================
 * Sparse GGUF runtime
 *
 * Knobs: threads x workers (graphs sharing the weights) x batch_size x
 * max_batch_tokens (token budget per batch). The default configuration of the
 * runtime is point 0 of the grid, so the result is never slower than what a
 * caller gets without tuning. Output-changing options are never in the grid.
 * ========================================================================= */

struct sparse_gguf_config {
    int threads;
    int workers;  /* 0 = the runtime's automatic split */
    int batch;
    int max_tokens;
    bool operator==(const sparse_gguf_config& o) const {
        return threads == o.threads && workers == o.workers && batch == o.batch &&
               max_tokens == o.max_tokens;
    }
};

inline std::string describe_sparse_gguf_config(const sparse_gguf_config& c) {
    return "threads=" + std::to_string(c.threads) + " workers=" + std::to_string(c.workers) +
           " batch=" + std::to_string(c.batch) + " max_batch_tokens=" + std::to_string(c.max_tokens);
}

/* What a caller gets with lembed_sparse_options_default(): every logical CPU,
 * automatic workers, LEMBED_GGUF_DEFAULT_BATCH_SIZE, the default token budget. */
inline sparse_gguf_config sparse_gguf_default_config() {
    return {std::max(1, cpu_logical_cores()), 0, LEMBED_GGUF_DEFAULT_BATCH_SIZE,
            kSparseMaxBatchTokens};
}

/* Default first, then the rest grouped by (threads, workers): the weights are
 * shared by workers, so a session is reloaded only when one of those changes. */
inline std::vector<sparse_gguf_config> sparse_gguf_grid(lembed_autotune_mode_t mode) {
    const sparse_gguf_config def = sparse_gguf_default_config();
    int physical = std::max(1, cpu_physical_cores());
    std::vector<int> threads = {def.threads, physical};
    if (mode == LEMBED_AUTOTUNE_FULL) threads.push_back(std::max(1, physical / 2));
    std::sort(threads.begin(), threads.end());
    threads.erase(std::unique(threads.begin(), threads.end()), threads.end());

    std::vector<int> workers = (mode == LEMBED_AUTOTUNE_QUICK) ? std::vector<int>{0, 1}
                                                               : std::vector<int>{0, 1, 2, 4};
    std::vector<int> batches = (mode == LEMBED_AUTOTUNE_QUICK) ? std::vector<int>{16, 32}
                                                               : std::vector<int>{8, 16, 32};
    std::vector<int> budgets = (mode == LEMBED_AUTOTUNE_QUICK) ? std::vector<int>{512, 1024}
                                                               : std::vector<int>{256, 512, 1024, 2048};
    std::vector<sparse_gguf_config> grid = {def};
    for (int t : threads)
        for (int w : workers) {
            if (w > t) continue;
            for (int b : batches)
                for (int m : budgets) {
                    sparse_gguf_config c = {t, w, b, m};
                    if (!(c == def)) grid.push_back(c);
                }
        }
    return grid;
}

/* One session per (threads, workers); the token budget is applied per point. */
using sparse_gguf_session_cache =
    autotune_session_cache<std::pair<int, int>, lembed_sparse_embedding_ctx_t>;

inline autotune_measurement bench_sparse_gguf_config(const std::string& path,
                                                     const std::vector<std::string>& texts,
                                                     const sparse_gguf_config& c,
                                                     const autotune_protocol& proto,
                                                     sparse_gguf_session_cache& sessions) {
    autotune_measurement failed;
    if (texts.empty()) return failed;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.num_threads = c.threads;
    opts.workers = c.workers;
    opts.offline = 1;
    opts.show_download_progress = 0;
    lembed_sparse_embedding_ctx_t* ctx = sessions.get(
        std::make_pair(c.threads, c.workers), [&]() -> lembed_sparse_embedding_ctx_t* {
            lembed_sparse_embedding_ctx_t* created = nullptr;
            return lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts,
                                                                      &created) == LEMBED_OK
                       ? created
                       : nullptr;
        });
    if (!ctx) return failed;
    if (lembed_sparse_text_embedding_set_max_batch_tokens(ctx, c.max_tokens) != LEMBED_OK)
        return failed;

    std::vector<const char*> c_texts = corpus_c_strings(texts);
    const int n = (int)c_texts.size();
    return autotune_measure_calls(proto, n, sessions.rss_before_mb(), [&]() {
        lembed_sparse_embeddings_t r = {0};
        bool ok = lembed_sparse_text_embedding_embed(ctx, c_texts.data(), n, c.batch, &opts, &r) ==
                  LEMBED_OK;
        if (ok) lembed_sparse_embeddings_free(&r);
        return ok;
    });
}

/* Local file of a GGUF target: the path itself, or the cached file of a registry
 * entry (never downloaded: tuning needs the weights present, like the ONNX
 * tuner). Empty with the reason recorded on failure. */
inline std::string sparse_gguf_local_path(const char* name) {
    if (const lembed_gguf_model_info_t* e = sparse_gguf_entry(name)) {
        char* resolved = nullptr;
        if (lembed_resolve_gguf_path(e->gguf_url, nullptr, 1, &resolved) != LEMBED_OK || !resolved)
            return std::string();
        std::string p(resolved);
        lembed_free_string(resolved);
        return p;
    }
    return std::string(name);
}

/* Size and mtime of the file, so a replaced model never reuses an old entry. */
inline std::string sparse_gguf_file_stamp(const std::string& path) {
    std::error_code ec;
    auto size = std::filesystem::file_size(path, ec);
    long long sz = ec ? -1 : (long long)size;
    auto mt = std::filesystem::last_write_time(path, ec);
    long long t = ec ? 0 : (long long)mt.time_since_epoch().count();
    return std::to_string(sz) + "_" + std::to_string(t);
}

/* =========================================================================
 * Tuner
 * ========================================================================= */

inline lembed_objective_t sparse_default_objective() { return LEMBED_OBJECTIVE_THROUGHPUT; }

/* GGUF branch of sparse_autotune_run(). The unified result carries workers in
 * `workers` and the token budget in `max_tokens`. */
inline lembed_status_t sparse_gguf_autotune_run(const char* model_name,
                                                const std::vector<std::string>& corpus,
                                                lembed_autotune_mode_t mode,
                                                lembed_objective_t objective,
                                                double max_latency_ms,
                                                lembed_unified_tuning_result_t* result,
                                                bool* from_cache) {
    std::string path = sparse_gguf_local_path(model_name);
    if (path.empty() || !std::filesystem::exists(path))
        return autotune_fail(LEMBED_ERROR_MODEL_NOT_FOUND,
                             std::string("sparse_autotune: GGUF model '") + model_name +
                                 "' is not available locally");

    std::vector<std::string> texts =
        corpus.empty() ? synthetic_text_corpus(sparse_tune_corpus_size(mode)) : corpus;

    char variant[64];
    std::snprintf(variant, sizeof(variant), "gguf_%s_p95_%.3f",
                  corpus.empty() ? "synthetic" : "custom", max_latency_ms);
    autotune_cache_identity id;
    id.model = sparse_gguf_model_id(model_name);
    id.variant = std::string(variant) + "_" + sparse_gguf_file_stamp(path);
    id.corpus = corpus_fingerprint(texts);
    id.objective = (int)objective;
    id.mode = (int)mode;

    if (read_unified_cache(sparse_cache_subdir(), id, *result)) {
        if (from_cache) *from_cache = true;
        return LEMBED_OK;
    }

    autotune_search_options sopts;
    sopts.objective = objective;
    sopts.constraints.max_p95_ms = max_latency_ms;
    sopts.protocol = autotune_protocol_for(mode);
    sopts.label = "sparse_gguf_autotune";

    sparse_gguf_session_cache sessions(lembed_sparse_text_embedding_free);
    auto search = autotune_grid_search(
        sparse_gguf_grid(mode),
        [&](const sparse_gguf_config& c, const autotune_protocol& p) {
            return bench_sparse_gguf_config(path, texts, c, p, sessions);
        },
        describe_sparse_gguf_config, sopts);
    sessions.reset();
    if (!search.found()) return autotune_fail_no_config("sparse_gguf_autotune", id.model);
    /* Point 0 is the default configuration: never return something worse. */
    autotune_keep_baseline(search, 0, objective, sopts.constraints);

    lembed_sparse_options_t defaults = lembed_sparse_options_default();
    const sparse_gguf_config& best = search.best_config();
    result->threads = best.threads;
    result->workers = best.workers;
    result->batch_size = best.batch;
    result->max_tokens = best.max_tokens;
    result->top_k = defaults.top_k;
    result->min_weight = defaults.min_weight;
    result->storage_format = defaults.storage_format;
    set_result_metrics(*result, search.best_measurement());

    write_unified_cache(sparse_cache_subdir(), id, *result);
    return LEMBED_OK;
}

/* `corpus` empty = synthetic corpus. max_latency_ms bounds the p95 of one call
 * over the corpus (0 = none). `model_name` is an ONNX registry name, a sparse
 * GGUF registry name or a ".gguf" path. `from_cache` (optional) is set when the
 * result came from the tune cache and no grid ran. */
inline lembed_status_t sparse_autotune_run(const char* model_name,
                                           const std::vector<std::string>& corpus,
                                           lembed_autotune_mode_t mode,
                                           lembed_objective_t objective,
                                           double max_latency_ms,
                                           lembed_unified_tuning_result_t* result,
                                           bool* from_cache = nullptr) {
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = make_unified_result(LEMBED_TASK_SPARSE);
    if (from_cache) *from_cache = false;

    if (sparse_gguf_entry(model_name))
        return sparse_gguf_autotune_run(model_name, corpus, mode, objective, max_latency_ms,
                                        result, from_cache);

    int idx = lembed_resolve_sparse_model(model_name);
    if (idx < 0 && sparse_is_gguf_path(model_name))
        return sparse_gguf_autotune_run(model_name, corpus, mode, objective, max_latency_ms,
                                        result, from_cache);
    if (idx < 0)
        return autotune_fail(LEMBED_ERROR_MODEL_NOT_FOUND,
                             std::string("sparse_autotune: unknown sparse model '") + model_name +
                                 "'");
    std::string canon = canonical_sparse_id(model_name);

    std::vector<std::string> texts =
        corpus.empty() ? synthetic_text_corpus(sparse_tune_corpus_size(mode)) : corpus;

    char variant[64];
    std::snprintf(variant, sizeof(variant), "%s_p95_%.3f", corpus.empty() ? "synthetic" : "custom",
                  max_latency_ms);
    autotune_cache_identity id;
    id.model = canon;
    id.variant = variant;
    id.corpus = corpus_fingerprint(texts);
    id.objective = (int)objective;
    id.mode = (int)mode;

    if (read_unified_cache(sparse_cache_subdir(), id, *result)) {
        if (from_cache) *from_cache = true;
        return LEMBED_OK;
    }

    autotune_search_options sopts;
    sopts.objective = objective;
    sopts.constraints.max_p95_ms = max_latency_ms;
    sopts.protocol = autotune_protocol_for(mode);
    sopts.label = "sparse_autotune";

    lembed_sparse_model_t model = (lembed_sparse_model_t)idx;
    sparse_session_cache sessions(lembed_sparse_text_embedding_free);
    auto search = autotune_grid_search(
        sparse_tune_grid(mode),
        [&](const sparse_tune_config& c, const autotune_protocol& p) {
            return bench_sparse_config(model, texts, c, p, sessions);
        },
        describe_sparse_config, sopts);
    sessions.reset();
    if (!search.found()) return autotune_fail_no_config("sparse_autotune", canon);

    lembed_sparse_options_t defaults = lembed_sparse_options_default();
    result->threads = search.best_config().threads;
    result->batch_size = search.best_config().batch;
    result->top_k = defaults.top_k;
    result->min_weight = defaults.min_weight;
    result->storage_format = defaults.storage_format;
    set_result_metrics(*result, search.best_measurement());

    write_unified_cache(sparse_cache_subdir(), id, *result);
    return LEMBED_OK;
}

inline lembed_sparse_tuning_result_t to_sparse_result(const lembed_unified_tuning_result_t& u) {
    lembed_sparse_tuning_result_t r = {0};
    r.top_k = u.top_k;
    r.min_weight = u.min_weight;
    r.storage_format = u.storage_format;
    r.threads = u.threads;
    r.batch_size = u.batch_size;
    r.throughput_docs_sec = u.throughput_docs_sec;
    r.latency_ms = u.latency_ms;
    r.memory_mb = u.memory_mb;
    return r;
}

extern "C" lembed_status_t lembed_sparse_autotune_v2(const char* model_name,
                                                     lembed_autotune_mode_t mode,
                                                     lembed_sparse_tuning_result_v2_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = lembed_sparse_tuning_result_v2_t{};
    {
        lembed_status_t es = check_autotune_mode(mode);
        if (es != LEMBED_OK) return es;
    }
    return autotune_guard("sparse_autotune", [&]() {
        lembed_unified_tuning_result_t u;
        bool hit = false;
        lembed_status_t s =
            sparse_autotune_run(model_name, {}, mode, sparse_default_objective(), 0.0, &u, &hit);
        if (s == LEMBED_OK) {
            result->base = to_sparse_result(u);
            result->workers = u.workers;
            result->max_batch_tokens = u.max_tokens;
            result->from_cache = hit ? 1 : 0;
        }
        return s;
    });
}

extern "C" lembed_status_t lembed_sparse_autotune(const char* model_name,
                                                  lembed_autotune_mode_t mode,
                                                  lembed_sparse_tuning_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    {
        lembed_status_t es = check_autotune_mode(mode);
        if (es != LEMBED_OK) {
            *result = lembed_sparse_tuning_result_t{};
            return es;
        }
    }
    lembed_sparse_tuning_result_v2_t v2;
    lembed_status_t s = lembed_sparse_autotune_v2(model_name, mode, &v2);
    *result = v2.base;
    return s;
}

/* NULL clears every sparse entry; otherwise the model's entries only. */
inline void lembed_sparse_autotune_clear_cache(const char* model_name) {
    autotune_guard("sparse_autotune_clear_cache", [&]() {
        if (!model_name) clear_autotune_cache(nullptr, sparse_cache_subdir());
        else clear_autotune_cache(canonical_sparse_id(model_name).c_str(), sparse_cache_subdir());
        return LEMBED_OK;
    });
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP */
