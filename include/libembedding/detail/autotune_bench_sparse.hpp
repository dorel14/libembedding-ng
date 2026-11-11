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
 * Version: 1.13.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_SPARSE_HPP

#include "libembedding/autotuner.h"
#include "libembedding/sparse_text_embedding.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"
#include "autotune_corpus.hpp"
#include "autotune_result.hpp"
#include "autotune_search.hpp"
#include "autotune_session.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
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

/* Canonical registry code, so both name forms share entries and purges. */
inline std::string canonical_sparse_id(const char* model) {
    if (!model) return std::string();
    int idx = lembed_resolve_sparse_model(model);
    if (idx < 0) return std::string(model);
    return std::string(lembed__sparse_models[idx].model_code);
}

/* =========================================================================
 * Tuner
 * ========================================================================= */

inline lembed_objective_t sparse_default_objective() { return LEMBED_OBJECTIVE_THROUGHPUT; }

/* `corpus` empty = synthetic corpus. max_latency_ms bounds the p95 of one call
 * over the corpus (0 = none). */
inline lembed_status_t sparse_autotune_run(const char* model_name,
                                           const std::vector<std::string>& corpus,
                                           lembed_autotune_mode_t mode,
                                           lembed_objective_t objective,
                                           double max_latency_ms,
                                           lembed_unified_tuning_result_t* result) {
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = make_unified_result(LEMBED_TASK_SPARSE);

    int idx = lembed_resolve_sparse_model(model_name);
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

    if (read_unified_cache(sparse_cache_subdir(), id, *result)) return LEMBED_OK;

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

extern "C" lembed_status_t lembed_sparse_autotune(const char* model_name,
                                                  lembed_autotune_mode_t mode,
                                                  lembed_sparse_tuning_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = lembed_sparse_tuning_result_t{0};
    return autotune_guard("sparse_autotune", [&]() {
        lembed_unified_tuning_result_t u;
        lembed_status_t s =
            sparse_autotune_run(model_name, {}, mode, sparse_default_objective(), 0.0, &u);
        if (s == LEMBED_OK) *result = to_sparse_result(u);
        return s;
    });
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
