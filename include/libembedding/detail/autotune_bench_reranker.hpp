/*
 * libembedding - detail/autotune_bench_reranker.hpp
 * Reranker auto-tuner
 *
 * Tunes threads x batch_size x max_tokens for one reranker request: a query
 * scored against a fixed set of documents. max_tokens truncates documents, so
 * it trades ranking quality for speed; callers that care set a floor through
 * the constrained entry point (min_tokens).
 *
 * Every entry point -- default, constrained, latency budget, profile, custom
 * corpus -- runs the same search (autotune_search.hpp) and differs only by its
 * grid filter, constraints and cache identity.
 *
 * Auteur: David Orel
 * Version: 1.13.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP

#include "libembedding/autotuner.h"
#include "libembedding/reranker.h"
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
 * Workload
 * ========================================================================= */

/* Every configuration sees the same 256-token documents; only truncation
 * varies. A corpus rebuilt per max_tokens would make configurations
 * incomparable and bias the tuner toward the smallest max_tokens. */
inline int reranker_bench_corpus_tokens() { return 256; }
inline int reranker_bench_corpus_docs() { return 20; }
inline const char* reranker_bench_query() { return "What is deep learning?"; }

inline std::vector<std::string> generate_synthetic_corpus(int n_docs) {
    std::vector<std::string> docs;
    docs.reserve(n_docs > 0 ? (size_t)n_docs : 0);
    for (int i = 0; i < n_docs; i++)
        docs.push_back(synthetic_reranker_doc(reranker_bench_corpus_tokens(), i + 1));
    return docs;
}

/* =========================================================================
 * Configuration space
 * ========================================================================= */

struct reranker_tune_config {
    int threads;
    int batch;
    int max_tokens;
};

inline std::string describe_reranker_config(const reranker_tune_config& c) {
    return "threads=" + std::to_string(c.threads) + " batch=" + std::to_string(c.batch) +
           " tokens=" + std::to_string(c.max_tokens);
}

/* Thread counts derived from the machine instead of a fixed {1, 4, 8}.
 * QUICK: every physical core, every logical core. A single request never gets
 * faster with fewer threads on an idle CPU, and the single-threaded points were
 * the slowest of the grid (most of the QUICK budget). FULL adds 1, 2 and half
 * the cores, for hosts that run several rerankers side by side. */
inline std::vector<int> reranker_thread_options(lembed_autotune_mode_t mode) {
    int physical = std::max(1, cpu_physical_cores());
    int logical = std::max(physical, cpu_logical_cores());
    std::vector<int> t;
    if (mode == LEMBED_AUTOTUNE_FULL) {
        t.push_back(1);
        t.push_back(2);
        t.push_back(physical / 2);
    }
    t.push_back(physical);
    t.push_back(logical);
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    t.erase(std::remove_if(t.begin(), t.end(), [](int v) { return v < 1; }), t.end());
    return t;
}

/* Grid, optionally restricted to max_tokens >= min_tokens. When the floor is
 * above every candidate, the floor itself becomes the only token setting.
 *
 * Order matters twice:
 *   - threads, then tokens, then batch: points sharing (threads, max_tokens)
 *     are consecutive, so they reuse one session (autotune_session.hpp);
 *   - tokens from the largest: ties are resolved toward the earliest point, so
 *     between two statistically equal configurations the one that truncates
 *     less (better ranking) wins. */
inline std::vector<reranker_tune_config> reranker_tune_grid(lembed_autotune_mode_t mode,
                                                            int min_tokens = 0) {
    std::vector<int> batches = (mode == LEMBED_AUTOTUNE_QUICK) ? std::vector<int>{4, 16}
                                                               : std::vector<int>{4, 8, 16};
    std::vector<int> tokens = (mode == LEMBED_AUTOTUNE_QUICK) ? std::vector<int>{256, 64}
                                                              : std::vector<int>{256, 128, 64, 32};
    std::vector<int> kept;
    for (int k : tokens)
        if (k >= min_tokens) kept.push_back(k);
    if (kept.empty()) kept.push_back(min_tokens);

    std::vector<reranker_tune_config> grid;
    for (int t : reranker_thread_options(mode))
        for (int k : kept)
            for (int b : batches) grid.push_back({t, b, k});
    return grid;
}

/* =========================================================================
 * Measuring one configuration
 * ========================================================================= */

/* Session-level settings of a reranker: what cannot change per call. */
struct reranker_session_key {
    int threads = 0;
    int max_tokens = 0;
    bool operator==(const reranker_session_key& o) const {
        return threads == o.threads && max_tokens == o.max_tokens;
    }
};
using reranker_session_cache = autotune_session_cache<reranker_session_key, lembed_reranker_t>;

/* Times one configuration; the session comes from `sessions`, so consecutive
 * points that differ only by batch size share it. */
inline autotune_measurement bench_reranker_config(lembed_reranker_model_t model,
                                                  const std::vector<std::string>& docs,
                                                  const char* query,
                                                  const reranker_tune_config& c,
                                                  const autotune_protocol& proto,
                                                  reranker_session_cache& sessions) {
    autotune_measurement failed;
    if (docs.empty()) return failed;

    lembed_reranker_t* ctx = sessions.get({c.threads, c.max_tokens}, [&]() -> lembed_reranker_t* {
        lembed_reranker_options_t opts = lembed_reranker_options_default();
        opts.model = model;
        opts.num_threads = c.threads;
        opts.batch_size = c.batch;
        opts.max_length = c.max_tokens;
        opts.show_download_progress = 0;
        lembed_reranker_t* created = nullptr;
        return lembed_reranker_create(&opts, &created) == LEMBED_OK ? created : nullptr;
    });
    if (!ctx) return failed;

    std::vector<const char*> c_docs = corpus_c_strings(docs);
    const int n = (int)c_docs.size();
    return autotune_measure_calls(proto, n, sessions.rss_before_mb(), [&]() {
        lembed_rerank_results_t r = {0};
        bool ok = lembed_reranker_rerank(ctx, query, c_docs.data(), n, c.batch, &r) == LEMBED_OK;
        if (ok) lembed_rerank_results_free(&r);
        return ok;
    });
}

/* =========================================================================
 * Cache
 * ========================================================================= */

inline const char* reranker_cache_subdir() { return "reranker"; }

/* Identity of a reranker tuning request: every dimension that changes the
 * optimum (corpus, objective, mode, constraints via `variant`). */
inline autotune_cache_identity reranker_cache_identity(const char* model_name, const char* variant,
                                                       const std::string& corpus,
                                                       lembed_objective_t objective,
                                                       lembed_autotune_mode_t mode) {
    autotune_cache_identity id;
    id.model = model_name ? model_name : "";
    id.variant = variant ? variant : "";
    id.corpus = corpus;
    id.objective = (int)objective;
    id.mode = (int)mode;
    return id;
}

/* Variant tag of a constrained request, so each (floor, budget) pair has its
 * own entry. The unconstrained request keeps the historical "default" tag. */
inline std::string reranker_variant(const char* base, int min_tokens, double max_latency_ms) {
    if (min_tokens <= 0 && max_latency_ms <= 0.0) return base;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s_tok%d_p95_%.3f", base, min_tokens, max_latency_ms);
    return buf;
}

/* Canonical registry code for a reranker named by either form, so the repo
 * and the canonical name share cache entries and clear_cache() purges both.
 * Unknown models keep the caller string; NULL gives an empty string. */
inline std::string canonical_reranker_id(const char* model) {
    if (!model) return std::string();
    int idx = lembed_resolve_reranker_model(model);
    if (idx < 0) return std::string(model);
    return std::string(lembed__reranker_models[idx].model_code);
}

/* =========================================================================
 * Tuner
 * ========================================================================= */

/* One reranker tuning request, fully described. */
struct reranker_tune_request {
    const char* model_name = nullptr;          /* repo or canonical name */
    lembed_autotune_mode_t mode = LEMBED_AUTOTUNE_QUICK;
    lembed_objective_t objective = LEMBED_OBJECTIVE_BALANCED;
    int min_tokens = 0;                        /* quality floor, 0 = none */
    double max_latency_ms = 0.0;               /* p95 budget, 0 = none */
    const std::vector<std::string>* docs = nullptr; /* nullptr = synthetic */
    const char* query = nullptr;               /* required with docs */
    const char* label = "reranker_autotune";
};

/* Runs (or reads from cache) one request. When constraints leave nothing
 * feasible, the unconstrained optimum of the same measurements is returned, as
 * the API has always done; p95_latency_ms tells the caller by how much. */
inline lembed_status_t reranker_autotune_run(const reranker_tune_request& req,
                                             lembed_unified_tuning_result_t* result) {
    if (!req.model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = make_unified_result(LEMBED_TASK_RERANKING);

    int idx = lembed_resolve_reranker_model(req.model_name);
    if (idx < 0)
        return autotune_fail(LEMBED_ERROR_MODEL_NOT_FOUND,
                             std::string(req.label) + ": unknown reranker model '" +
                                 req.model_name + "'");
    std::string canon = canonical_reranker_id(req.model_name);

    std::vector<std::string> synthetic;
    const std::vector<std::string>* docs = req.docs;
    const char* query = req.query;
    if (!docs) {
        synthetic = generate_synthetic_corpus(reranker_bench_corpus_docs());
        docs = &synthetic;
        query = reranker_bench_query();
    }
    if (docs->empty() || !query)
        return autotune_fail(LEMBED_ERROR_INVALID_ARGUMENT,
                             std::string(req.label) + ": a reranking workload needs a query and documents");

    /* The query is part of the workload, so it is part of the fingerprint. */
    std::vector<std::string> workload = *docs;
    workload.push_back(query);
    std::string variant = reranker_variant(req.docs ? "custom" : "default", req.min_tokens,
                                           req.max_latency_ms);
    autotune_cache_identity id = reranker_cache_identity(
        canon.c_str(), variant.c_str(), corpus_fingerprint(workload), req.objective, req.mode);

    if (read_unified_cache(reranker_cache_subdir(), id, *result)) return LEMBED_OK;

    autotune_search_options sopts;
    sopts.objective = req.objective;
    sopts.constraints.max_p95_ms = req.max_latency_ms;
    sopts.protocol = autotune_protocol_for(req.mode);
    sopts.label = req.label;

    lembed_reranker_model_t model = (lembed_reranker_model_t)idx;
    reranker_session_cache sessions(lembed_reranker_free);
    auto search = autotune_grid_search(
        reranker_tune_grid(req.mode, req.min_tokens),
        [&](const reranker_tune_config& c, const autotune_protocol& p) {
            return bench_reranker_config(model, *docs, query, c, p, sessions);
        },
        describe_reranker_config, sopts);
    sessions.reset();

    if (!search.found()) return autotune_fail_no_config(req.label, canon);

    const reranker_tune_config& best = search.best_config();
    result->threads = best.threads;
    result->batch_size = best.batch;
    result->max_tokens = best.max_tokens;
    set_result_metrics(*result, search.best_measurement());

    write_unified_cache(reranker_cache_subdir(), id, *result);
    return LEMBED_OK;
}

inline lembed_reranker_tuning_result_t to_reranker_result(const lembed_unified_tuning_result_t& u) {
    lembed_reranker_tuning_result_t r = {0};
    r.threads = u.threads;
    r.batch_size = u.batch_size;
    r.max_tokens = u.max_tokens;
    r.throughput_docs_sec = u.throughput_docs_sec;
    r.latency_ms = u.latency_ms;
    r.memory_mb = u.memory_mb;
    r.p95_latency_ms = u.p95_latency_ms;
    return r;
}

/* Runs a request and converts to the reranker struct; exceptions become a
 * status (autotune_guard): nothing may unwind through the C ABI. */
inline lembed_status_t reranker_autotune_legacy(const reranker_tune_request& req,
                                                lembed_reranker_tuning_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = lembed_reranker_tuning_result_t{0};
    return autotune_guard(req.label, [&]() {
        lembed_unified_tuning_result_t u;
        lembed_status_t s = reranker_autotune_run(req, &u);
        if (s == LEMBED_OK) *result = to_reranker_result(u);
        return s;
    });
}

/* =========================================================================
 * C entry points (thin facades over reranker_autotune_run)
 * ========================================================================= */

extern "C" lembed_status_t lembed_reranker_autotune(const char* model_name,
                                                    lembed_autotune_mode_t mode,
                                                    lembed_objective_t objective,
                                                    lembed_reranker_tuning_result_t* result) {
    reranker_tune_request req;
    req.model_name = model_name;
    req.mode = mode;
    req.objective = objective;
    return reranker_autotune_legacy(req, result);
}

extern "C" lembed_status_t lembed_reranker_autotune_constrained(
    const char* model_name, lembed_autotune_mode_t mode, lembed_objective_t objective,
    int min_tokens, double max_latency_ms, lembed_reranker_tuning_result_t* result) {
    reranker_tune_request req;
    req.model_name = model_name;
    req.mode = mode;
    req.objective = objective;
    req.min_tokens = min_tokens;
    req.max_latency_ms = max_latency_ms;
    return reranker_autotune_legacy(req, result);
}

/* Best configuration whose p95 fits `target_latency_ms`, ranked by `objective`
 * (QUICK grid). Falls back to the unconstrained tuning when nothing fits. */
extern "C" lembed_status_t lembed_reranker_auto_config(const char* model_name,
                                                       double target_latency_ms,
                                                       lembed_objective_t objective,
                                                       lembed_reranker_tuning_result_t* result) {
    return lembed_reranker_autotune_constrained(model_name, LEMBED_AUTOTUNE_QUICK, objective, 0,
                                                target_latency_ms, result);
}

/* Latency budget per profile: interactive 100 ms, balanced 300 ms, quality 1 s. */
inline double reranker_profile_budget_ms(lembed_reranker_profile_t profile) {
    switch (profile) {
        case LEMBED_PROFILE_INTERACTIVE: return 100.0;
        case LEMBED_PROFILE_QUALITY:     return 1000.0;
        case LEMBED_PROFILE_BALANCED:
        default:                         return 300.0;
    }
}

extern "C" lembed_status_t lembed_reranker_auto_config_profile(
    const char* model_name, lembed_reranker_profile_t profile,
    lembed_reranker_tuning_result_t* result) {
    return lembed_reranker_auto_config(model_name, reranker_profile_budget_ms(profile),
                                       LEMBED_OBJECTIVE_BALANCED, result);
}

/* NULL clears the whole reranker cache; any other value is resolved to the
 * registry code the entries are keyed on. */
inline void lembed_reranker_autotune_clear_cache(const char* model_name) {
    autotune_guard("reranker_autotune_clear_cache", [&]() {
        if (!model_name) clear_autotune_cache(nullptr, reranker_cache_subdir());
        else clear_autotune_cache(canonical_reranker_id(model_name).c_str(), reranker_cache_subdir());
        return LEMBED_OK;
    });
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP */
