/*
 * libembedding - detail/autotune_bench_text.hpp
 * Text embedding auto-tuner
 *
 * Tunes how a dense text model is driven on this machine:
 *   workers  -- independent sessions embedding disjoint shards in parallel
 *   threads  -- ORT intra-op threads per session
 *   batch    -- texts per inference call
 *
 * Only the task-specific parts live here (corpus, grid, how one configuration
 * is measured, cache identity). Timing, scoring and selection come from the
 * shared engine: autotune_measure.hpp, autotune_objective.hpp,
 * autotune_search.hpp; the result layout from autotune_result.hpp.
 *
 * One timed call embeds the whole tuning corpus across all workers, so
 * throughput is texts/s of the full configuration, and latency_ms /
 * p95_latency_ms keep their historical meaning for this task: call time
 * amortised per text.
 *
 * Auteur: David Orel
 * Version: 1.13.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_HPP

#include "libembedding/autotuner.h"
#include "libembedding/text_embedding.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"
#include "autotune_corpus.hpp"
#include "autotune_result.hpp"
#include "autotune_search.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lembed { namespace detail {

/* =========================================================================
 * Configuration space
 * ========================================================================= */

struct text_tune_config {
    int workers;
    int threads;
    int batch;
};

inline std::string describe_text_config(const text_tune_config& c) {
    return "workers=" + std::to_string(c.workers) + " threads=" + std::to_string(c.threads) +
           " batch=" + std::to_string(c.batch);
}

/* Parallel splits of the physical cores between sessions and intra-op threads.
 *
 * ORT's intra-op pool scales with physical cores; hyper-threads mostly compete
 * for the same execution units, so splits are built on physical cores, plus a
 * single "all logical cores" session (FULL) to let SMT prove itself. */
inline std::vector<std::pair<int, int>> text_parallel_splits(lembed_autotune_mode_t mode) {
    int physical = std::max(1, cpu_physical_cores());
    int logical = std::max(physical, cpu_logical_cores());
    std::vector<std::pair<int, int>> splits;
    auto add = [&](int w, int t) {
        w = std::max(1, w);
        t = std::max(1, t);
        for (const auto& s : splits)
            if (s.first == w && s.second == t) return;
        splits.push_back({w, t});
    };
    add(1, physical);                    /* one session, every core */
    if (physical >= 4) add(2, physical / 2);
    add(physical, 1);                    /* one single-threaded session per core */
    if (mode == LEMBED_AUTOTUNE_FULL) {
        if (logical > physical) add(1, logical);
        if (physical >= 8) add(4, physical / 4);
    }
    return splits;
}

/* QUICK measures the parallel shapes at one batch size; FULL crosses every
 * shape with the batch sizes that matter on CPU. */
inline std::vector<text_tune_config> text_tune_grid(lembed_autotune_mode_t mode) {
    std::vector<int> batches = (mode == LEMBED_AUTOTUNE_QUICK)
                                   ? std::vector<int>{32}
                                   : std::vector<int>{16, 32, 64, 128};
    std::vector<text_tune_config> grid;
    for (const auto& s : text_parallel_splits(mode))
        for (int b : batches) grid.push_back({s.first, s.second, b});
    return grid;
}

/* Synthetic corpus size: every worker sees at least a few texts in QUICK and
 * more than one batch in FULL, while QUICK stays within seconds. */
inline int text_tune_corpus_size(lembed_autotune_mode_t mode) {
    return (mode == LEMBED_AUTOTUNE_QUICK) ? 32 : 256;
}

/* =========================================================================
 * Measuring one configuration
 * ========================================================================= */

/* Creates `c.workers` sessions of `model`, then times one call = every worker
 * embedding its shard of `corpus` concurrently. Sessions are created before the
 * clock starts: load time is not what is being tuned. */
inline autotune_measurement bench_text_config(lembed_text_model_t model,
                                              const std::vector<std::string>& corpus,
                                              const text_tune_config& c,
                                              const autotune_protocol& proto) {
    autotune_measurement failed;
    if (corpus.empty() || c.workers <= 0) return failed;

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = model;
    opts.num_threads = c.threads;
    opts.batch_size = c.batch;
    opts.offline = 1;
    opts.show_download_progress = 0;

    /* Owned sessions: freed on every exit path, exceptions included. */
    struct session_deleter {
        void operator()(lembed_text_embedding_t* s) const { lembed_text_embedding_free(s); }
    };
    using session_ptr = std::unique_ptr<lembed_text_embedding_t, session_deleter>;

    double rss_before = autotune_current_rss_mb();
    std::vector<session_ptr> sessions;
    sessions.reserve((size_t)c.workers);
    for (int i = 0; i < c.workers; i++) {
        lembed_text_embedding_t* s = nullptr;
        if (lembed_text_embedding_create(&opts, &s) != LEMBED_OK) return failed;
        sessions.emplace_back(s);
    }

    std::vector<const char*> texts = corpus_c_strings(corpus);
    const int n = (int)texts.size();
    const int per = n / c.workers;

    auto embed_shard = [&](int w) -> bool {
        int start = w * per;
        int count = (w == c.workers - 1) ? n - start : per;
        if (count <= 0) return true;
        lembed_embeddings_t r = {0};
        /* batch_size = 0: use the session's configured batch, i.e. c.batch. */
        bool ok = lembed_text_embedding_embed(sessions[(size_t)w].get(), texts.data() + start,
                                              count, 0, &r) == LEMBED_OK;
        lembed_embeddings_free(&r);
        return ok;
    };

    auto one_call = [&]() -> bool {
        if (c.workers == 1) return embed_shard(0);
        std::vector<char> ok((size_t)c.workers, 0);
        std::vector<std::thread> pool;
        pool.reserve((size_t)c.workers);
        /* Joins whatever was started, even if a later emplace_back throws:
         * destroying a joinable std::thread would call std::terminate. */
        struct joiner {
            std::vector<std::thread>& threads;
            ~joiner() {
                for (auto& t : threads)
                    if (t.joinable()) t.join();
            }
        } join_all{pool};
        for (int w = 0; w < c.workers; w++)
            pool.emplace_back([&, w]() { ok[(size_t)w] = embed_shard(w) ? 1 : 0; });
        for (auto& t : pool) t.join();
        return std::all_of(ok.begin(), ok.end(), [](char v) { return v != 0; });
    };

    return autotune_measure_calls(proto, n, rss_before, one_call);
}

/* =========================================================================
 * Tuner
 * ========================================================================= */

/* Text entries stay at the cache root, where earlier releases put them. */
inline const char* text_cache_subdir() { return ""; }

/* Embedding is a batch workload: throughput is what the legacy entry points
 * optimised, and remains the default. */
inline lembed_objective_t text_default_objective() { return LEMBED_OBJECTIVE_THROUGHPUT; }

/* One text tuning request. An empty corpus selects the synthetic one.
 * max_latency_ms bounds the p95 of one call over the corpus (0 = none). */
inline lembed_status_t text_autotune_run(lembed_text_model_t model,
                                         const std::vector<std::string>& corpus,
                                         lembed_autotune_mode_t mode,
                                         lembed_objective_t objective,
                                         double max_latency_ms,
                                         lembed_unified_tuning_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = make_unified_result(LEMBED_TASK_EMBEDDING);

    lembed_model_info_t info;
    if (lembed_get_text_model_info(model, &info) != LEMBED_OK)
        return autotune_fail(LEMBED_ERROR_MODEL_NOT_FOUND, "autotune: unknown text model index");
    const char* model_code = info.model_code;

    std::vector<std::string> bench_corpus =
        corpus.empty() ? synthetic_text_corpus(text_tune_corpus_size(mode)) : corpus;

    /* The benchmarked corpus, the objective and the budget are all part of the
     * identity: a tuning is never served for a request it does not answer. */
    std::string variant = corpus.empty() ? "synthetic" : "custom";
    if (max_latency_ms > 0.0) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "_p95_%.3f", max_latency_ms);
        variant += buf;
    }
    autotune_cache_identity id;
    id.model = model_code ? model_code : "";
    id.variant = variant;
    id.corpus = corpus_fingerprint(bench_corpus);
    id.objective = (int)objective;
    id.mode = (int)mode;

    if (read_unified_cache(text_cache_subdir(), id, *result)) return LEMBED_OK;

    std::fprintf(stderr, "autotune: cache miss for %s, running benchmark...\n", model_code);

    autotune_search_options sopts;
    sopts.objective = objective;
    sopts.constraints.max_p95_ms = max_latency_ms;
    sopts.protocol = autotune_protocol_for(mode);
    sopts.label = "autotune";

    auto search = autotune_grid_search(
        text_tune_grid(mode),
        [&](const text_tune_config& c, const autotune_protocol& p) {
            return bench_text_config(model, bench_corpus, c, p);
        },
        describe_text_config, sopts);

    /* Nothing could be measured: report the failure, never an unmeasured default. */
    if (!search.found()) return autotune_fail_no_config("autotune", id.model);

    const text_tune_config& best = search.best_config();
    const autotune_measurement& m = search.best_measurement();
    result->workers = best.workers;
    result->threads = best.threads;
    result->batch_size = best.batch;
    set_result_metrics(*result, m);
    if (m.items_per_call > 0) {
        result->latency_ms /= m.items_per_call;
        result->p95_latency_ms /= m.items_per_call;
    }

    write_unified_cache(text_cache_subdir(), id, *result);
    return LEMBED_OK;
}

inline lembed_tuning_result_t to_text_result(const lembed_unified_tuning_result_t& u) {
    lembed_tuning_result_t r = {0};
    r.workers = u.workers;
    r.threads = u.threads;
    r.batch_size = u.batch_size;
    r.throughput_docs_sec = u.throughput_docs_sec;
    r.latency_ms = u.latency_ms;
    r.memory_mb = u.memory_mb;
    return r;
}

/* Legacy entry used by lembed_autotune / lembed_autotune_custom: default
 * objective, no latency budget. */
inline lembed_status_t autotune_text_impl(lembed_text_model_t model,
                                          const std::vector<std::string>& corpus,
                                          lembed_autotune_mode_t mode,
                                          lembed_tuning_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = lembed_tuning_result_t{0};
    return autotune_guard("autotune", [&]() {
        lembed_unified_tuning_result_t u;
        lembed_status_t s =
            text_autotune_run(model, corpus, mode, text_default_objective(), 0.0, &u);
        if (s == LEMBED_OK) *result = to_text_result(u);
        return s;
    });
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_HPP */
