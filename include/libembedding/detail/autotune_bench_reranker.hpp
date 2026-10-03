/*
 * libembedding - detail/autotune_bench_reranker.hpp
 * Reranker auto-tuner
 *
 * Auteur: David Orel
 * Version: 1.10.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP

#include "libembedding/autotuner.h"
#include "libembedding/reranker.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Forward declarations */
static lembed_status_t lembed_reranker_autotune_impl(
    const char* model_name,
    const std::vector<std::string>& docs,
    const char* query,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    const autotune_cache_identity& id,
    const char* label,
    lembed_reranker_tuning_result_t* result);
void write_reranker_cache(const char* model_name, const autotune_cache_identity& id,
                          const lembed_reranker_tuning_result_t& result);
bool read_reranker_cache(const char* model_name, const autotune_cache_identity& id,
                         lembed_reranker_tuning_result_t* out);

/* Generate synthetic documents with target token count */
inline std::string generate_synthetic_doc(int target_tokens, int seed) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "data", "model", "system", "algorithm",
        "search", "query", "document", "text", "information", "result",
        "process", "analysis", "method", "approach", "technique",
        "application", "performance", "evaluation", "research", "study",
    };
    int n_words = sizeof(words) / sizeof(words[0]);

    std::string doc;
    int target_words = std::max(1, (int)(target_tokens * 0.75));
    for (int i = 0; i < target_words; i++) {
        if (i > 0) doc += " ";
        doc += words[(seed + i) % n_words];
    }
    return doc;
}

/* Get cache directory for reranker autotune */
inline std::string reranker_autotune_cache_dir() {
    return autotune_cache_dir() + "/reranker";
}

/* Identity of a reranker tuning request.
 *
 * Every dimension that changes the optimal configuration is part of it, so a
 * cached tuning can never be served for a request it does not answer: the
 * corpus (custom tunings), the objective and the mode. */
inline autotune_cache_identity reranker_cache_identity(
    const char* model_name, const char* variant, const std::string& corpus,
    lembed_objective_t objective, lembed_autotune_mode_t mode) {
    autotune_cache_identity id;
    id.model = model_name ? model_name : "";
    id.variant = variant ? variant : "";
    id.corpus = corpus;
    id.objective = (int)objective;
    id.mode = (int)mode;
    return id;
}

/* Get cache file path for a reranker tuning request */
inline std::string get_reranker_cache_path(const autotune_cache_identity& id) {
    std::string variant = cache_variant_key(id.variant.c_str(), id.corpus, id.objective, id.mode);
    return get_cache_path(id.model.c_str(), "reranker", variant);
}

/* Write reranker tune result to cache (atomically) */
inline void write_reranker_cache(const char* model_name, const autotune_cache_identity& id,
                                 const lembed_reranker_tuning_result_t& result) {
    std::ostringstream os;
    os << "{\n";
    write_cache_identity(os, id);
    os << "  \"cores\": " << cpu_logical_cores() << ",\n";
    os << "  \"threads\": " << result.threads << ",\n";
    os << "  \"batch_size\": " << result.batch_size << ",\n";
    os << "  \"max_tokens\": " << result.max_tokens << ",\n";
    os << "  \"throughput_docs_sec\": " << result.throughput_docs_sec << ",\n";
    os << "  \"latency_ms\": " << result.latency_ms << ",\n";
    os << "  \"p95_latency_ms\": " << result.p95_latency_ms << ",\n";
    os << "  \"memory_mb\": " << result.memory_mb << "\n";
    os << "}\n";
    write_file_atomic(get_reranker_cache_path(id), os.str());
    (void)model_name;
}

/* Read reranker tune result from cache.
 *
 * Returns false on a miss, on a foreign/legacy entry, and on a partial entry:
 * the caller must be able to trust that *out is fully populated. */
inline bool read_reranker_cache(const char* model_name, const autotune_cache_identity& id,
                                lembed_reranker_tuning_result_t* out) {
    if (!out) return false;
    std::string path = get_reranker_cache_path(id);
    if (!cache_identity_matches(path, id)) return false;

    std::ifstream f(path);
    if (!f.is_open()) return false;

    *out = {0};
    int found = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (json_number(line, "threads", out->threads)) found++;
        else if (json_number(line, "batch_size", out->batch_size)) found++;
        else if (json_number(line, "max_tokens", out->max_tokens)) found++;
        else if (json_number(line, "throughput_docs_sec", out->throughput_docs_sec)) found++;
        else if (json_number(line, "latency_ms", out->latency_ms)) found++;
        else if (json_number(line, "p95_latency_ms", out->p95_latency_ms)) found++;
        else if (json_number(line, "memory_mb", out->memory_mb)) found++;
    }
    (void)model_name;

    /* A configuration field left at 0 would be handed straight to
     * lembed_reranker_options_t, so require every field to have been read. */
    static const int kExpectedFields = 7;
    if (found < kExpectedFields) {
        *out = {0};
        return false;
    }
    return true;
}

/* Reference corpus used by every configuration.
 *
 * The corpus must NOT be rebuilt per configuration: document length drives
 * latency, so a corpus that depends on the max_tokens under test would make
 * configurations incomparable and bias the tuner toward the smallest
 * max_tokens. Every configuration now sees the same 256-token documents and
 * only truncation varies. */
inline int reranker_bench_corpus_tokens() { return 256; }

inline const char* reranker_bench_query() { return "What is deep learning?"; }

inline std::vector<std::string> generate_synthetic_corpus(int n_docs) {
    int tokens = reranker_bench_corpus_tokens();
    std::vector<std::string> docs;
    docs.reserve(n_docs > 0 ? (size_t)n_docs : 0);
    for (int i = 0; i < n_docs; i++) {
        /* Seed depends on the document index only, so the corpus is stable
         * across calls and identical for every configuration. */
        docs.push_back(generate_synthetic_doc(tokens, i + 1));
    }
    return docs;
}

/* Benchmark a single reranker configuration */
inline lembed_reranker_tuning_result_t bench_reranker_config(
    const char* model_name,
    int threads,
    int batch_size,
    int max_tokens,
    const std::vector<std::string>& docs,
    const char* query,
    int warmup_iters,
    int bench_iters)
{
    lembed_reranker_tuning_result_t res = {0};
    res.threads = threads;
    res.batch_size = batch_size;
    res.max_tokens = max_tokens;

    if (docs.empty()) {
        res.latency_ms = 999999;
        return res;
    }
    int n_docs = (int)docs.size();

    /* Create reranker */
    lembed_reranker_options_t opts = lembed_reranker_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.max_length = max_tokens;
    opts.show_download_progress = 0;

    /* Resolve model by name or code */
    int model_idx = -1;
    {
        const lembed_model_info_t* models = nullptr;
        int count = 0;
        lembed_list_reranker_models(&models, &count);
        for (int i = 0; i < count; i++) {
            std::string name = models[i].model_name;
            std::string code = models[i].model_code;
            if (model_name == name || model_name == code) {
                model_idx = i;
                break;
            }
        }
    }
    if (model_idx < 0) {
        /* Unknown model: do not silently benchmark a different one. */
        res.latency_ms = 999999;
        return res;
    }
    opts.model = static_cast<lembed_reranker_model_t>(model_idx);

    lembed_reranker_t* ctx = nullptr;
    lembed_status_t s = lembed_reranker_create(&opts, &ctx);
    if (s != LEMBED_OK) {
        res.latency_ms = 999999;
        return res;
    }

    /* Build C string array */
    std::vector<const char*> c_docs;
    for (const auto& d : docs) c_docs.push_back(d.c_str());

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        lembed_rerank_results_t result = {0};
        if (lembed_reranker_rerank(ctx, query, c_docs.data(), n_docs, batch_size, &result) == LEMBED_OK)
            lembed_rerank_results_free(&result);
    }

    /* Benchmark */
    std::vector<double> times;
    times.reserve(bench_iters > 0 ? (size_t)bench_iters : 0);
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        lembed_rerank_results_t result = {0};
        lembed_status_t rs = lembed_reranker_rerank(ctx, query, c_docs.data(), n_docs,
                                                    batch_size, &result);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (rs != LEMBED_OK) continue;
        lembed_rerank_results_free(&result);
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    lembed_reranker_free(ctx);

    /* Every iteration failed: report the sentinel so this configuration is
     * never mistaken for a zero-latency winner. */
    if (times.empty()) {
        res.latency_ms = 999999;
        return res;
    }

    /* Compute stats */
    std::sort(times.begin(), times.end());
    double p50 = percentile_sorted(times, 0.50);
    double p95 = percentile_sorted(times, 0.95);

    res.latency_ms = p50;
    res.p95_latency_ms = p95;
    res.throughput_docs_sec = (p50 > 0) ? (1000.0 / p50) * n_docs : 0;
    res.memory_mb = 0;  /* TODO: measure RSS */

    return res;
}

/* Score a reranker configuration based on objective (lower is better) */
inline double score_reranker_config(const lembed_reranker_tuning_result_t& r, lembed_objective_t obj) {
    switch (obj) {
        case LEMBED_OBJECTIVE_LATENCY:
            return r.latency_ms + (r.p95_latency_ms - r.latency_ms) * 0.5;
        case LEMBED_OBJECTIVE_THROUGHPUT:
            return r.throughput_docs_sec > 0 ? (1.0 / r.throughput_docs_sec) * 1000000 : 999999;
        case LEMBED_OBJECTIVE_BALANCED:
            return r.latency_ms * 0.5 + (1.0 / (r.throughput_docs_sec + 1)) * 1000;
        case LEMBED_OBJECTIVE_MEMORY:
            return r.memory_mb > 0 ? r.memory_mb : r.latency_ms;
        default:
            return r.latency_ms;
    }
}

/* Main reranker auto-tune implementation, shared by the default and the
 * custom-corpus entry points.
 *
 * `docs` is the corpus that is actually benchmarked and `id` must describe it
 * exactly, so a cached tuning is never served for another corpus, objective or
 * mode. `label` prefixes the progress lines so both callers stay traceable. */
static lembed_status_t lembed_reranker_autotune_impl(
    const char* model_name,
    const std::vector<std::string>& docs,
    const char* query,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    const autotune_cache_identity& id,
    const char* label,
    lembed_reranker_tuning_result_t* result)
{
    /* Check cache first */
    lembed_reranker_tuning_result_t cached = {0};
    if (read_reranker_cache(model_name, id, &cached)) {
        *result = cached;
        return LEMBED_OK;
    }

    int cores = cpu_logical_cores();
    int warmup = 1;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

    /* Configurations to test */
    std::vector<int> threads_vec, batch_vec, tokens_vec;
    if (mode == LEMBED_AUTOTUNE_QUICK) {
        threads_vec = {1, 4, 8};
        batch_vec = {4, 16};
        tokens_vec = {64, 256};
    } else {
        threads_vec = {1, 2, 4, 8};
        batch_vec = {4, 8, 16};
        tokens_vec = {32, 64, 128, 256};
    }

    /* Filter threads > cores */
    std::vector<int> valid_threads;
    for (int t : threads_vec) {
        if (t <= cores) valid_threads.push_back(t);
    }
    if (valid_threads.empty()) valid_threads.push_back(1);

    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total_configs = (int)(valid_threads.size() * batch_vec.size() * tokens_vec.size());

    fprintf(stderr, "%s: testing %d configurations (mode=%s, objective=%d, %d docs)...\n",
            label, total_configs, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL",
            objective, (int)docs.size());

    for (int t : valid_threads) {
        for (int b : batch_vec) {
            for (int k : tokens_vec) {
                auto r = bench_reranker_config(model_name, t, b, k, docs, query, warmup, bench_iters);

                double score = score_reranker_config(r, objective);
                double best_score = score_reranker_config(best, objective);

                if (score < best_score) {
                    best = r;
                }
            }
        }
    }

    /* Every configuration failed: do not cache the sentinel, and do not report
     * a tuning the caller would use as if it were valid. */
    if (best.latency_ms >= 999999) {
        fprintf(stderr, "%s: no configuration could be benchmarked\n", label);
        *result = {0};
        return LEMBED_ERROR_ONNX_RUNTIME;
    }

    fprintf(stderr, "%s: best config: threads=%d batch=%d tokens=%d (P50=%.1fms, P95=%.1fms)\n",
            label, best.threads, best.batch_size, best.max_tokens, best.latency_ms,
            best.p95_latency_ms);

    write_reranker_cache(model_name, id, best);

    *result = best;
    return LEMBED_OK;
}

/* Identity of the default (synthetic corpus) tuning. */
inline autotune_cache_identity default_reranker_identity(
    const char* model_name, const std::vector<std::string>& docs,
    lembed_objective_t objective, lembed_autotune_mode_t mode) {
    return reranker_cache_identity(model_name, "default", corpus_fingerprint(docs),
                                   objective, mode);
}

/* Canonical registry code for a reranker named by either form.
 *
 * The benchmark itself resolves model_name or model_code (see
 * bench_reranker_config), but the cache identity was the raw caller string, so
 * a tuning written with the repo was invisible to a clear_cache issued with the
 * canonical name, and the reverse. Keying the identity on the registry code
 * makes both forms interchangeable. Returns an empty string only for NULL. */
inline std::string canonical_reranker_id(const char* model) {
    if (!model) return std::string();
    int idx = lembed_resolve_reranker_model(model);
    if (idx < 0) return std::string(model);
    return std::string(lembed__reranker_models[idx].model_code);
}

/* Main reranker auto-tune function */
extern "C" lembed_status_t lembed_reranker_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        std::vector<std::string> docs = generate_synthetic_corpus(20);
        if (docs.empty()) return LEMBED_ERROR_ONNX_RUNTIME;
        std::string canon = canonical_reranker_id(model_name);
        autotune_cache_identity id =
            default_reranker_identity(canon.c_str(), docs, objective, mode);
        return lembed_reranker_autotune_impl(canon.c_str(), docs, reranker_bench_query(), mode,
                                             objective, id, "reranker_autotune", result);
    } catch (const std::exception& e) {
        fprintf(stderr, "reranker_autotune: exception: %s\n", e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

/* Reranker auto-tune with constraints (min_tokens, max_latency) */
extern "C" lembed_status_t lembed_reranker_autotune_constrained(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    int min_tokens,
    double max_latency_ms,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    int cores = cpu_logical_cores();
    int warmup = 1;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

    std::vector<std::string> docs = generate_synthetic_corpus(20);
    if (docs.empty()) return LEMBED_ERROR_ONNX_RUNTIME;

    std::vector<int> threads_vec, batch_vec, tokens_vec;
    if (mode == LEMBED_AUTOTUNE_QUICK) {
        threads_vec = {1, 4, 8};
        batch_vec = {4, 16};
        tokens_vec = {64, 256};
    } else {
        threads_vec = {1, 2, 4, 8};
        batch_vec = {4, 8, 16};
        tokens_vec = {32, 64, 128, 256};
    }

    /* Filter by constraints */
    std::vector<int> valid_threads;
    for (int t : threads_vec) {
        if (t <= cores) valid_threads.push_back(t);
    }
    if (valid_threads.empty()) valid_threads.push_back(1);
    std::vector<int> valid_tokens;
    for (int k : tokens_vec) {
        if (k >= min_tokens) valid_tokens.push_back(k);
    }
    if (valid_tokens.empty()) {
        valid_tokens = {min_tokens};
    }

    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total_configs = (int)(valid_threads.size() * batch_vec.size() * valid_tokens.size());

    fprintf(stderr, "reranker_autotune: testing %d configurations (mode=%s, objective=%d, min_tokens=%d, max_latency=%.0fms)...\n",
            total_configs, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL", objective, min_tokens, max_latency_ms);

    for (int t : valid_threads) {
        for (int b : batch_vec) {
            for (int k : valid_tokens) {
                auto r = bench_reranker_config(model_name, t, b, k, docs,
                                                reranker_bench_query(), warmup, bench_iters);

                /* A configuration that failed to run has no p95; the sentinel
                 * latency would otherwise pass the constraint test. */
                if (r.latency_ms >= 999999) continue;

                /* Check latency constraint */
                if (r.p95_latency_ms > max_latency_ms) continue;

                double score = score_reranker_config(r, objective);
                double best_score = score_reranker_config(best, objective);

                if (score < best_score) {
                    best = r;
                }
            }
        }
    }

    if (best.latency_ms >= 999999) {
        fprintf(stderr, "reranker_autotune: no config satisfies constraints, falling back...\n");
        std::string canon = canonical_reranker_id(model_name);
        autotune_cache_identity id = default_reranker_identity(canon.c_str(), docs, objective, mode);
        return lembed_reranker_autotune_impl(canon.c_str(), docs, reranker_bench_query(), mode,
                                             objective, id, "reranker_autotune", result);
    }

    fprintf(stderr, "reranker_autotune: best config: threads=%d batch=%d tokens=%d (P50=%.1fms, P95=%.1fms)\n",
            best.threads, best.batch_size, best.max_tokens, best.latency_ms, best.p95_latency_ms);

    *result = best;
    return LEMBED_OK;
}

/* Auto-configure reranker for target latency */
extern "C" lembed_status_t lembed_reranker_auto_config(
    const char* model_name,
    double target_latency_ms,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int cores = cpu_logical_cores();
    int warmup = 2;
    int bench_iters = 10;

    std::vector<std::string> docs = generate_synthetic_corpus(20);
    if (docs.empty()) return LEMBED_ERROR_ONNX_RUNTIME;

    std::vector<int> threads_vec = {1, 4, 8};
    std::vector<int> batch_vec = {4, 16};
    std::vector<int> tokens_vec = {64, 256};

    for (int t : threads_vec) {
        if (t > cores) continue;
        for (int b : batch_vec) {
            for (int k : tokens_vec) {
                auto r = bench_reranker_config(model_name, t, b, k, docs,
                                                reranker_bench_query(), warmup, bench_iters);
                if (r.latency_ms >= 999999) continue;
                if (r.p95_latency_ms > target_latency_ms) continue;
                if (r.throughput_docs_sec > best.throughput_docs_sec) {
                    best = r;
                }
            }
        }
    }

    if (best.latency_ms >= 999999) {
        std::string canon = canonical_reranker_id(model_name);
        autotune_cache_identity id =
            default_reranker_identity(canon.c_str(), docs, objective, LEMBED_AUTOTUNE_QUICK);
        return lembed_reranker_autotune_impl(canon.c_str(), docs, reranker_bench_query(),
                                             LEMBED_AUTOTUNE_QUICK, objective, id,
                                             "reranker_autotune", result);
    }

    *result = best;
    return LEMBED_OK;
}

/* Profile-based auto-config */
extern "C" lembed_status_t lembed_reranker_auto_config_profile(
    const char* model_name,
    lembed_reranker_profile_t profile,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    double target_ms = 300;
    switch (profile) {
        case LEMBED_PROFILE_INTERACTIVE: target_ms = 100; break;
        case LEMBED_PROFILE_BALANCED:    target_ms = 300; break;
        case LEMBED_PROFILE_QUALITY:     target_ms = 1000; break;
    }

    return lembed_reranker_auto_config(model_name, target_ms, LEMBED_OBJECTIVE_BALANCED, result);
}

/* Clear reranker autotune cache.
 * NULL clears the whole reranker cache, as before. Any other value is resolved
 * to the registry code the entries are keyed on, so the two name forms purge
 * the same entries. An unknown model purges nothing. */
inline void lembed_reranker_autotune_clear_cache(const char* model_name) {
    if (!model_name) {
        clear_autotune_cache(nullptr, "reranker");
        return;
    }
    clear_autotune_cache(canonical_reranker_id(model_name).c_str(), "reranker");
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP */




