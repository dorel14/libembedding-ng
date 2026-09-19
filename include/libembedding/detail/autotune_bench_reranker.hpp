/*
 * libembedding - detail/autotune_bench_reranker.hpp
 * Reranker auto-tuner
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP

#include "libembedding/autotuner.h"
#include "autotune_cache.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Forward declarations */
static lembed_status_t lembed_reranker_autotune_impl(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result);
void write_reranker_cache(const char* model_name, const lembed_reranker_tuning_result_t& result);
bool read_reranker_cache(const char* model_name, lembed_reranker_tuning_result_t* out);

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

/* Get cache file path for a reranker model */
inline std::string get_reranker_cache_path(const char* model_name) {
    std::string dir = reranker_autotune_cache_dir();
    std::filesystem::create_directories(dir);
    int cores = cpu_logical_cores();
    std::string key = std::string(model_name) + "_cores_" + std::to_string(cores);
    return dir + "/" + get_cache_key(key.c_str()) + ".json";
}

/* Write reranker tune result to cache */
inline void write_reranker_cache(const char* model_name, const lembed_reranker_tuning_result_t& result) {
    std::string path = get_reranker_cache_path(model_name);
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "{\n";
    f << "  \"threads\": " << result.threads << ",\n";
    f << "  \"batch_size\": " << result.batch_size << ",\n";
    f << "  \"max_tokens\": " << result.max_tokens << ",\n";
    f << "  \"throughput_docs_sec\": " << result.throughput_docs_sec << ",\n";
    f << "  \"latency_ms\": " << result.latency_ms << ",\n";
    f << "  \"p95_latency_ms\": " << result.p95_latency_ms << ",\n";
    f << "  \"memory_mb\": " << result.memory_mb << "\n";
    f << "}\n";
}

/* Read reranker tune result from cache */
inline bool read_reranker_cache(const char* model_name, lembed_reranker_tuning_result_t* out) {
    std::string path = get_reranker_cache_path(model_name);
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line)) {
        auto find_val = [](const std::string& s, const char* key) -> double {
            std::string search = std::string("\"") + key + "\": ";
            size_t pos = s.find(search);
            if (pos == std::string::npos) return -1;
            return std::stod(s.substr(pos + search.length()));
        };
        if (line.find("\"threads\"") != std::string::npos) out->threads = (int)find_val(line, "threads");
        if (line.find("\"batch_size\"") != std::string::npos) out->batch_size = (int)find_val(line, "batch_size");
        if (line.find("\"max_tokens\"") != std::string::npos) out->max_tokens = (int)find_val(line, "max_tokens");
        if (line.find("\"throughput_docs_sec\"") != std::string::npos) out->throughput_docs_sec = find_val(line, "throughput_docs_sec");
        if (line.find("\"latency_ms\"") != std::string::npos) out->latency_ms = find_val(line, "latency_ms");
        if (line.find("\"p95_latency_ms\"") != std::string::npos) out->p95_latency_ms = find_val(line, "p95_latency_ms");
        if (line.find("\"memory_mb\"") != std::string::npos) out->memory_mb = find_val(line, "memory_mb");
    }
    return true;
}

/* Benchmark a single reranker configuration */
inline lembed_reranker_tuning_result_t bench_reranker_config(
    const char* model_name,
    int threads,
    int batch_size,
    int max_tokens,
    int n_docs,
    int warmup_iters,
    int bench_iters)
{
    lembed_reranker_tuning_result_t res = {0};
    res.threads = threads;
    res.batch_size = batch_size;
    res.max_tokens = max_tokens;

    /* Generate synthetic documents */
    std::vector<std::string> docs;
    for (int i = 0; i < n_docs; i++) {
        docs.push_back(generate_synthetic_doc(max_tokens, i + max_tokens));
    }
    const char* query = "What is deep learning?";

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
    if (model_idx < 0) model_idx = 0;
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
        lembed_reranker_rerank(ctx, query, c_docs.data(), n_docs, batch_size, &result);
        lembed_rerank_results_free(&result);
    }

    /* Benchmark */
    std::vector<double> times;
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        lembed_rerank_results_t result = {0};
        lembed_reranker_rerank(ctx, query, c_docs.data(), n_docs, batch_size, &result);
        lembed_rerank_results_free(&result);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    lembed_reranker_free(ctx);

    /* Compute stats */
    std::sort(times.begin(), times.end());
    double p50 = times[times.size() / 2];
    double p95 = times[(size_t)(0.95 * times.size())];

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

/* Main reranker auto-tune implementation */
static lembed_status_t lembed_reranker_autotune_impl(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result)
{
    /* Check cache first */
    lembed_reranker_tuning_result_t cached;
    if (read_reranker_cache(model_name, &cached)) {
        fprintf(stderr, "reranker_autotune: using cached result for %s\n", model_name);
        *result = cached;
        return LEMBED_OK;
    }

    int cores = cpu_logical_cores();
    int n_docs = 20;
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

    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total_configs = valid_threads.size() * batch_vec.size() * tokens_vec.size();
    int current = 0;

    fprintf(stderr, "reranker_autotune: testing %d configurations (mode=%s, objective=%d)...\n",
            total_configs, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL", objective);

    for (int t : valid_threads) {
        for (int b : batch_vec) {
            for (int k : tokens_vec) {
                current++;

                auto r = bench_reranker_config(model_name, t, b, k, n_docs, warmup, bench_iters);

                double score = score_reranker_config(r, objective);
                double best_score = score_reranker_config(best, objective);

                if (score < best_score) {
                    best = r;
                }
            }
        }
    }

    fprintf(stderr, "reranker_autotune: best config: threads=%d batch=%d tokens=%d (P50=%.1fms, P95=%.1fms)\n",
            best.threads, best.batch_size, best.max_tokens, best.latency_ms, best.p95_latency_ms);

    /* Write to cache */
    write_reranker_cache(model_name, best);

    *result = best;
    return LEMBED_OK;
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
        return lembed_reranker_autotune_impl(model_name, mode, objective, result);
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
    int n_docs = 20;
    int warmup = 1;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

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
    std::vector<int> valid_tokens;
    for (int k : tokens_vec) {
        if (k >= min_tokens) valid_tokens.push_back(k);
    }
    if (valid_tokens.empty()) {
        valid_tokens = {min_tokens};
    }

    lembed_reranker_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total_configs = valid_threads.size() * batch_vec.size() * valid_tokens.size();
    int current = 0;

    fprintf(stderr, "reranker_autotune: testing %d configurations (mode=%s, objective=%d, min_tokens=%d, max_latency=%.0fms)...\n",
            total_configs, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL", objective, min_tokens, max_latency_ms);

    for (int t : valid_threads) {
        for (int b : batch_vec) {
            for (int k : valid_tokens) {
                current++;

                auto r = bench_reranker_config(model_name, t, b, k, n_docs, warmup, bench_iters);

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
        return lembed_reranker_autotune_impl(model_name, mode, objective, result);
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
    int n_docs = 20;
    int warmup = 2;
    int bench_iters = 10;

    std::vector<int> threads_vec = {1, 4, 8};
    std::vector<int> batch_vec = {4, 16};
    std::vector<int> tokens_vec = {64, 256};

    for (int t : threads_vec) {
        if (t > cores) continue;
        for (int b : batch_vec) {
            for (int k : tokens_vec) {
                auto r = bench_reranker_config(model_name, t, b, k, n_docs, warmup, bench_iters);
                if (r.p95_latency_ms > target_latency_ms) continue;
                if (r.throughput_docs_sec > best.throughput_docs_sec) {
                    best = r;
                }
            }
        }
    }

    if (best.latency_ms >= 999999) {
        return lembed_reranker_autotune_impl(model_name, LEMBED_AUTOTUNE_QUICK, objective, result);
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

/* Clear reranker autotune cache */
inline void lembed_reranker_autotune_clear_cache(const char* model_name) {
    clear_autotune_cache(model_name, "reranker");
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_HPP */




