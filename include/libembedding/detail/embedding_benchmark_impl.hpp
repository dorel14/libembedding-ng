/*
 * libembedding - detail/embedding_benchmark_impl.hpp
 * Model selection: constraints -> Pareto -> scoring -> selection
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_EMBEDDING_BENCHMARK_IMPL_HPP
#define LIBEMBEDDING_EMBEDDING_BENCHMARK_IMPL_HPP

#include "../embedding_benchmark.h"
#include "llama_session_impl.hpp"
#include "../llama_session_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace lembed {
namespace detail {

static double bench_now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double bench_median_d(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* =========================================================================
 * Quality test set
 * ========================================================================= */
static const char* q_corpus[] = {
    "Machine learning algorithms learn patterns from data without explicit programming",
    "Deep neural networks use backpropagation for training multi-layer architectures",
    "The stock market experienced significant volatility due to economic uncertainty",
    "Investment portfolios should be diversified across asset classes",
    "The football match ended in a dramatic penalty shootout championship",
    "Soccer players train extensively for endurance and tactical awareness",
    "Climate change is causing rising sea levels and extreme weather events",
    "Renewable energy sources like solar and wind reduce carbon emissions",
    "The Renaissance period saw remarkable achievements in art and science",
    "Ancient Roman architecture influenced building design for centuries",
    "Quantum computing uses qubits to perform calculations faster than classical computers",
    "Cryptographic protocols secure communications using mathematical primitives",
};
static const int q_corpus_size = 12;

struct QTest { const char* query; std::vector<int> relevant; };
static QTest q_tests[] = {
    {"AI and machine learning",        {0, 1}},
    {"neural network training",         {1, 0}},
    {"financial markets and investing", {2, 3}},
    {"sports competition",              {4, 5}},
    {"environmental science",           {6, 7}},
    {"history and culture",             {8, 9}},
    {"computer science advances",       {10, 11}},
};
static const int q_test_count = 7;

static float cosine_sim(const float* a, const float* b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-8f) ? dot / denom : 0.0f;
}

static float measure_quality(lembed_text_embedding_t* emb, int dim) {
    std::vector<const char*> cptrs;
    for (int i = 0; i < q_corpus_size; i++) cptrs.push_back(q_corpus[i]);
    std::vector<std::vector<float>> cembs(q_corpus_size);
    for (int i = 0; i < q_corpus_size; i++) {
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(emb, &cptrs[i], 1, 1, &r);
        cembs[i].resize(dim);
        memcpy(cembs[i].data(), r.data, dim * sizeof(float));
        lembed_embeddings_free(&r);
    }
    int hits = 0;
    for (int t = 0; t < q_test_count; t++) {
        lembed_embeddings_t qr = {0};
        lembed_text_embedding_embed(emb, &q_tests[t].query, 1, 1, &qr);
        std::vector<std::pair<float, int>> scores;
        for (int d = 0; d < q_corpus_size; d++)
            scores.push_back({cosine_sim(qr.data, cembs[d].data(), dim), d});
        std::sort(scores.begin(), scores.end(), std::greater<>());
        bool hit = false;
        for (int k = 0; k < 3 && k < (int)scores.size(); k++)
            for (int rel : q_tests[t].relevant)
                if (scores[k].second == rel) { hit = true; break; }
        if (hit) hits++;
        lembed_embeddings_free(&qr);
    }
    return (float)hits / q_test_count;
}

/* =========================================================================
 * Throughput measurement (real, N sessions)
 * Uses a shared pool cache to avoid reloading model for each session count
 * ========================================================================= */

static float measure_pool_throughput(const char* path, int n_sess, int n_threads,
                                     const std::vector<std::string>& texts) {
    /* Use cached pool or create new one */
    static std::mutex cache_mtx;
    static std::unordered_map<std::string, std::unique_ptr<lembed::LlamaSessionPool>> cache;
    std::string cache_key = std::string(path) + "_" + std::to_string(n_sess) + "_" + std::to_string(n_threads);

    lembed::LlamaSessionPool* pool_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(cache_mtx);
        auto it = cache.find(cache_key);
        if (it != cache.end()) {
            pool_ptr = it->second.get();
        } else {
            auto pool = std::make_unique<lembed::LlamaSessionPool>();
            try { pool->load_from_file(path, n_sess, n_threads, 0, 0, 0, false); }
            catch (...) { return 0; }
            pool_ptr = pool.get();
            cache[cache_key] = std::move(pool);
        }
    }

    /* Warmup */
    for (int i = 0; i < n_sess && i < (int)texts.size(); i++)
        pool_ptr->embed(texts[i].c_str());

    /* Benchmark with round-robin */
    int total = (int)texts.size();
    std::atomic<int> idx{0};
    double t0 = bench_now_ms();
    std::vector<std::thread> workers;
    for (int w = 0; w < n_sess; w++)
        workers.emplace_back([&, w]() {
            while (true) {
                int i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= total) break;
                pool_ptr->embed(texts[i].c_str());
            }
        });
    for (auto& t : workers) t.join();
    double t1 = bench_now_ms();
    return (float)total / ((float)(t1 - t0) / 1000.0f);
}

/* Detect optimal sessions: measure 1..max, stop if <5% gain relative to best */
static int detect_optimal_sessions(const char* path, int max_sessions, float* best_throughput) {
    std::vector<std::string> texts;
    for (int i = 0; i < 64; i++)
        texts.push_back("Document " + std::to_string(i) + " about topic " + std::to_string(i % 8));
    float best_tps = 0;
    int best_n = 1;
    int plateau_count = 0;
    for (int n = 1; n <= max_sessions; n++) {
        float tps = measure_pool_throughput(path, n, 1, texts);
        if (tps > best_tps) { best_tps = tps; best_n = n; plateau_count = 0; }
        else if (tps < best_tps * 0.95f) {  /* More than 5% below best */
            plateau_count++;
            if (plateau_count >= 2) break;  /* Stop after 2 consecutive declines */
        }
    }
    if (best_throughput) *best_throughput = best_tps;
    return best_n;
}

/* =========================================================================
 * Candidate: benchmark result + Pareto rank
 * ========================================================================= */
struct Candidate {
    std::string path;
    std::string name;
    float quality;
    float throughput;
    float file_mb;
    float score;
    int sessions;
    int dim;
    bool dominated;
};

/* Pareto frontier: mark dominated candidates.
 * O(n log n) algorithm:
 * 1. Sort by quality descending (primary objective)
 * 2. Maintain a 2D Pareto frontier on (throughput, file_mb)
 * 3. For each candidate, binary search in frontier to check dominance
 */
static void compute_pareto(std::vector<Candidate>& cands) {
    for (auto& c : cands) c.dominated = false;
    if (cands.size() <= 1) return;

    // Sort by quality descending, then throughput descending, then file_mb ascending
    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        if (std::abs(a.quality - b.quality) > 0.01f) return a.quality > b.quality;
        if (std::abs(a.throughput - b.throughput) > 1.0f) return a.throughput > b.throughput;
        return a.file_mb < b.file_mb;
    });

    // Frontier of non-dominated candidates (indices into cands)
    // Maintained sorted by throughput descending for binary search
    std::vector<size_t> frontier;

    for (size_t i = 0; i < cands.size(); i++) {
        // Binary search in frontier for insertion point by throughput
        size_t lo = 0, hi = frontier.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (cands[frontier[mid]].throughput >= cands[i].throughput - 1.0f)
                lo = mid + 1;
            else
                hi = mid;
        }

        // Check if any frontier element dominates cands[i]
        // Elements before lo have throughput >= cands[i].throughput
        bool dominated = false;
        for (size_t k = 0; k < lo; k++) {
            size_t idx = frontier[k];
            bool better_or_eq = (cands[idx].throughput >= cands[i].throughput - 1.0f) &&
                                (cands[idx].file_mb <= cands[i].file_mb + 0.5f);
            bool strictly_better = (cands[idx].quality > cands[i].quality + 0.01f) ||
                                   (cands[idx].throughput > cands[i].throughput + 1.0f) ||
                                   (cands[idx].file_mb < cands[i].file_mb - 0.5f);
            if (better_or_eq && strictly_better) {
                dominated = true;
                break;
            }
        }

        if (dominated) {
            cands[i].dominated = true;
        } else {
            // Remove frontier elements dominated by this candidate
            auto it = std::remove_if(frontier.begin(), frontier.end(),
                [&](size_t idx) {
                    bool better_or_eq = (cands[i].throughput >= cands[idx].throughput - 1.0f) &&
                                        (cands[i].file_mb <= cands[idx].file_mb + 0.5f);
                    bool strictly_better = (cands[i].quality > cands[idx].quality + 0.01f) ||
                                           (cands[i].throughput > cands[idx].throughput + 1.0f) ||
                                           (cands[i].file_mb < cands[idx].file_mb - 0.5f);
                    return better_or_eq && strictly_better;
                });
            frontier.erase(it, frontier.end());
            frontier.insert(frontier.begin() + lo, i);
        }
    }
}

/* =========================================================================
 * Benchmark single model
 * ========================================================================= */
static float benchmark_model(const char* path, const char* name,
                             const lembed_benchmark_constraints_t* constraints,
                             const lembed_benchmark_weights_t& weights,
                             Candidate* out) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.show_download_progress = 0;
    lembed_text_embedding_t* emb = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(path, &opts, &emb);
    if (s != LEMBED_OK) return -1.0f;
    int dim = lembed_text_embedding_dim(emb);
    std::vector<std::string> texts;
    for (int i = 0; i < 64; i++)
        texts.push_back("Document " + std::to_string(i) + " about topic " + std::to_string(i % 8));
    std::vector<const char*> tptrs;
    for (auto& t : texts) tptrs.push_back(t.c_str());
    lembed_embeddings_t w = {0};
    lembed_text_embedding_embed(emb, tptrs.data(), 4, 4, &w);
    lembed_embeddings_free(&w);
    float quality = measure_quality(emb, dim);
    lembed_text_embedding_free(emb);
    float file_mb = 0;
    try { file_mb = (float)std::filesystem::file_size(path) / (1024.0f * 1024.0f); } catch (...) {}
    /* Hard constraints */
    if (constraints) {
        if (constraints->quality_min > 0.0f && quality < constraints->quality_min) return -1.0f;
        if (constraints->memory_max_mb > 0.0f && file_mb > constraints->memory_max_mb) return -1.0f;
    }
    float best_tps = 0;
    int optimal_sessions = detect_optimal_sessions(path, 8, &best_tps);
    if (constraints && constraints->throughput_min > 0.0f && best_tps < constraints->throughput_min)
        return -1.0f;
    /* Score */
    float nq = quality;
    float nt = std::min(best_tps / 300.0f, 1.0f);
    float nc = std::min(20.0f / std::max(file_mb, 1.0f), 1.0f);
    float score = weights.quality_weight * nq + weights.throughput_weight * nt + weights.cost_weight * nc;
    if (out) {
        out->path = path; out->name = name; out->quality = quality;
        out->throughput = best_tps; out->file_mb = file_mb; out->score = score;
        out->sessions = optimal_sessions; out->dim = dim; out->dominated = false;
    }
    return score;
}

} /* namespace detail */
} /* namespace lembed */

/* =========================================================================
 * C API
 * ========================================================================= */
#ifdef LIBEMBEDDING_IMPLEMENTATION

const char* lembed_benchmark_default_cache_dir(void) {
    static std::string dir;
    static std::mutex dir_mutex;
    std::lock_guard<std::mutex> lock(dir_mutex);
    if (dir.empty()) {
        const char* home = getenv("USERPROFILE");
        if (!home) home = getenv("HOME");
        if (!home) home = ".";
        dir = std::string(home) + "/.cache/libembedding";
    }
    return dir.c_str();
}

lembed_status_t lembed_benchmark_select_model(
    const char* model_dir,
    lembed_objective_t objective,
    const lembed_benchmark_constraints_t* constraints,
    const lembed_benchmark_weights_t* custom_weights,
    lembed_benchmark_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    if (!model_dir) model_dir = lembed_benchmark_default_cache_dir();
    lembed_benchmark_weights_t weights = custom_weights ? *custom_weights
                                                        : lembed_benchmark_profile_weights(objective);
    namespace fs = std::filesystem;
    std::vector<fs::path> models;
    try {
        // Recursively scan for .gguf files (models may be in subdirectories like models--{repo}/)
        for (auto& entry : fs::recursive_directory_iterator(model_dir))
            if (entry.is_regular_file() && entry.path().extension() == ".gguf")
                models.push_back(entry.path());
    } catch (...) { return LEMBED_ERROR_IO; }
    if (models.empty()) return LEMBED_ERROR_MODEL_NOT_FOUND;
    /* Benchmark all */
    std::vector<lembed::detail::Candidate> cands;
    for (auto& m : models) {
        lembed::detail::Candidate c;
        float score = lembed::detail::benchmark_model(m.string().c_str(), m.stem().string().c_str(),
                                                       constraints, weights, &c);
        if (score >= 0) cands.push_back(c);
    }
    if (cands.empty()) return LEMBED_ERROR_MODEL_NOT_FOUND;
    /* Pareto frontier */
    lembed::detail::compute_pareto(cands);
    /* Select best among non-dominated */
    float best_score = -1.0f;
    lembed::detail::Candidate best;
    bool found = false;
    for (auto& c : cands) {
        if (c.dominated) continue;
        if (c.score > best_score) { best_score = c.score; best = c; found = true; }
    }
    if (!found) return LEMBED_ERROR_MODEL_NOT_FOUND;
    /* Fill result */
    snprintf(result->model_path, sizeof(result->model_path), "%s", best.path.c_str());
    snprintf(result->model_name, sizeof(result->model_name), "%s", best.name.c_str());
    snprintf(result->backend, sizeof(result->backend), "%s", "llama.cpp");
    result->config.batch_size = best.sessions;
    result->config.num_threads = (best.sessions > 1) ? 1 : 4;
    result->metrics.dim = best.dim;
    result->metrics.throughput_docs_sec = best.throughput;
    result->quality_score = best.quality;
    result->score = best.score;
    result->file_size_mb = best.file_mb;
    result->pareto_rank = 1;
    return LEMBED_OK;
}

lembed_status_t lembed_benchmark_detect_sessions(
    const char* model_path, int max_sessions, int* optimal_sessions, float* best_throughput) {
    if (!model_path || !optimal_sessions || max_sessions <= 0)
        return LEMBED_ERROR_INVALID_ARGUMENT;
    float tps = 0;
    int n = lembed::detail::detect_optimal_sessions(model_path, max_sessions, &tps);
    *optimal_sessions = n;
    if (best_throughput) *best_throughput = tps;
    return LEMBED_OK;
}

int lembed_benchmark_list_models(const char* model_dir, char* model_paths, int max_models) {
    if (!model_dir || !model_paths || max_models <= 0) return 0;
    namespace fs = std::filesystem;
    int count = 0;
    try {
        for (auto& entry : fs::directory_iterator(model_dir)) {
            if (count >= max_models) break;
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                snprintf(model_paths + count * 512, 512, "%s", entry.path().string().c_str());
                count++;
            }
        }
    } catch (...) {}
    return count;
}

#endif /* LIBEMBEDDING_IMPLEMENTATION */
#endif /* LIBEMBEDDING_EMBEDDING_BENCHMARK_IMPL_HPP */







