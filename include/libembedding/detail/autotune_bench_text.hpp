/*
 * libembedding - detail/autotune_bench_text.hpp
 * Text embedding auto-tuner
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_HPP

#include "libembedding/autotuner.h"
#include "libembedding/text_embedding.h"
#include "autotune_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace lembed { namespace detail {

/* Generate synthetic benchmark corpus with varied lengths */
inline std::vector<std::string> generate_corpus(int n_samples) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "algorithms", "process", "data", "efficiently",
        "natural", "language", "processing", "enables", "understanding", "semantic",
        "embeddings", "represent", "meaningful", "vector", "representations",
        "transformer", "models", "utilize", "attention", "mechanisms",
        "deep", "neural", "networks", "learn", "patterns", "from", "training",
        "optimization", "techniques", "improve", "convergence", "accuracy",
        "inference", "latency", "throughput", "benchmark", "performance",
        "production", "deployment", "scalable", "reliable", "systems"
    };
    int nwords = sizeof(words) / sizeof(words[0]);

    /* Distribution: 40% short, 40% medium, 20% long */
    int lengths[] = {16, 16, 16, 16, 64, 64, 64, 64, 128, 128};
    int n_lengths = sizeof(lengths) / sizeof(lengths[0]);

    std::vector<std::string> corpus;
    corpus.reserve(n_samples);

    for (int i = 0; i < n_samples; i++) {
        int ntok = lengths[i % n_lengths];
        std::string text;
        for (int j = 0; j < ntok; j++) {
            if (j > 0) text += " ";
            text += words[(i * 7 + j) % nwords];
        }
        corpus.push_back(text);
    }

    return corpus;
}

/* Benchmark a single text embedding configuration */
inline double benchmark_text_config(
        const std::vector<std::string>& corpus,
        int workers, int threads, int batch_size,
        double& out_latency_ms) {

    if (corpus.empty()) return 0.0;

    /* Create workers */
    std::vector<lembed_text_embedding_t*> emb(workers);
    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.offline = 1;
    opts.show_download_progress = 0;

    for (int i = 0; i < workers; i++) {
        if (lembed_text_embedding_create(&opts, &emb[i]) != LEMBED_OK) {
            for (int j = 0; j < i; j++) lembed_text_embedding_free(emb[j]);
            return 0.0;
        }
    }

    int n = (int)corpus.size();
    int per = n / workers;

    /* Warmup */
    for (int w = 0; w < 3; w++) {
        for (int i = 0; i < workers; i++) {
            std::vector<const char*> ct;
            int start = i * per;
            int cnt = (i == workers - 1) ? n - start : per;
            for (int k = 0; k < cnt; k++)
                ct.push_back(corpus[start + k].c_str());
            lembed_embeddings_t r = {0};
            lembed_text_embedding_embed(emb[i], ct.data(), cnt, cnt, &r);
            lembed_embeddings_free(&r);
        }
    }

    /* Measure */
    auto t0 = std::chrono::high_resolution_clock::now();
    int n_batches = 3;

    for (int b = 0; b < n_batches; b++) {
        std::vector<std::thread> th;
        for (int i = 0; i < workers; i++) {
            th.emplace_back([&emb, i, &corpus, per, n, workers]() {
                std::vector<const char*> ct;
                int start = i * per;
                int cnt = (i == workers - 1) ? n - start : per;
                for (int k = 0; k < cnt; k++)
                    ct.push_back(corpus[start + k].c_str());
                lembed_embeddings_t r = {0};
                lembed_text_embedding_embed(emb[i], ct.data(), cnt, cnt, &r);
                lembed_embeddings_free(&r);
            });
        }
        for (auto& t : th) t.join();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double total_texts = n_batches * corpus.size();
    double docs_per_sec = total_texts / (total_ms / 1000.0);
    out_latency_ms = total_ms / total_texts;

    for (int i = 0; i < workers; i++)
        lembed_text_embedding_free(emb[i]);

    return docs_per_sec;
}

/* Scoring function: higher is better */
inline double score_text_config(double throughput, double latency_ms, double memory_mb) {
    double mem_penalty = 0.0;
    if (memory_mb > 500.0) {
        mem_penalty = (memory_mb - 500.0) / 100.0;
    }
    double lat_penalty = 0.0;
    if (latency_ms > 20.0) {
        lat_penalty = (latency_ms - 20.0) / 5.0;
    }
    return throughput - mem_penalty - lat_penalty;
}

/* Main text autotune implementation */
inline lembed_status_t autotune_text_impl(
        lembed_text_model_t model,
        const std::vector<std::string>& corpus,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {

    /* Get model code for cache key */
    lembed_model_info_t info;
    if (lembed_get_text_model_info(model, &info) != LEMBED_OK) {
        return LEMBED_ERROR_MODEL_NOT_FOUND;
    }
    const char* model_code = info.model_code;

    /* Check cache first */
    lembed_tuning_result_t cached;
    {
        std::ifstream f(get_cache_path(model_code));
        if (f.is_open()) {
            std::string line;
            while (std::getline(f, line)) {
                auto pos = line.find(':');
                if (pos == std::string::npos) continue;
                std::string key = line.substr(0, pos);
                std::string val = line.substr(pos + 1);
                trim_json_value(key);
                trim_json_value(val);
                if (key == "workers") cached.workers = std::stoi(val);
                else if (key == "threads") cached.threads = std::stoi(val);
                else if (key == "batch_size") cached.batch_size = std::stoi(val);
                else if (key == "throughput_docs_sec") cached.throughput_docs_sec = std::stod(val);
                else if (key == "latency_ms") cached.latency_ms = std::stod(val);
                else if (key == "memory_mb") cached.memory_mb = std::stod(val);
            }
            fprintf(stderr, "autotune: cache hit for %s\n", model_code);
            if (result) *result = cached;
            return LEMBED_OK;
        }
    }

    fprintf(stderr, "autotune: cache miss for %s, running benchmark...\n", model_code);

    int cores = cpu_logical_cores();
    int n_samples = (mode == LEMBED_AUTOTUNE_QUICK) ? 16 : 64;

    /* Generate corpus if not provided */
    std::vector<std::string> bench_corpus;
    if (corpus.empty()) {
        bench_corpus = generate_corpus(n_samples);
    } else {
        bench_corpus = corpus;
    }

    /* Configurations to test */
    struct Config { int workers; int threads; int batch; };
    std::vector<Config> configs;

    if (mode == LEMBED_AUTOTUNE_QUICK) {
        int worker_opts[] = {1, std::min(4, cores), std::min(8, cores)};
        for (int w : worker_opts) {
            if (w > cores) continue;
            configs.push_back({w, 1, 32});
        }
    } else {
        int worker_opts[] = {1, 2, 4, 8, 16};
        int thread_opts[] = {1, 2, 4};
        int batch_opts[] = {16, 32, 64, 128, 256};

        for (int w : worker_opts) {
            if (w > cores) continue;
            for (int t : thread_opts) {
                if (w * t > cores) continue;
                for (int b : batch_opts) {
                    configs.push_back({w, t, b});
                }
            }
        }
    }

    /* Benchmark each config */
    double best_score = -1e18;
    lembed_tuning_result_t best = {1, 1, 64, 0, 0, 0};

    for (const auto& cfg : configs) {
        double latency = 0.0;
        double throughput = benchmark_text_config(bench_corpus, cfg.workers, cfg.threads, cfg.batch, latency);
        double memory = cfg.workers * 100.0;
        double score = score_text_config(throughput, latency, memory);

        fprintf(stderr, "  autotune: workers=%d threads=%d batch=%d -> %.1f docs/s\n",
                cfg.workers, cfg.threads, cfg.batch, throughput);

        if (score > best_score) {
            best_score = score;
            best.workers = cfg.workers;
            best.threads = cfg.threads;
            best.batch_size = cfg.batch;
            best.throughput_docs_sec = throughput;
            best.latency_ms = latency;
            best.memory_mb = memory;
        }
    }

    if (result) *result = best;

    /* Write to cache */
    std::ofstream of(get_cache_path(model_code));
    if (of.is_open()) {
        of << "{\n";
        of << "  \"workers\": " << best.workers << ",\n";
        of << "  \"threads\": " << best.threads << ",\n";
        of << "  \"batch_size\": " << best.batch_size << ",\n";
        of << "  \"throughput_docs_sec\": " << best.throughput_docs_sec << ",\n";
        of << "  \"latency_ms\": " << best.latency_ms << ",\n";
        of << "  \"memory_mb\": " << best.memory_mb << "\n";
        of << "}\n";
    }

    return LEMBED_OK;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_HPP */




