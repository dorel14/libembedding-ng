/*
 * libembedding - detail/unified_benchmark_impl.hpp
 * Unified Backend Benchmark implementation
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_UNIFIED_BENCHMARK_IMPL_HPP
#define LIBEMBEDDING_UNIFIED_BENCHMARK_IMPL_HPP

#include "../unified_benchmark.h"
#include "llama_session_impl.hpp"
#include "../llama_session_pool.hpp"
#include "win_psapi_init.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace lembed {
namespace detail {

static double ubench_now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median_d(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double percentile95(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)std::ceil(0.95 * v.size()) - 1;
    return v[std::min(idx, v.size() - 1)];
}

#ifdef _WIN32
static double peak_rss_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
}
#else
static double peak_rss_mb() { return 0.0; }
#endif

/* =========================================================================
 * Deduplicated test corpus from plan (lines 974+)
 * ========================================================================= */

/* Short texts (< 20 tokens) -- 12 unique */
static const char* corpus_short[] = {
    "Hello world.",
    "Bonjour le monde.",
    "Hallo Welt.",
    "Hola mundo.",
    "Ciao mondo.",
    "Ola mundo.",
    "Konnichiwa sekai ni yoroshiku.",
    "Xin chao the gioi.",
    "Zdravstvuyte mir.",
    "Assalamu alaikum.",
    "Hello world.",
    "Machine learning transforms data into insights.",
    "L'intelligence artificielle transforme les donnees.",
};

/* Medium texts (20-80 tokens) -- 10 unique */
static const char* corpus_medium[] = {
    "The quick brown fox jumps over the lazy dog near the riverbank while the sun sets behind the mountains.",
    "Le renard brun rapide saute par-dessus le chien paresseux pres de la riviere pendant que le soleil se couche.",
    "Der schnelle braune Fuchs springt ueber den faulen Hund in der Naehe des Flusses, waehrend die Sonne hinter den Bergen untergeht.",
    "Machine learning algorithms can identify patterns in large datasets automatically without explicit programming instructions.",
    "Climate change affects global weather patterns and sea levels significantly across all continents and ocean regions worldwide.",
    "The history of ancient Rome spans over a thousand years of civilization from its founding to the fall of the western empire.",
    "Quantum computing promises to revolutionize cryptography drug discovery and materials science through parallel processing capabilities.",
    "Les algorithmes d'apprentissage automatique peuvent identifier des motifs dans de grands ensembles de donnees.",
    "Die kuenstliche Intelligenz veraendert die Art und Weise wie wir arbeiten kommunizieren und Probleme loesen.",
    "El aprendizaje automatico permite a las computadoras aprender de los datos y mejorar con la experiencia.",
};

/* Long texts (80-200 tokens) -- 6 unique */
static const char* corpus_long[] = {
    "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with the interactions between computers and human language in particular how to program computers to process and analyze large amounts of natural language data.",
    "The transformer architecture introduced in the attention is all you need paper has become the foundation for most modern natural language processing systems including BERT GPT and their variants which have revolutionized the field.",
    "Deep learning is part of a broader family of machine learning methods based on artificial networks with representation learning and has been applied to fields including computer vision speech recognition natural language processing and bioinformatics.",
    "Le traitement automatique du langage naturel est un domaine de l'informatique et de l'intelligence artificielle qui s'interesse aux interactions entre les ordinateurs et le langage humain.",
    "Die kuenstliche Intelligenz ist ein Gebiet der Informatik das sich mit der Automatisierung intelligentem Verhalten und dem maschinellen Lernen befasst.",
    "El procesamiento del lenguaje natural es un campo de la informatica la inteligence artificial y la lingustica interesado en las interacciones entre las computadoras y el lenguaje humano.",
};

/* Very long texts (200+ tokens) -- 3 unique */
static const char* corpus_very_long[] = {
    "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy. Large language models trained on vast amounts of text data have demonstrated capabilities that were previously thought to be decades away including reasoning summarization and creative writing.",
    "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think. Since then the field has gone through periods of optimism and disappointment known as AI winters. Today we are in a period of rapid advancement driven by increases in computational power the availability of large datasets and improvements in algorithms particularly deep learning.",
    "L'intelligence artificielle a fait des progrès significatifs ces dernières années en particulier dans les domaines de l'apprentissage automatique de l'apprentissage profond et du traitement du langage naturel. Ces avancées ont permis le développement de systèmes capables de comprendre de générer et de traduire le langage humain avec une précision remarquable.",
};

/* Edge cases -- 20 unique */
static const char* corpus_edge[] = {
    "",                           /* Empty */
    "a",                          /* Single char */
    "   ",                        /* Whitespace */
    "12345 67890",                /* Numbers */
    "!@#$%^&*()",                 /* Special chars */
    "Hello world world world",       /* Emoji */
    "Hello world CJK mixed",            /* CJK mixed */
    "Hello RTL world",          /* RTL */
    "test test test test test test test test test test test test test test test test",  /* Repeated */
    "def foo(): return 42",       /* Code */
    "{\"key\": \"value\"}",       /* JSON */
    "user@example.com",           /* Email */
    "<div>Hello</div>",           /* HTML */
    "word word word word word word word word word word word word word word word word",  /* 16 tokens */
    "word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word ",  /* 64 tokens */
    "word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word ",  /* 256 tokens */
    "cafe resume naive",          /* Accents */
    "+++++",                     /* Math */
    "$$$",                       /* Currency */
};

/* Multilingual -- 10 unique */
static const char* corpus_multilingual[] = {
    "Hello world.",
    "Bonjour le monde.",
    "Hallo Welt.",
    "Hola mundo.",
    "Ola mundo.",
    "Konnichiwa sekai ni yoroshiku.",
    "Xin chao the gioi.",
    "Zdravstvuyte mir.",
    "Assalamu alaikum.",
    "Le traitement automatique du langage naturel est un domaine de l'IA.",
};

/* All texts combined (deduplicated) */
static std::vector<const char*> get_all_texts() {
    std::vector<const char*> all;
    for (auto t : corpus_short) all.push_back(t);
    for (auto t : corpus_medium) all.push_back(t);
    for (auto t : corpus_long) all.push_back(t);
    for (auto t : corpus_very_long) all.push_back(t);
    return all;
}

static const std::vector<const char*>& corpus_by_type(lembed_corpus_type_t type) {
    static std::vector<const char*> empty;
    static std::vector<const char*> all = get_all_texts();
    static std::vector<const char*> edge_vec(std::begin(corpus_edge), std::end(corpus_edge));
    static std::vector<const char*> multi_vec(std::begin(corpus_multilingual), std::end(corpus_multilingual));
    static std::vector<const char*> short_vec(std::begin(corpus_short), std::end(corpus_short));
    static std::vector<const char*> medium_vec(std::begin(corpus_medium), std::end(corpus_medium));
    static std::vector<const char*> long_vec(std::begin(corpus_long), std::end(corpus_long));
    static std::vector<const char*> vlong_vec(std::begin(corpus_very_long), std::end(corpus_very_long));
    switch (type) {
        case LEMBED_CORPUS_SHORT: return short_vec;
        case LEMBED_CORPUS_MEDIUM: return medium_vec;
        case LEMBED_CORPUS_LONG: return long_vec;
        case LEMBED_CORPUS_VERY_LONG: return vlong_vec;
        case LEMBED_CORPUS_MIXED: return all;
        case LEMBED_CORPUS_MULTILINGUAL: return multi_vec;
        case LEMBED_CORPUS_EDGE_CASES: return edge_vec;
        default: return empty;
    }
}

/* =========================================================================
 * llama.cpp backend benchmark
 * ========================================================================= */
static lembed_status_t bench_llama(
    const char* model_path,
    const lembed_backend_config_t* config,
    const std::vector<const char*>& texts,
    lembed_benchmark_metrics_t* metrics) {
    int n_sessions = config->workers > 0 ? std::max(1, config->workers) : std::max(1, config->batch_size);
    int n_threads = std::max(1, config->num_threads);
    lembed::LlamaSessionPool pool;
    try {
        pool.load_from_file(model_path, n_sessions, n_threads, 0, 0, 0, false);
    } catch (...) { return LEMBED_ERROR_LLAMA; }
    metrics->dim = pool.dimension();
    /* Warmup */
    for (int i = 0; i < n_sessions && i < (int)texts.size(); i++)
        pool.embed(texts[i]);
    /* Benchmark */
    int total = (int)texts.size();
    std::atomic<int> idx{0};
    double t0 = ubench_now_ms();
    std::vector<std::thread> workers;
    for (int w = 0; w < n_sessions; w++)
        workers.emplace_back([&]() {
            while (true) {
                int i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= total) break;
                pool.embed(texts[i]);
            }
        });
    for (auto& t : workers) t.join();
    double t1 = ubench_now_ms();
    metrics->throughput_docs_sec = (float)total / ((float)(t1 - t0) / 1000.0f);
    metrics->num_texts = total;
    metrics->num_errors = 0;
    metrics->peak_memory_mb = (float)peak_rss_mb();
    /* Latency: measure single-text */
    std::vector<double> latencies;
    for (int i = 0; i < std::min(20, total); i++) {
        double s0 = ubench_now_ms();
        pool.embed(texts[i]);
        double s1 = ubench_now_ms();
        latencies.push_back(s1 - s0);
    }
    metrics->latency_p50_ms = (float)median_d(latencies);
    metrics->latency_p95_ms = (float)percentile95(latencies);
    return LEMBED_OK;
}


/* =========================================================================
 * ONNX backend benchmark
 * ========================================================================= */
static lembed_status_t bench_onnx(
    const char* model_path,
    const lembed_backend_config_t* config,
    const std::vector<const char*>& texts,
    lembed_benchmark_metrics_t* metrics) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = config->num_threads;
    opts.show_download_progress = 0;
    double t0 = ubench_now_ms();
    lembed_text_embedding_t* emb = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_path(model_path, &opts, &emb);
    double t1 = ubench_now_ms();
    if (s != LEMBED_OK) return s;
    metrics->load_time_ms = (float)(t1 - t0);
    metrics->dim = lembed_text_embedding_dim(emb);
    /* Warmup */
    lembed_embeddings_t w = {0};
    lembed_text_embedding_embed(emb, texts.data(), std::min(4, (int)texts.size()), config->batch_size, &w);
    lembed_embeddings_free(&w);
    /* Benchmark */
    int total = (int)texts.size();
    int bsz = config->batch_size;
    double bt0 = ubench_now_ms();
    for (int off = 0; off < total; off += bsz) {
        int n = std::min(bsz, total - off);
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(emb, texts.data() + off, n, bsz, &r);
        lembed_embeddings_free(&r);
    }
    double bt1 = ubench_now_ms();
    metrics->throughput_docs_sec = (float)total / ((float)(bt1 - bt0) / 1000.0f);
    metrics->num_texts = total;
    metrics->num_errors = 0;
    metrics->peak_memory_mb = (float)peak_rss_mb();
    /* Latency */
    std::vector<double> latencies;
    for (int i = 0; i < std::min(20, total); i++) {
        double s0 = ubench_now_ms();
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(emb, &texts[i], 1, 1, &r);
        lembed_embeddings_free(&r);
        double s1 = ubench_now_ms();
        latencies.push_back(s1 - s0);
    }
    metrics->latency_p50_ms = (float)median_d(latencies);
    metrics->latency_p95_ms = (float)percentile95(latencies);
    lembed_text_embedding_free(emb);
    return LEMBED_OK;
}

} /* namespace detail */
} /* namespace lembed */

/* =========================================================================
 * C API
 * ========================================================================= */
#ifdef LIBEMBEDDING_IMPLEMENTATION

lembed_status_t lembed_benchmark_get_corpus(
    lembed_corpus_type_t type,
    const char* const** out_texts,
    int* out_count) {
    if (!out_texts || !out_count) return LEMBED_ERROR_INVALID_ARGUMENT;
    static std::vector<const char*> cache;
    const std::vector<const char*>& vec = lembed::detail::corpus_by_type(type);
    cache.assign(vec.begin(), vec.end());
    *out_texts = cache.data();
    *out_count = (int)vec.size();
    return LEMBED_OK;
}

lembed_status_t lembed_benchmark_run(
    const char* model_path,
    const char* backend,
    lembed_corpus_type_t corpus_type,
    const lembed_backend_config_t* config,
    lembed_benchmark_result_t* result) {
    if (!model_path || !backend || !config || !result) return LEMBED_ERROR_INVALID_ARGUMENT;
    const auto& texts_vec = lembed::detail::corpus_by_type(corpus_type);
    std::vector<const char*> texts(texts_vec.begin(), texts_vec.end());
    memset(result, 0, sizeof(*result));
    // Store only filename (not full path) for privacy
    std::string fname = std::filesystem::path(model_path).filename().string();
    snprintf(result->model_name, sizeof(result->model_name), "%s", fname.c_str());
    snprintf(result->backend, sizeof(result->backend), "%s", backend);
    result->config = *config;
    lembed_benchmark_metrics_t metrics = {0};
    lembed_status_t s;
    if (strcmp(backend, "onnx") == 0) {
        s = lembed::detail::bench_onnx(model_path, config, texts, &metrics);
    } else {
        s = lembed::detail::bench_llama(model_path, config, texts, &metrics);
    }
    if (s != LEMBED_OK) return s;
    result->metrics = metrics;
    return LEMBED_OK;
}

int lembed_benchmark_compare(
    const char* onnx_path,
    const char* gguf_path,
    lembed_corpus_type_t corpus_type,
    lembed_benchmark_result_t* results) {
    if (!results) return 0;
    int count = 0;
    if (onnx_path) {
        lembed_backend_config_t cfg = {"onnx", 4, 64, 1};
        if (lembed_benchmark_run(onnx_path, "onnx", corpus_type, &cfg, &results[count]) == LEMBED_OK)
            count++;
    }
    if (gguf_path) {
        lembed_backend_config_t cfg = {"llama.cpp", 1, 4, 0};
        if (lembed_benchmark_run(gguf_path, "llama.cpp", corpus_type, &cfg, &results[count]) == LEMBED_OK)
            count++;
    }
    return count;
}

lembed_status_t lembed_benchmark_autotune(
    const char* model_path,
    const char* backend,
    lembed_objective_t objective,
    lembed_benchmark_result_t* result) {
    if (!model_path || !backend || !result) return LEMBED_ERROR_INVALID_ARGUMENT;
    const auto& texts_vec = lembed::detail::corpus_by_type(LEMBED_CORPUS_MIXED);
    std::vector<const char*> texts(texts_vec.begin(), texts_vec.end());
    float best_score = -1;
    lembed_benchmark_result_t best = {0};
    if (strcmp(backend, "llama.cpp") == 0) {
        for (int n_sess = 1; n_sess <= 8; n_sess++) {
            lembed_backend_config_t cfg = {"llama.cpp", 1, n_sess, 0};
            lembed_benchmark_result_t r = {0};
            if (lembed_benchmark_run(model_path, backend, LEMBED_CORPUS_MIXED, &cfg, &r) != LEMBED_OK) continue;
            float score = r.metrics.throughput_docs_sec;
            if (score > best_score) { best_score = score; best = r; }
            if (n_sess > 1 && score < best_score * 1.05f) break;
        }
    } else {
        for (int thr = 1; thr <= 4; thr++) {
            for (int bsz : {32, 64, 128}) {
                lembed_backend_config_t cfg = {"onnx", thr, bsz, 1};
                lembed_benchmark_result_t r = {0};
                if (lembed_benchmark_run(model_path, backend, LEMBED_CORPUS_MIXED, &cfg, &r) != LEMBED_OK) continue;
                float score = r.metrics.throughput_docs_sec;
                if (score > best_score) { best_score = score; best = r; }
            }
        }
    }
    if (best_score < 0) return LEMBED_ERROR_LLAMA;
    *result = best;
    return LEMBED_OK;
}

#endif /* LIBEMBEDDING_IMPLEMENTATION */
#endif /* LIBEMBEDDING_UNIFIED_BENCHMARK_IMPL_HPP */
