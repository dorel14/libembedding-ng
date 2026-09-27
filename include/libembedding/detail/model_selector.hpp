/*
 * libembedding - model_selector.hpp
 * Auto model selection based on hardware and use case
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_MODEL_SELECTOR_HPP
#define LIBEMBEDDING_MODEL_SELECTOR_HPP

#include "libembedding/autotuner.h"
#include "libembedding/model_registry.h"
#include "libembedding/text_embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Model categories for different use cases */
struct ModelCategory {
    const char* model_code;
    const char* name;
    int dim;
    int max_tokens;
    int quantization;  /* 0=none, 1=static, 2=dynamic */
    double quality_score;  /* 1-10, higher is better quality */
};

/* Pre-categorized models for auto-selection */
static const ModelCategory g_models[] = {
    /* Fast models (speed priority) */
    {"Qdrant/all-MiniLM-L6-v2-onnx", "all-MiniLM-L6-v2", 384, 512, 0, 7.0},
    {"Xenova/all-MiniLM-L6-v2", "all-MiniLM-L6-v2-Q", 384, 512, 2, 6.5},
    {"Qdrant/all-MiniLM-L12-v2-onnx", "all-MiniLM-L12-v2", 384, 512, 0, 7.5},

    /* Quality models */
    {"Xenova/bge-small-en-v1.5", "bge-small-en-v1.5", 384, 512, 0, 8.5},
    {"Xenova/bge-base-en-v1.5", "bge-base-en-v1.5", 768, 512, 0, 9.0},

    /* Multilingual models */
    {"Xenova/paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-multilingual-MiniLM-L12-v2", 384, 512, 0, 7.0},
    {"intfloat/multilingual-e5-small", "multilingual-e5-small", 384, 512, 0, 8.0},
};

static const int g_n_models = sizeof(g_models) / sizeof(g_models[0]);

/* Result of model selection */
struct ModelSelection {
    const char* model_code;
    const char* model_name;
    int dim;
    int workers;
    int threads;
    int batch_size;
    double throughput_docs_sec;
    double latency_ms;
    double memory_mb;
    double score;
};

/* Score a model configuration based on use case */
static double score_model(double throughput, double latency_ms, double memory_mb,
                          double quality, const char* use_case) {
    double score = 0.0;

    if (strcmp(use_case, "speed") == 0) {
        /* Prioritize throughput */
        score = throughput * 10.0;
        score -= latency_ms * 0.5;
        score -= memory_mb * 0.01;
        score += quality * 2.0;  /* small quality bonus */
    } else if (strcmp(use_case, "quality") == 0) {
        /* Prioritize quality */
        score = quality * 100.0;
        score += throughput * 0.1;
        score -= latency_ms * 0.1;
        score -= memory_mb * 0.005;
    } else {
        /* Balanced (default) */
        score = throughput * 5.0;
        score += quality * 10.0;
        score -= latency_ms * 0.2;
        score -= memory_mb * 0.01;
    }

    return score;
}

/* Run auto model selection */
static lembed_status_t auto_select_impl(
        const char* use_case,
        ModelSelection& result) {
    if (!use_case) use_case = "balanced";

    /* Validate use_case */
    if (strcmp(use_case, "speed") != 0 &&
        strcmp(use_case, "quality") != 0 &&
        strcmp(use_case, "balanced") != 0) {
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }

    fprintf(stderr, "Auto model selection (use_case=%s)...\n", use_case);

    double best_score = -1e18;
    bool found = false;

    for (int i = 0; i < g_n_models; i++) {
        const auto& cat = g_models[i];

        fprintf(stderr, "  Testing %s...\n", cat.name);

        /* Quick autotune for this model */
        lembed_tuning_result_t tune = {0};
        lembed_status_t s = lembed_autotune(cat.model_code, LEMBED_AUTOTUNE_QUICK, &tune);
        if (s != LEMBED_OK) {
            fprintf(stderr, "    SKIP (autotune failed)\n");
            continue;
        }

        double memory_mb = tune.workers * 100.0;  /* estimate */
        double score = score_model(tune.throughput_docs_sec, tune.latency_ms,
                                   memory_mb, cat.quality_score, use_case);

        fprintf(stderr, "    -> %.0f docs/s, score=%.1f\n", tune.throughput_docs_sec, score);

        if (score > best_score) {
            best_score = score;
            result.model_code = cat.model_code;
            result.model_name = cat.name;
            result.dim = cat.dim;
            result.workers = tune.workers;
            result.threads = tune.threads;
            result.batch_size = tune.batch_size;
            result.throughput_docs_sec = tune.throughput_docs_sec;
            result.latency_ms = tune.latency_ms;
            result.memory_mb = memory_mb;
            result.score = score;
            found = true;
        }
    }

    if (!found) {
        return LEMBED_ERROR_MODEL_NOT_FOUND;
    }

    fprintf(stderr, "  Winner: %s (score=%.1f)\n", result.model_name, result.score);
    return LEMBED_OK;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_MODEL_SELECTOR_HPP */




