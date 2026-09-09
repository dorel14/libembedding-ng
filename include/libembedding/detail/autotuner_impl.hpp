/*
 * libembedding - detail/autotuner_impl.hpp
 * Auto-tuning implementation (orchestrator)
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP

#include "libembedding/autotuner.h"
#include "autotune_cache.hpp"
#include "autotune_bench_text.hpp"
#include "autotune_bench_reranker.hpp"
#include "autotune_bench_sparse.hpp"
#ifndef LIBEMBEDDING_NO_IMAGE
#include "autotune_bench_image.hpp"
#endif

#include <cstring>

namespace lembed { namespace detail {

/* =========================================================================
 * Unified Auto-Tuner Implementation
 * ========================================================================= */

lembed_status_t lembed_autotune_unified(
    lembed_task_t task,
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_unified_tuning_result_t* result)
{
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(result, 0, sizeof(*result));
    result->task = task;

    switch (task) {
        case LEMBED_TASK_EMBEDDING: {
            lembed_tuning_result_t emb_result = {0};
            lembed_status_t s = lembed_autotune(model_name, mode, &emb_result);
            if (s != LEMBED_OK) return s;
            result->threads = emb_result.threads;
            result->batch_size = emb_result.batch_size;
            result->workers = emb_result.workers;
            result->throughput_docs_sec = emb_result.throughput_docs_sec;
            result->latency_ms = emb_result.latency_ms;
            result->memory_mb = emb_result.memory_mb;
            return LEMBED_OK;
        }
        case LEMBED_TASK_RERANKING: {
            lembed_reranker_tuning_result_t rerank_result = {0};
            lembed_status_t s = lembed_reranker_autotune(model_name, mode, LEMBED_OBJECTIVE_BALANCED, &rerank_result);
            if (s != LEMBED_OK) return s;
            result->threads = rerank_result.threads;
            result->batch_size = rerank_result.batch_size;
            result->max_tokens = rerank_result.max_tokens;
            result->throughput_docs_sec = rerank_result.throughput_docs_sec;
            result->latency_ms = rerank_result.latency_ms;
            result->p95_latency_ms = rerank_result.p95_latency_ms;
            result->memory_mb = rerank_result.memory_mb;
            return LEMBED_OK;
        }
        case LEMBED_TASK_IMAGE: {
#ifndef LIBEMBEDDING_NO_IMAGE
            lembed_image_tuning_result_t image_result = {0};
            lembed_status_t s = lembed_image_autotune(model_name, mode, &image_result);
            if (s != LEMBED_OK) return s;
            result->threads = image_result.threads;
            result->batch_size = image_result.batch_size;
            result->throughput_docs_sec = image_result.throughput_docs_sec;
            result->latency_ms = image_result.latency_ms;
            result->memory_mb = image_result.memory_mb;
            return LEMBED_OK;
#else
            return LEMBED_ERROR_UNSUPPORTED;
#endif
        }
        case LEMBED_TASK_SPARSE: {
            lembed_sparse_tuning_result_t sparse_result = {0};
            lembed_status_t s = lembed_sparse_autotune(model_name, mode, &sparse_result);
            if (s != LEMBED_OK) return s;
            result->threads = sparse_result.threads;
            result->batch_size = sparse_result.batch_size;
            result->top_k = sparse_result.top_k;
            result->min_weight = sparse_result.min_weight;
            result->storage_format = sparse_result.storage_format;
            result->throughput_docs_sec = sparse_result.throughput_docs_sec;
            result->latency_ms = sparse_result.latency_ms;
            result->memory_mb = sparse_result.memory_mb;
            return LEMBED_OK;
        }
        default:
            return LEMBED_ERROR_INVALID_ARGUMENT;
    }
}

lembed_status_t lembed_autotune_unified_config(
    lembed_task_t task,
    const char* model_name,
    double target_latency_ms,
    lembed_unified_tuning_result_t* result)
{
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(result, 0, sizeof(*result));
    result->task = task;

    switch (task) {
        case LEMBED_TASK_RERANKING: {
            lembed_reranker_tuning_result_t rerank_result = {0};
            lembed_status_t s = lembed_reranker_auto_config(model_name, target_latency_ms, LEMBED_OBJECTIVE_BALANCED, &rerank_result);
            if (s != LEMBED_OK) return s;
            result->threads = rerank_result.threads;
            result->batch_size = rerank_result.batch_size;
            result->max_tokens = rerank_result.max_tokens;
            result->throughput_docs_sec = rerank_result.throughput_docs_sec;
            result->latency_ms = rerank_result.latency_ms;
            result->p95_latency_ms = rerank_result.p95_latency_ms;
            result->memory_mb = rerank_result.memory_mb;
            return LEMBED_OK;
        }
        case LEMBED_TASK_EMBEDDING: {
            lembed_tuning_result_t emb_result = {0};
            lembed_status_t s = lembed_autotune(model_name, LEMBED_AUTOTUNE_QUICK, &emb_result);
            if (s != LEMBED_OK) return s;
            result->threads = emb_result.threads;
            result->batch_size = emb_result.batch_size;
            result->workers = emb_result.workers;
            result->throughput_docs_sec = emb_result.throughput_docs_sec;
            result->latency_ms = emb_result.latency_ms;
            result->memory_mb = emb_result.memory_mb;
            return LEMBED_OK;
        }
        case LEMBED_TASK_IMAGE: {
#ifndef LIBEMBEDDING_NO_IMAGE
            lembed_image_tuning_result_t image_result = {0};
            lembed_status_t s = lembed_image_autotune(model_name, LEMBED_AUTOTUNE_QUICK, &image_result);
            if (s != LEMBED_OK) return s;
            result->threads = image_result.threads;
            result->batch_size = image_result.batch_size;
            result->throughput_docs_sec = image_result.throughput_docs_sec;
            result->latency_ms = image_result.latency_ms;
            result->memory_mb = image_result.memory_mb;
            return LEMBED_OK;
#else
            return LEMBED_ERROR_UNSUPPORTED;
#endif
        }
        case LEMBED_TASK_SPARSE:
            return LEMBED_ERROR_UNSUPPORTED;
        default:
            return LEMBED_ERROR_INVALID_ARGUMENT;
    }
}

void lembed_autotune_unified_clear_cache(lembed_task_t task, const char* model_name) {
    switch (task) {
        case LEMBED_TASK_EMBEDDING:
            lembed_autotune_clear_cache(model_name);
            break;
        case LEMBED_TASK_RERANKING:
            lembed_reranker_autotune_clear_cache(model_name);
            break;
        case LEMBED_TASK_IMAGE:
        case LEMBED_TASK_SPARSE:
            break;
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP */
