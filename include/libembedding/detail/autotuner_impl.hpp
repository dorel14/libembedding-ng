/*
 * libembedding - detail/autotuner_impl.hpp
 * Unified auto-tuner: one router over the four task tuners.
 *
 * lembed_autotune_ex() is the single implementation; lembed_autotune_unified()
 * and lembed_autotune_unified_config() are facades that fill in the options
 * their signature cannot express.
 *
 * Auteur: David Orel
 * Version: 1.12.2
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP

#include "libembedding/autotuner.h"
#include "autotune_cache.hpp"
#include "autotune_result.hpp"
/* Order matters: autotune_bench_text.hpp pulls the model/ONNX/llama impl chain
 * that autotune_bench_reranker.hpp and the sparse/image headers rely on. Keep
 * the text header first. */
#include "autotune_bench_text.hpp"
#include "autotune_bench_reranker.hpp"
#include "autotune_bench_reranker_custom.hpp"
#include "autotune_bench_sparse.hpp"
#ifndef LIBEMBEDDING_NO_IMAGE
#include "autotune_bench_image.hpp"
#endif

#include <cstring>
#include <string>
#include <vector>

namespace lembed { namespace detail {

inline lembed_autotune_options_t autotune_options_default(lembed_task_t task) {
    lembed_autotune_options_t o;
    std::memset(&o, 0, sizeof(o));
    o.task = task;
    o.mode = LEMBED_AUTOTUNE_QUICK;
    o.objective = (task == LEMBED_TASK_RERANKING) ? LEMBED_OBJECTIVE_BALANCED
                                                  : LEMBED_OBJECTIVE_THROUGHPUT;
    return o;
}

inline bool valid_objective(lembed_objective_t o) {
    return o == LEMBED_OBJECTIVE_LATENCY || o == LEMBED_OBJECTIVE_THROUGHPUT ||
           o == LEMBED_OBJECTIVE_BALANCED || o == LEMBED_OBJECTIVE_MEMORY;
}

inline std::vector<std::string> copy_texts(const char* const* texts, int n) {
    std::vector<std::string> out;
    if (!texts || n <= 0) return out;
    out.reserve((size_t)n);
    for (int i = 0; i < n; i++) out.push_back(texts[i] ? texts[i] : "");
    return out;
}

inline lembed_status_t autotune_ex_impl(const char* model_name,
                                        const lembed_autotune_options_t* options,
                                        lembed_unified_tuning_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    lembed_autotune_options_t o =
        options ? *options : autotune_options_default(LEMBED_TASK_EMBEDDING);
    *result = make_unified_result(o.task);

    const lembed_status_t bad = LEMBED_ERROR_INVALID_ARGUMENT;
    if (!model_name) return autotune_fail(bad, "autotune: model_name is NULL");
    if (o.mode != LEMBED_AUTOTUNE_QUICK && o.mode != LEMBED_AUTOTUNE_FULL)
        return autotune_fail(bad, "autotune: mode must be QUICK or FULL");
    if (!valid_objective(o.objective)) return autotune_fail(bad, "autotune: unknown objective");
    if (o.n_texts < 0 || (o.n_texts > 0 && !o.texts))
        return autotune_fail(bad, "autotune: texts/n_texts are inconsistent");
    if (o.max_latency_ms < 0.0 || o.min_tokens < 0)
        return autotune_fail(bad, "autotune: max_latency_ms and min_tokens must be >= 0");
    if (o.min_tokens > 0 && o.task != LEMBED_TASK_RERANKING)
        return autotune_fail(bad, "autotune: min_tokens only applies to RERANKING");

    return autotune_guard("autotune", [&]() -> lembed_status_t {
        std::vector<std::string> texts = copy_texts(o.texts, o.n_texts);
        switch (o.task) {
            case LEMBED_TASK_EMBEDDING: {
                int idx = lembed_resolve_text_model(model_name);
                if (idx < 0)
                    return autotune_fail(LEMBED_ERROR_MODEL_NOT_FOUND,
                                         std::string("autotune: unknown text model '") +
                                             model_name + "'");
                return text_autotune_run((lembed_text_model_t)idx, texts, o.mode, o.objective,
                                         o.max_latency_ms, result);
            }
            case LEMBED_TASK_RERANKING: {
                reranker_tune_request req;
                req.model_name = model_name;
                req.mode = o.mode;
                req.objective = o.objective;
                req.min_tokens = o.min_tokens;
                req.max_latency_ms = o.max_latency_ms;
                std::string query;
                std::vector<std::string> docs;
                if (!texts.empty()) {
                    if (!split_query_and_documents(texts, query, docs))
                        return autotune_fail(bad, "autotune: reranking workload is empty");
                    req.docs = &docs;
                    req.query = query.c_str();
                    req.label = "reranker_autotune(custom)";
                }
                return reranker_autotune_run(req, result);
            }
            case LEMBED_TASK_SPARSE:
                return sparse_autotune_run(model_name, texts, o.mode, o.objective,
                                           o.max_latency_ms, result);
            case LEMBED_TASK_IMAGE:
#ifndef LIBEMBEDDING_NO_IMAGE
                if (!texts.empty())
                    return autotune_fail(bad, "autotune: IMAGE takes no custom text workload");
                return image_autotune_run(model_name, o.mode, o.objective, o.max_latency_ms,
                                          result);
#else
                return autotune_fail(LEMBED_ERROR_UNSUPPORTED,
                                     "autotune: built without image support");
#endif
            default:
                return autotune_fail(bad, "autotune: unknown task");
        }
    });
}

/* =========================================================================
 * Facades
 * ========================================================================= */

inline lembed_status_t lembed_autotune_unified(lembed_task_t task, const char* model_name,
                                               lembed_autotune_mode_t mode,
                                               lembed_unified_tuning_result_t* result) {
    lembed_autotune_options_t o = autotune_options_default(task);
    o.mode = mode;
    return autotune_ex_impl(model_name, &o, result);
}

/* Latency-budget variant, QUICK grid, task's default objective. */
inline lembed_status_t lembed_autotune_unified_config(lembed_task_t task, const char* model_name,
                                                      double target_latency_ms,
                                                      lembed_unified_tuning_result_t* result) {
    lembed_autotune_options_t o = autotune_options_default(task);
    o.max_latency_ms = target_latency_ms > 0.0 ? target_latency_ms : 0.0;
    return autotune_ex_impl(model_name, &o, result);
}

/* Task-scoped purge. For EMBEDDING, a model is purged through
 * lembed_autotune_clear_cache() (it resolves both name forms); NULL removes the
 * root-level text entries only. lembed_autotune_clear_cache(NULL) would wipe
 * every task's cache, which is not what a task-scoped call promises. */
inline void lembed_autotune_unified_clear_cache(lembed_task_t task, const char* model_name) {
    switch (task) {
        case LEMBED_TASK_EMBEDDING:
            autotune_guard("autotune_clear_cache", [&]() {
                if (model_name) lembed_autotune_clear_cache(model_name);
                else clear_autotune_cache_root_entries();
                return LEMBED_OK;
            });
            break;
        case LEMBED_TASK_RERANKING:
            lembed_reranker_autotune_clear_cache(model_name);
            break;
        case LEMBED_TASK_SPARSE:
            lembed_sparse_autotune_clear_cache(model_name);
            break;
        case LEMBED_TASK_IMAGE:
#ifndef LIBEMBEDDING_NO_IMAGE
            lembed_image_autotune_clear_cache(model_name);
#endif
            break;
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNER_IMPL_HPP */
