#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#ifdef _MSC_VER
/* These headers contain C++ inline helpers in namespace lembed::detail.
 * MSVC emits C4190 because they are included while the umbrella header
 * opens extern "C" — false positive, the functions are not C-linkage. */
#pragma warning(push)
#pragma warning(disable: 4190)
#endif

/* Autotuner C API implementation */
#include "libembedding/detail/autotuner_impl.hpp"
#include "libembedding/detail/autotune_bench_text.hpp"
#include "libembedding/detail/autotune_bench_text_custom.hpp"
#include "libembedding/detail/autotune_bench_reranker_custom.hpp"
#include "libembedding/detail/autotune_cache.hpp"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cstdio>

lembed_status_t lembed_autotune(
        const char* model_name,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    int idx = lembed_find_text_model_by_code(model_name);
    if (idx < 0) return LEMBED_ERROR_MODEL_NOT_FOUND;

    return lembed::detail::autotune_text_impl(
        (lembed_text_model_t)idx,
        {},  /* use generated corpus */
        mode,
        result
    );
}

lembed_status_t lembed_autotune_custom(
        const char* model_name,
        const char* const* texts,
        int n_texts,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    return lembed::detail::lembed_autotune_custom_impl(
        model_name, texts, n_texts, mode, result);
}

lembed_status_t lembed_reranker_autotune_custom(
        const char* model_name,
        const char* const* texts,
        int n_texts,
        lembed_autotune_mode_t mode,
        lembed_objective_t objective,
        lembed_reranker_tuning_result_t* result) {
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    return lembed::detail::lembed_reranker_autotune_custom_impl(
        model_name, texts, n_texts, mode, objective, result);
}

#include "libembedding/detail/model_selector.hpp"

lembed_status_t lembed_auto_select_model(
        const char* use_case,
        lembed_model_selection_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    if (!use_case) use_case = "balanced";

    lembed::detail::ModelSelection sel;
    lembed_status_t s = lembed::detail::auto_select_impl(use_case, sel);
    if (s != LEMBED_OK) return s;

    result->model_code = sel.model_code;
    result->model_name = sel.model_name;
    result->dim = sel.dim;
    result->workers = sel.workers;
    result->threads = sel.threads;
    result->batch_size = sel.batch_size;
    result->throughput_docs_sec = sel.throughput_docs_sec;
    result->latency_ms = sel.latency_ms;
    result->memory_mb = sel.memory_mb;
    result->score = sel.score;

    return LEMBED_OK;
}

/* Autotune cache management */
void lembed_autotune_clear_cache(const char* model_name) {
    lembed::detail::clear_autotune_cache(model_name);
}

/* C-linkage wrappers for autotune unified API (exported from shared lib) */
extern "C" {

lembed_status_t lembed_autotune_unified(
        lembed_task_t task,
        const char* model_name,
        lembed_autotune_mode_t mode,
        lembed_unified_tuning_result_t* result) {
    return lembed::detail::lembed_autotune_unified(task, model_name, mode, result);
}

lembed_status_t lembed_autotune_unified_config(
        lembed_task_t task,
        const char* model_name,
        double target_latency_ms,
        lembed_unified_tuning_result_t* result) {
    return lembed::detail::lembed_autotune_unified_config(task, model_name, target_latency_ms, result);
}

void lembed_autotune_unified_clear_cache(lembed_task_t task, const char* model_name) {
    lembed::detail::lembed_autotune_unified_clear_cache(task, model_name);
}

} /* extern "C" */
