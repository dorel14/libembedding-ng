/*
 * libembedding - detail/autotune_bench_text_custom.hpp
 * Text embedding auto-tuner with custom corpus
 *
 * Auteur: David Orel
 * Version: 1.9.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_CUSTOM_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_CUSTOM_HPP

#include "libembedding/autotuner.h"
#include "libembedding/text_embedding.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"
#include "autotune_bench_text.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace lembed { namespace detail {

/* C API: autotune with custom corpus.
 *
 * Delegates to the standard text tuner: the configuration grid, the scoring
 * and the cache are shared. The only difference is the corpus, and
 * autotune_text_impl() folds its fingerprint into the cache identity, so a
 * tuning measured on a user corpus is never served for the synthetic corpus
 * nor for another user corpus. */
inline lembed_status_t lembed_autotune_custom_impl(
        const char* model_name,
        const char* const* texts,
        int n_texts,
        lembed_autotune_mode_t mode,
        lembed_tuning_result_t* result) {
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    /* Accepts either the HuggingFace repo or the canonical registry name. */
    int idx = lembed_resolve_text_model(model_name);
    if (idx < 0) return LEMBED_ERROR_MODEL_NOT_FOUND;

    std::vector<std::string> corpus;
    corpus.reserve((size_t)n_texts);
    for (int i = 0; i < n_texts; i++)
        corpus.push_back(texts[i] ? texts[i] : "");

    if (result) memset(result, 0, sizeof(*result));
    return autotune_text_impl((lembed_text_model_t)idx, corpus, mode, result);
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_TEXT_CUSTOM_HPP */
