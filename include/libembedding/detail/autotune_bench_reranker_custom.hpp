/*
 * libembedding - detail/autotune_bench_reranker_custom.hpp
 * Reranker auto-tuner with custom corpus
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_CUSTOM_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_CUSTOM_HPP

#include "libembedding/autotuner.h"
#include "libembedding/reranker.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"
#include "autotune_bench_reranker.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Split a user corpus into a query and the documents it ranks.
 *
 * The last text becomes the query and is removed from the document list, so the
 * query is never scored against itself. Returns false when the corpus cannot
 * produce a single (query, document) pair. */
inline bool split_query_and_documents(const std::vector<std::string>& texts,
                                      std::string& query_out,
                                      std::vector<std::string>& docs_out) {
    docs_out.clear();
    query_out.clear();
    if (texts.empty()) return false;

    query_out = texts.back();
    if (texts.size() == 1) {
        /* A single text is duplicated as its own document so the benchmark
         * still has one (query, document) pair to measure. */
        docs_out.push_back(texts[0]);
    } else {
        docs_out.assign(texts.begin(), texts.end() - 1);
    }
    return !docs_out.empty();
}

/* C API: reranker autotune with custom corpus
 *
 * Shares the tuning loop, the scoring and the cache with the default entry
 * point; only the corpus and the cache identity differ. The fingerprint of the
 * corpus is part of the identity, so a tuning measured on one corpus is never
 * served for another. */
inline lembed_status_t lembed_reranker_autotune_custom_impl(
    const char* model_name,
    const char* const* texts,
    int n_texts,
    lembed_autotune_mode_t mode,
    lembed_objective_t objective,
    lembed_reranker_tuning_result_t* result)
{
    if (!model_name || !texts || n_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        std::vector<std::string> corpus;
        corpus.reserve((size_t)n_texts);
        for (int i = 0; i < n_texts; i++)
            corpus.push_back(texts[i] ? texts[i] : "");

        std::string query;
        std::vector<std::string> docs;
        if (!split_query_and_documents(corpus, query, docs))
            return LEMBED_ERROR_INVALID_ARGUMENT;

        std::string canon = canonical_reranker_id(model_name);
        autotune_cache_identity id =
            reranker_cache_identity(canon.c_str(), "custom", corpus_fingerprint(docs),
                                    objective, mode);
        return lembed_reranker_autotune_impl(canon.c_str(), docs, query.c_str(), mode, objective,
                                             id, "reranker_autotune(custom)", result);
    } catch (const std::exception& e) {
        fprintf(stderr, "reranker_autotune_custom: exception: %s\n", e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_RERANKER_CUSTOM_HPP */
