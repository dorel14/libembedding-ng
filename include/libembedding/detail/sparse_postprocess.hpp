/*
 * libembedding - detail/sparse_postprocess.hpp
 * SPLADE and BGE-M3 sparse embedding postprocessing
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_SPARSE_POSTPROCESS_HPP
#define LIBEMBEDDING_DETAIL_SPARSE_POSTPROCESS_HPP

#include "../types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace lembed { namespace detail {

/* Single sparse embedding result */
struct SparseResult {
    std::vector<int32_t> indices;
    std::vector<float>   values;
};

/*
 * SPLADE pooling: max over a document's tokens of log(1 + relu(logit)).
 *
 * This is the whole SPLADE formula, and it is shared by the ONNX backend and the
 * GGUF runtime on purpose. Two implementations of "max over tokens of
 * log(1 + relu)" would be free to drift, and the drift would only show up as a
 * fidelity number nobody could explain.
 *
 * `logits` points at the first token of one document and is laid out
 * [num_tokens, vocab_size] row-major, which is both what an ONNX output buffer
 * looks like and what a ggml [vocab_size, n_tokens] tensor looks like in memory
 * (dimension 0 is contiguous).
 *
 * `token_mask[i] != 0` keeps token i. Padding is expressed here rather than by
 * trimming the sequence, so a padded batch and a ragged one take the same path.
 */
inline void splade_pool(const float* logits, int vocab_size,
                        const int64_t* token_mask, int num_tokens,
                        SparseResult& out) {
    out.indices.clear();
    out.values.clear();
    if (vocab_size <= 0 || num_tokens <= 0 || !logits) return;

    std::vector<float> scores((size_t)vocab_size, 0.0f);

    for (int s = 0; s < num_tokens; s++) {
        if (token_mask && token_mask[s] == 0) continue;

        const float* tok = logits + (size_t)s * (size_t)vocab_size;
        for (int v = 0; v < vocab_size; v++) {
            /* relu then log(1 + x) */
            const float x = tok[v];
            if (x <= 0.0f) continue;
            const float val = std::log(1.0f + x);
            if (val > scores[v]) scores[v] = val;
        }
    }

    for (int v = 0; v < vocab_size; v++) {
        if (scores[v] > 0.0f) {
            out.indices.push_back((int32_t)v);
            out.values.push_back(scores[v]);
        }
    }
}

/*
 * SPLADE++ postprocessing over a whole [batch x seq_len x vocab_size] output.
 *
 * model_output: flat [batch x seq_len x vocab_size]
 * attention_mask: flat [batch x seq_len]
 * batch, seq_len, vocab_size: tensor dimensions
 */
inline std::vector<SparseResult> splade_postprocess(
        const float* model_output, const int64_t* attention_mask,
        int batch, int seq_len, int vocab_size) {

    std::vector<SparseResult> results((size_t)batch);
    for (int b = 0; b < batch; b++) {
        splade_pool(model_output + (size_t)b * (size_t)seq_len * (size_t)vocab_size,
                    vocab_size,
                    attention_mask ? attention_mask + (size_t)b * (size_t)seq_len
                                   : nullptr,
                    seq_len, results[(size_t)b]);
    }
    return results;
}

/*
 * Generic sparse postprocessing: extract nonzero elements from
 * a [batch x vocab_size] score matrix (after model-specific processing)
 */
inline std::vector<SparseResult> extract_sparse(
        const float* scores, int batch, int vocab_size, float threshold = 0.0f) {

    std::vector<SparseResult> results(batch);

    for (int b = 0; b < batch; b++) {
        const float* row = scores + (size_t)b * vocab_size;
        for (int v = 0; v < vocab_size; v++) {
            if (row[v] > threshold) {
                results[b].indices.push_back((int32_t)v);
                results[b].values.push_back(row[v]);
            }
        }
    }

return results;
}

/*
 * Pruning and ordering of a sparse vector, shared by every sparse backend.
 *
 * This used to live inline in lembed_sparse_text_embedding_embed(), which meant
 * the GGUF backend had no way to produce the same shape of output as the ONNX
 * one without copy-pasting it. One implementation, one behaviour: the two
 * backends cannot drift.
 *
 * Order of operations matters and is part of the contract:
 *   1. min_weight drops the lightest terms,
 *   2. top_k keeps the heaviest survivors,
 *   3. storage_format orders what is left (descending weight, or ascending
 *      index).
 * Applying top_k before the ordering is why top_k returns a descending-by-
 * weight vector even in LEMBED_SPARSE_FORMAT_INDEX_ORDER.
 */
inline void sparse_prune_and_sort(SparseResult& sr, int top_k, float min_weight,
                                  int storage_format) {
    /* Apply min_weight pruning */
    if (min_weight > 0.0f) {
        std::vector<int32_t> filt_idx;
        std::vector<float> filt_val;
        filt_idx.reserve(sr.indices.size());
        filt_val.reserve(sr.values.size());
        for (size_t j = 0; j < sr.indices.size(); j++) {
            if (sr.values[j] >= min_weight) {
                filt_idx.push_back(sr.indices[j]);
                filt_val.push_back(sr.values[j]);
            }
        }
        sr.indices = std::move(filt_idx);
        sr.values = std::move(filt_val);
    }

    /* Apply top_k selection */
    if (top_k > 0 && (int)sr.indices.size() > top_k) {
        std::vector<std::pair<float, int32_t>> val_idx;
        val_idx.reserve(sr.indices.size());
        for (size_t j = 0; j < sr.indices.size(); j++) {
            val_idx.emplace_back(sr.values[j], sr.indices[j]);
        }
        std::partial_sort(
            val_idx.begin(),
            val_idx.begin() + top_k,
            val_idx.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
        std::vector<int32_t> filt_idx(top_k);
        std::vector<float> filt_val(top_k);
        for (int k = 0; k < top_k; k++) {
            filt_idx[k] = val_idx[k].second;
            filt_val[k] = val_idx[k].first;
        }
        sr.indices = std::move(filt_idx);
        sr.values = std::move(filt_val);
    }

    /* Apply storage format: DICT (default) sorts by weight descending,
     * INDEX_ORDER sorts by index ascending */
    if (storage_format == LEMBED_SPARSE_FORMAT_INDEX_ORDER) {
        std::vector<std::pair<int32_t, float>> idx_val;
        idx_val.reserve(sr.indices.size());
        for (size_t j = 0; j < sr.indices.size(); j++) {
            idx_val.emplace_back(sr.indices[j], sr.values[j]);
        }
        std::sort(idx_val.begin(), idx_val.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        for (size_t j = 0; j < sr.indices.size(); j++) {
            sr.indices[j] = idx_val[j].first;
            sr.values[j] = idx_val[j].second;
        }
    } else {
        /* LEMBED_SPARSE_FORMAT_DICT (0) or unknown: sort by weight descending */
        std::vector<std::pair<float, int32_t>> val_idx;
        val_idx.reserve(sr.indices.size());
        for (size_t j = 0; j < sr.indices.size(); j++) {
            val_idx.emplace_back(sr.values[j], sr.indices[j]);
        }
        std::sort(val_idx.begin(), val_idx.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first;
                  });
        for (size_t j = 0; j < sr.indices.size(); j++) {
            sr.indices[j] = val_idx[j].second;
            sr.values[j] = val_idx[j].first;
        }
    }
}

/*
 * Moves one SparseResult into a slot of a C result array.
 *
 * Shared so that a backend can never publish a length that does not match an
 * allocated buffer, and so that an empty vector is represented the same way on
 * both sides: length 0 with null pointers.
 */
inline void sparse_result_store(lembed_sparse_embeddings_t* result, int index,
                                const SparseResult& sr) {
    result->items[index].length = 0;
    result->items[index].indices = nullptr;
    result->items[index].values = nullptr;
    if (sr.indices.empty()) return;

    result->items[index].indices =
        (int32_t*)malloc(sr.indices.size() * sizeof(int32_t));
    result->items[index].values =
        (float*)malloc(sr.values.size() * sizeof(float));
    if (!result->items[index].indices || !result->items[index].values) {
        free(result->items[index].indices);
        free(result->items[index].values);
        result->items[index].indices = nullptr;
        result->items[index].values = nullptr;
        return;
    }
    result->items[index].length = (int)sr.indices.size();
    memcpy(result->items[index].indices, sr.indices.data(),
           sr.indices.size() * sizeof(int32_t));
    memcpy(result->items[index].values, sr.values.data(),
           sr.values.size() * sizeof(float));
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_SPARSE_POSTPROCESS_HPP */




