/*
 * libembedding - detail/sparse_postprocess.hpp
 * SPLADE and BGE-M3 sparse embedding postprocessing
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_SPARSE_POSTPROCESS_HPP
#define LIBEMBEDDING_DETAIL_SPARSE_POSTPROCESS_HPP

#include <cmath>
#include <cstdint>
#include <vector>

namespace lembed { namespace detail {

/* Single sparse embedding result */
struct SparseResult {
    std::vector<int32_t> indices;
    std::vector<float>   values;
};

/*
 * SPLADE++ postprocessing
 *
 * model_output: flat [batch x seq_len x vocab_size]
 * attention_mask: flat [batch x seq_len]
 * batch, seq_len, vocab_size: tensor dimensions
 */
inline std::vector<SparseResult> splade_postprocess(
        const float* model_output, const int64_t* attention_mask,
        int batch, int seq_len, int vocab_size) {

    std::vector<SparseResult> results(batch);

    for (int b = 0; b < batch; b++) {
        /* Compute max(relu_log * mask) over sequence dimension for each vocab token */
        std::vector<float> scores(vocab_size, 0.0f);

        for (int s = 0; s < seq_len; s++) {
            if (attention_mask[b * seq_len + s] == 0) continue;

            const float* tok = model_output + ((size_t)b * seq_len + s) * vocab_size;
            for (int v = 0; v < vocab_size; v++) {
                /* relu then log(1 + x) */
                float val = tok[v];
                if (val > 0.0f) {
                    val = std::log(1.0f + val);
                    if (val > scores[v]) scores[v] = val;
                }
            }
        }

        /* Extract nonzero entries */
        for (int v = 0; v < vocab_size; v++) {
            if (scores[v] > 0.0f) {
                results[b].indices.push_back((int32_t)v);
                results[b].values.push_back(scores[v]);
            }
        }
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

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_SPARSE_POSTPROCESS_HPP */




