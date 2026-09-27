/*
 * libembedding - detail/pooling.hpp
 * CLS and Mean pooling implementations
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_POOLING_HPP
#define LIBEMBEDDING_DETAIL_POOLING_HPP

#include <cstdint>
#include <cstring>
#include <vector>

namespace lembed { namespace detail {

/*
 * CLS pooling: take the first token's embedding from each sequence.
 *
 * tensor: flat [batch x seq_len x dim] (3D) or [batch x dim] (2D)
 * ndim: 2 or 3
 * out: flat [batch x dim] pre-allocated
 */
inline void pool_cls(const float* tensor, int batch, int seq_len, int dim,
                     int ndim, float* out) {
    if (ndim == 2) {
        /* Already [batch, dim] - just copy */
        std::memcpy(out, tensor, (size_t)batch * dim * sizeof(float));
    } else {
        /* [batch, seq_len, dim] - take [:,0,:] */
        for (int b = 0; b < batch; b++) {
            const float* src = tensor + (size_t)b * seq_len * dim;
            float* dst = out + (size_t)b * dim;
            std::memcpy(dst, src, (size_t)dim * sizeof(float));
        }
    }
}

/*
 * Mean pooling: attention-masked mean of token embeddings.
 *
 * tensor: flat [batch x seq_len x dim] (3D) or [batch x dim] (2D)
 * mask:   flat [batch x seq_len] int64 attention mask (1=keep, 0=pad)
 * ndim: 2 or 3
 * out: flat [batch x dim] pre-allocated
 */
inline void pool_mean(const float* tensor, const int64_t* mask,
                       int batch, int seq_len, int dim,
                       int ndim, float* out) {
    if (ndim == 2) {
        /* Already pooled - just copy */
        std::memcpy(out, tensor, (size_t)batch * dim * sizeof(float));
        return;
    }

    for (int b = 0; b < batch; b++) {
        float* dst = out + (size_t)b * dim;
        std::memset(dst, 0, (size_t)dim * sizeof(float));

        const float* row_tensor = tensor + (size_t)b * seq_len * dim;
        const int64_t* row_mask = mask + (size_t)b * seq_len;

        float mask_sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            if (row_mask[s]) {
                const float* tok = row_tensor + (size_t)s * dim;
                for (int d = 0; d < dim; d++) {
                    dst[d] += tok[d];
                }
                mask_sum += 1.0f;
            }
        }

        float inv = (mask_sum > 1e-9f) ? (1.0f / mask_sum) : 1.0f;
        for (int d = 0; d < dim; d++) {
            dst[d] *= inv;
        }
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_POOLING_HPP */

