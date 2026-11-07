/*
 * libembedding - detail/sparse_batching.hpp
 * How the sparse backends split a length-sorted corpus into batches.
 *
 * Shared by the ONNX and the GGUF runtime on purpose, like splade_pool(): both
 * materialise a [batch, seq_len, vocab] F32 logits tensor, so both pay for the
 * padded width and not for the number of texts, and one policy cannot drift
 * from the other. No ggml or ONNX Runtime dependency, so it is tested on its
 * own: the vectors do not depend on it (a batch of N gives the same vector as a
 * batch of one), the cost does.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_SPARSE_BATCHING_HPP
#define LIBEMBEDDING_DETAIL_SPARSE_BATCHING_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace lembed {
namespace detail {

/* Default cap on batch * seq_len for both sparse backends. Measured, not
 * guessed (i7-1065G7, 4 threads, docs/gguf_sparse_runtime.md): over
 * 128..2048 tokens, 512 gave the best throughput on short texts for ONNX
 * Runtime (41.3 docs/s against 22.6 with the old uncapped batch of 256) and
 * for ggml alike, while long texts were compute-bound and flat. Peak RSS fell
 * from 1.6-4 GB to under 900 MB. */
constexpr int kSparseMaxBatchTokens = 512;

/* Same cap for the dense ONNX path (all-MiniLM-L6-v2, i7-1065G7, 4 threads, 256
 * texts, library defaults, sweep 0/256/512/1024/2048, two passes in opposite
 * order): 512 was the best or tied on mixed (8-160 words, 28.4-29.4 -> 80-83
 * docs/s), short (352-365 -> 495-503) and long (31 -> 35) corpora. Dense output
 * is [batch, seq, dim], so unlike the sparse case this is a compute/cache
 * effect, not a memory one: a 256-text batch used to be padded to the longest
 * text (47104 padded tokens for 19915 real ones). */
constexpr int kDenseMaxBatchTokens = 512;

/* One past the last document of the batch that starts at `start`, out of
 * `count` documents sorted longest first.
 *
 * Two bounds: at most `batch_size` documents, and at most `max_batch_tokens`
 * padded token columns (0 disables that bound). The second one is what keeps
 * the cost proportional to the real work: the [vocab, batch * seq_len] F32
 * logits grow with the padded width, and with the documents sorted longest
 * first that width is `first_length`, the length of the batch's first
 * document.
 *
 * A document longer than the budget still gets a batch of its own rather than
 * being refused, and the result is always past `start` when start < count, so
 * a caller looping on it always makes progress. */
inline size_t sparse_batch_end(size_t start, size_t count, size_t first_length,
                               int batch_size, int max_batch_tokens) {
    if (start >= count) return count;
    const size_t docs = (size_t)std::max(1, batch_size);
    const size_t limit = std::min(count, start + docs);
    if (max_batch_tokens <= 0) return limit;

    const size_t width = std::max<size_t>(1, first_length);
    const size_t fit = std::max<size_t>(1, (size_t)max_batch_tokens / width);
    return std::min(limit, start + fit);
}

/* Processing order for `token_ids`: longest first, stable, so that each batch
 * is padded to its own first document and never mixes a long text with many
 * short ones. Identity when there is nothing to reorder. */
inline std::vector<size_t> sparse_batch_order(
        const std::vector<std::vector<int>>& token_ids, int batch_size) {
    std::vector<size_t> order(token_ids.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    if (batch_size > 1 && order.size() > 1) {
        std::stable_sort(order.begin(), order.end(),
            [&token_ids](size_t a, size_t b) {
                return token_ids[a].size() > token_ids[b].size();
            });
    }
    return order;
}

} /* namespace detail */
} /* namespace lembed */

#endif /* LIBEMBEDDING_DETAIL_SPARSE_BATCHING_HPP */
