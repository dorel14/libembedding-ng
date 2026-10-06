/*
 * libembedding - detail/gguf/gguf_bert_graph.hpp
 * The BERT encoder as a ggml graph, post-LayerNorm ordering.
 *
 * The layer order is not a free choice. BERT is post-LN, which is the opposite
 * of every transformer llama.cpp normally runs, and getting it wrong produces a
 * model that loads, runs and returns plausible vectors:
 *
 *   - the QKV projections read the *normalised* input, not the raw residual;
 *   - `ln1` is applied *after* the attention residual has been added;
 *   - `ln2` is applied *after* the feed-forward residual has been added.
 *
 * Both implementations of this graph agree on that ordering: llama.cpp's own
 * BERT builder (src/models/bert.cpp:114-225) does the QKV from the normalised
 * input and applies `attn_output_norm` after the attention residual and
 * `layer_output_norm` after the FFN residual. Copying that is deliberate --
 * this file exists because llama.cpp cannot *load* these files, not because
 * this ordering is preferable.
 *
 * Attention is the explicit mul_mat / soft_max_ext / mul_mat chain rather than
 * ggml_flash_attn_ext. The fused op wants a half-precision mask and a
 * half-precision-friendly layout; the explicit chain runs on F32 throughout and
 * shares its conventions with llama.cpp's non-fused path, so the two can be
 * read side by side. Fusion is a later optimisation, not a correctness
 * question.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_BERT_GRAPH_HPP
#define LIBEMBEDDING_GGUF_BERT_GRAPH_HPP

#include <ggml.h>

#include <cmath>

#include "gguf_weights.hpp"

namespace lembed {
namespace gguf {

/* LayerNorm, with the affine part applied separately.
 *
 * `ggml_norm` already subtracts the mean and divides by the biased variance
 * (ggml-cpu/ops.cpp:3721-3737), so it is a true LayerNorm and not a RMSNorm --
 * worth checking, because the difference is invisible from the name.
 *
 * The weight and bias go through `ggml_mul` / `ggml_add` rather than a fused
 * layernorm op: those go through the generic binary-op path, which accepts a
 * quantised operand, so a file whose LayerNorm scales are stored as Q8_0 still
 * works instead of aborting. */
inline ggml_tensor* bert_layernorm(ggml_context* ctx, ggml_tensor* x,
                                   ggml_tensor* weight, ggml_tensor* bias,
                                   float eps) {
    ggml_tensor* normed = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, normed, weight), bias);
}

/* Splits a [n_embd, seq_len * batch] tensor into
 * [head_dim, seq_len, n_head, batch].
 *
 * The batch is an axis of its own, not a flattened second dimension. It is the
 * last one, which is what keeps the attention scores [seq_len, seq_len, n_head,
 * * batch] -- linear in the batch instead of quadratic in batch * seq_len.
 *
 * The reshape is a genuine view. The flat index of the input is
 *
 *     dim + head_dim * head + n_embd * pos + n_embd * seq_len * doc
 *
 * and of the 4D view is
 *
 *     dim + head_dim * head + head_dim * n_head * pos
 *         + head_dim * n_head * seq_len * doc
 *
 * and the two are identical because n_embd == head_dim * n_head.
 *
 * ggml_reshape_4d cannot produce [head_dim, seq_len, n_head, batch] directly:
 * it repartitions the two input axes as [head_dim, n_head, seq_len, batch], which
 * puts n_head on axis 1 and makes ggml_mul_mat return [n_head, n_head, ...].
 * The head axis has to be moved to axis 2 afterwards, which is what the caller's
 * permute does. */
inline ggml_tensor* bert_split_heads(ggml_context* ctx, ggml_tensor* t,
                                     int n_embd, int n_head, int seq_len,
                                     int batch) {
    const int head_dim = n_embd / n_head;
    ggml_tensor* t4 = ggml_reshape_4d(ctx, t, head_dim, n_head, seq_len, batch);
    return ggml_cont(ctx, ggml_permute(ctx, t4, 0, 2, 1, 3));
}

/* Attention with the batch on its own axis, so the scores are
 * [seq_len, seq_len, n_head, batch] -- linear in the batch instead of quadratic
 * in batch * seq_len.
 *
 * q, k and v arrive as [head_dim, seq_len, n_head, batch] (the output of
 * bert_split_heads). The batch axis is the last one, which is what keeps it out
 * of the reduction: kq = k @ q^T is [seq_len, seq_len, n_head, batch] and
 * v_t @ kq lands back at [head_dim, seq_len, n_head, batch].
 *
 * q and k stay as permuted views. ggml_mul_mat reads its left operand's rows
 * with arbitrary nb1/nb2/nb3 strides (ggml-cpu/ggml-cpu.c:1226-1236) and a
 * permuted view has nb0 == sizeof(float), so the 64-element dot is contiguous;
 * ggml_cont would only be a copy. llama.cpp's own BERT does the same.
 *
 * `mask` is [seq_len, seq_len, 1, batch] F32 and is *added* to the scores before
 * the softmax. The head axis is left at 1: a BERT mask has no per-head
 * component, so every head sees the same plane, and ggml's softmax indexes it
 * with i02 % ne12 (ggml-cpu/ops.cpp:5499-5500), so a single plane is read for
 * every head and never out of bounds. The [seq_len, seq_len, n_head, batch]
 * spelling is legal too but twelve times larger for no numerical difference.
 *
 * A query row that is padding attends to its own document's real tokens instead
 * of to itself, so the softmax is well defined and the row's column is thrown
 * away at pooling. The "mask a padding row against itself only" special case
 * is gone: it existed to avoid a divide by zero, and it is unnecessary now that
 * every document carries at least one real token. */
inline ggml_tensor* bert_attention(ggml_context* ctx,
                                   ggml_tensor* q, ggml_tensor* k,
                                   ggml_tensor* v, ggml_tensor* mask,
                                   int n_embd, int n_head, int seq_len,
                                   int batch) {
    const int head_dim = n_embd / n_head;

    /* k @ q^T: [head_dim, seq_len, n_head, batch] x
     * [head_dim, seq_len, n_head, batch] -> [seq_len, seq_len, n_head, batch]. */
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt((float)head_dim), 0.0f);

    /* v^T @ kq: [seq_len, head_dim, n_head, batch] x
     * [seq_len, seq_len, n_head, batch] -> [head_dim, seq_len, n_head, batch].
     *
     * ggml_transpose swaps axes 0 and 1, which is what we want; ggml_cont then
     * materialises it, because the transposed view is not contiguous and the
     * next mul_mat needs a real [seq_len, ...] matrix on the left. */
    ggml_tensor* v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);

    /* [head_dim, seq_len, n_head, batch] -> [head_dim, n_head, seq_len, batch]
     * -> contiguous [n_embd, seq_len * batch], document-major, the same token
     * order as the input. */
    ggml_tensor* merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, merged, n_embd, seq_len * batch);
}

/* The full encoder: token ids and positions in, last hidden states out.
 *
 * `tokens` and `positions` are I32 [seq_len * batch]; `mask` is F32
 * [seq_len, seq_len, 1, batch]. The returned tensor is F32
 * [n_embd, seq_len * batch] -- the input of the MLM head, column by column,
 * one column per token, document-major.
 *
 * The batch is an axis of its own rather than a flattened second dimension:
 * the attention mask is per document, so the encoder must know where one
 * document ends and the next begins, and keeping the batch axis lets the
 * attention stay [seq_len, seq_len, n_head, batch] instead of
 * [seq_len * batch, seq_len * batch, n_head]. */
inline ggml_tensor* build_bert_encoder(ggml_context* ctx, const Weights& w,
                                       ggml_tensor* tokens, ggml_tensor* positions,
                                       ggml_tensor* mask, int seq_len, int batch) {
    const SpladeHparams& hp = w.hparams();
    const HeadWeights& head = w.head();
    const int n_embd = hp.n_embd;

    /* Embeddings: token + position + token type. */
    ggml_tensor* h = ggml_get_rows(ctx, head.tok_embd, tokens);      /* [H, T] */
    h = ggml_add(ctx, h, ggml_get_rows(ctx, head.pos_embd, positions));
    /* Every document is single-segment, so the token type is always row 0. A
     * view of that row beats gathering a vector of zeros: the broadcast does
     * the same work with no allocation and no index to get wrong. */
    h = ggml_add(ctx, h, ggml_view_1d(ctx, head.type_embd, n_embd, 0));

    /* BERT applies a LayerNorm to the embedding output before the first block;
     * without this the encoder silently disagrees with the ONNX export. */
    h = bert_layernorm(ctx, h, head.embd_ln_w, head.embd_ln_b, hp.eps);

    for (size_t il = 0; il < w.layers().size(); il++) {
        const LayerWeights& L = w.layers()[il];
        ggml_tensor* inp = h;

        ggml_tensor* q = ggml_add(ctx, ggml_mul_mat(ctx, L.q_w, h), L.q_b);
        ggml_tensor* k = ggml_add(ctx, ggml_mul_mat(ctx, L.k_w, h), L.k_b);
        ggml_tensor* v = ggml_add(ctx, ggml_mul_mat(ctx, L.v_w, h), L.v_b);

        q = bert_split_heads(ctx, q, n_embd, hp.n_head, seq_len, batch);
        k = bert_split_heads(ctx, k, n_embd, hp.n_head, seq_len, batch);
        v = bert_split_heads(ctx, v, n_embd, hp.n_head, seq_len, batch);

        ggml_tensor* att = bert_attention(ctx, q, k, v, mask, n_embd, hp.n_head,
                                          seq_len, batch);
        att = ggml_add(ctx, ggml_mul_mat(ctx, L.o_w, att), L.o_b);

        /* ln1 after the attention residual. */
        h = bert_layernorm(ctx, ggml_add(ctx, att, inp), L.ln1_w, L.ln1_b, hp.eps);

        /* ff.gelu is the tanh approximation (ggml-cpu/vec.h:968-970), which is
         * what a BERT export is trained against; ggml_gelu_erf is a different
         * function and must not be substituted. */
        ggml_tensor* f = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, L.fc1_w, h),
                                                 L.fc1_b));
        f = ggml_add(ctx, ggml_mul_mat(ctx, L.fc2_w, f), L.fc2_b);

        /* ln2 after the feed-forward residual. */
        h = bert_layernorm(ctx, ggml_add(ctx, f, h), L.ln2_w, L.ln2_b, hp.eps);
    }

    return h;
}

} /* namespace gguf */
} /* namespace lembed */

#endif /* LIBEMBEDDING_GGUF_BERT_GRAPH_HPP */