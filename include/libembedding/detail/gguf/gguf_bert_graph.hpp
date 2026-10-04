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

/* Splits a [n_embd, n_tokens] tensor into [n_head, n_embd/n_head, n_tokens].
 *
 * A reshape, not a gather: the head index lands in dimension 1 and n_embd is
 * the product of dimensions 0 and 1, which is exactly how the flat
 * [n_embd, n_tokens] buffer is already laid out. */
inline ggml_tensor* bert_split_heads(ggml_context* ctx, ggml_tensor* t,
                                     int n_embd, int n_head, int n_tokens) {
    return ggml_reshape_3d(ctx, t, n_embd / n_head, n_head, n_tokens);
}

/* Attention over the whole batch of concatenated tokens.
 *
 * q and k come in as [n_head_dim, n_tokens, n_head]; v is transposed to
 * [n_tokens, n_head_dim, n_head] first, because that is the orientation
 * ggml_mul_mat needs on its left operand to produce [n_head_dim, n_tokens,
 * n_head]. This is llama.cpp's own non-fused arrangement (llama-graph.cpp:2607,
 * 2651, 2660-2663), kept so the shapes can be checked against it.
 *
 * `mask` is [n_tokens, n_tokens] F32 and is *added* to the scores before the
 * softmax. */
inline ggml_tensor* bert_attention(ggml_context* ctx,
                                   ggml_tensor* q, ggml_tensor* k,
                                   ggml_tensor* v, ggml_tensor* mask,
                                   int n_embd, int n_head, int n_tokens) {
    const int head_dim = n_embd / n_head;

    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt((float)head_dim), 0.0f);

    /* [n_head_dim, n_tokens, n_head] -> [n_tokens, n_head_dim, n_head].
     *
     * Only dimensions 0 and 1 swap; ggml_transpose is that swap, and ggml_cont
     * is what materialises the result, because a permuted view is not
     * contiguous and the next mul_mat needs a real [n_tokens, ...] matrix on
     * the left. A general permute here is easy to get wrong: permute(2, 0, 1)
     * would move the head axis first and silently produce
     * [n_head, n_head_dim, n_tokens], which fails three calls later with an
     * unrelated-looking assertion in ggml_mul_mat. */
    ggml_tensor* v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);

    /* Back to [n_embd, n_tokens] for the output projection. */
    ggml_tensor* merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, merged, n_embd, n_tokens);
}

/* The full encoder: token ids and positions in, last hidden states out.
 *
 * `tokens` and `positions` are I32 [n_tokens]; `mask` is F32 [n_tokens,
 * n_tokens]. The returned tensor is F32 [n_embd, n_tokens] -- the input of the
 * MLM head, column by column, one column per token.
 *
 * All T tokens of all documents of the batch are processed as a single row.
 * `mask` is what keeps documents apart: it must be block-diagonal by document
 * *and* -infinity on padding. */
inline ggml_tensor* build_bert_encoder(ggml_context* ctx, const Weights& w,
                                       ggml_tensor* tokens, ggml_tensor* positions,
                                       ggml_tensor* mask, int n_tokens) {
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

        q = ggml_permute(ctx, bert_split_heads(ctx, q, n_embd, hp.n_head, n_tokens),
                         0, 2, 1, 3);
        k = ggml_permute(ctx, bert_split_heads(ctx, k, n_embd, hp.n_head, n_tokens),
                         0, 2, 1, 3);
        v = ggml_permute(ctx, bert_split_heads(ctx, v, n_embd, hp.n_head, n_tokens),
                         0, 2, 1, 3);

        ggml_tensor* att = bert_attention(ctx, q, k, v, mask, n_embd, hp.n_head,
                                          n_tokens);
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