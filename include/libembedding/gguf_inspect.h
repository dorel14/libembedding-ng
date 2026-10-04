/*
 * libembedding - gguf_inspect.h
 * Read-only inspection of a GGUF file: what it can do, not what it computes.
 *
 * This header answers one question -- "what does this file contain?" -- and
 * nothing else. Inference lives in detail/gguf_sparse_session.hpp and the public
 * sparse API; keeping the two apart is what stops the format convention from
 * growing into an inference engine nobody can review.
 *
 * It reads no weights. Opening a file here costs a header parse, not a load, so
 * a caller can classify a model before deciding whether to load it.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_INSPECT_H
#define LIBEMBEDDING_GGUF_INSPECT_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Capabilities
 *
 * A bitmask, not an enum. An enum forces every new head type to renumber the
 * values already published in this ABI; a bitmask lets a head be added in a
 * later minor release without touching the ones already shipped. Unknown bits
 * are ignored by older readers, which is exactly the forward compatibility we
 * want from a format that converters keep extending.
 * ========================================================================= */
typedef uint64_t lembed_gguf_capabilities_t;

#define LEMBED_GGUF_CAP_NONE     ((lembed_gguf_capabilities_t)0)

/* The encoder produces one pooled vector per input: a dense embedding. Present
 * in every model this library can read. */
#define LEMBED_GGUF_CAP_DENSE    ((lembed_gguf_capabilities_t)1 << 0)

/* An MLM head is present: encoder -> GELU(W.h + b) -> LayerNorm -> vocabulary
 * projection. This is what makes a SPLADE-style sparse vector computable. */
#define LEMBED_GGUF_CAP_SPLADE   ((lembed_gguf_capabilities_t)1 << 1)

/* A token-embedding-sized vocabulary bias (mlm_bias) exists. Optional: when it
 * is absent the decoder bias is zero. */
#define LEMBED_GGUF_CAP_SPLADE_BIAS ((lembed_gguf_capabilities_t)1 << 2)

/* A sparse_linear head is present: the BGE-M3 style projection, to either the
 * full vocabulary or a single scalar per token. Distinct from SPLADE: the
 * scalar form uses a different formula and must not share its code path. */
#define LEMBED_GGUF_CAP_SPARSE_LINEAR ((lembed_gguf_capabilities_t)1 << 3)

/* A ColBERT head is present: one vector per token rather than one per input. */
#define LEMBED_GGUF_CAP_COLBERT  ((lembed_gguf_capabilities_t)1 << 4)

/* A classifier head is present: the cross-encoder reranker path. */
#define LEMBED_GGUF_CAP_RERANKER ((lembed_gguf_capabilities_t)1 << 5)

/* The vocabulary and token embeddings are tied, so the decoder projection can be
 * reused from token_embd instead of being stored twice. Worth reporting: it is
 * the reason a SPLADE head costs almost nothing on disk. */
#define LEMBED_GGUF_CAP_TIED_DECODER ((lembed_gguf_capabilities_t)1 << 6)

/* Special-token ids were found in the metadata. Without them the specials must
 * not be filtered by guesswork, so a sparse vector would leak them. */
#define LEMBED_GGUF_CAP_SPECIAL_TOKENS ((lembed_gguf_capabilities_t)1 << 7)

/* How the sparse projection is combined over the sequence. The two are not
 * interchangeable and must not share a code path: they differ in the weighting
 * applied before the max over tokens. */
typedef enum {
    /* SPLADE: max over tokens of log(1 + relu(logits)). */
    LEMBED_GGUF_SPARSE_FORMULA_SPLADE = 0,
    /* BGE-M3 scalar form: a per-token weight scattered onto the token id, max
     * over tokens, with NO log(1+x) and no vocabulary projection. */
    LEMBED_GGUF_SPARSE_FORMULA_SCALAR = 1,
    /* Unknown or ambiguous: refuse rather than guess. */
    LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN = 2
} lembed_gguf_sparse_formula_t;

/* =========================================================================
 * Description
 * ========================================================================= */

/* Every field is reported as "present or not" through has_*, because a default
 * that silently stands in for a missing key is how a file gets read as an
 * embedding full of garbage with a success code. has_vocab_size=0 means the
 * metadata did not say; it does NOT mean 30522. */
typedef struct {
    /* general.architecture, e.g. "bert". Empty when absent. */
    char        architecture[64];
    char        name[128];

    int         has_vocab_size;
    int32_t     vocab_size;
    int         has_embedding_length;
    int32_t     embedding_length;
    int         has_block_count;
    int32_t     block_count;
    int         has_context_length;
    int32_t     context_length;
    int         has_colbert_dim;
    int32_t     colbert_dim;

    /* Number of tensors declared by the file, for diagnostics. */
    int64_t     n_tensors;

    /* Bitmask of LEMBED_GGUF_CAP_*. */
    lembed_gguf_capabilities_t capabilities;

    /* Which sparse formula applies, or LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN. */
    lembed_gguf_sparse_formula_t sparse_formula;

    /* Special-token ids, each valid only when the matching has_* below is set. */
    int32_t     pad_token_id;      int has_pad_token_id;
    int32_t     bos_token_id;      int has_bos_token_id;
    int32_t     eos_token_id;      int has_eos_token_id;
    int32_t     cls_token_id;      int has_cls_token_id;
    int32_t     separator_token_id; int has_separator_token_id;

    /* Keys that were looked for under the "<arch>." prefix and not found. Names
     * are joined with '|' so the caller can tell exactly what is missing without
     * the library guessing a default. Empty when nothing was missing. */
    char        missing_hparams[256];

    /* Non-empty when the file is not usable, explaining why. A description with
     * a non-empty diagnostic and LEMBED_GGUF_CAP_NONE is a refusal. */
    char        diagnostic[256];
} lembed_gguf_desc_t;

/* Inspect a GGUF file. Returns LEMBED_OK even for a file that turns out to be
 * unusable: the refusal is reported in desc->diagnostic and desc->capabilities,
 * so a caller can enumerate several files without losing the ones it can read.
 * Returns an error only when the path cannot be read at all. */
lembed_status_t lembed_gguf_inspect(const char* path, lembed_gguf_desc_t* desc);

/* Human-readable name of a capability bit, for logs and error messages.
 * Returns "unknown" for a bit this version does not know. */
const char* lembed_gguf_capability_name(lembed_gguf_capabilities_t cap);

/* "sparse" if the file can produce a SPLADE-style sparse vector, "dense" if it
 * only produces pooled vectors, "unsupported" otherwise. */
const char* lembed_gguf_capability_summary(lembed_gguf_desc_t* desc);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_GGUF_INSPECT_H */

/* ---- Implementation ---- */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(LIBEMBEDDING_GGUF_INSPECT_IMPL)
#define LIBEMBEDDING_GGUF_INSPECT_IMPL
#include "detail/gguf/gguf_spec.hpp"
#include "detail/gguf/gguf_probe.hpp"
#include "detail/gguf/gguf_inspect_entry.hpp"
#endif