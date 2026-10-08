/*
 * libembedding - detail/gguf/gguf_spec.hpp
 * The libembedding GGUF convention: which tensors mean what, and what a file is
 * therefore capable of.
 *
 * Three rules shape this file, all of them learned from the sparse GGUF
 * investigation recorded in benchmarks/sparse/LE-9.4-validation.md:
 *
 *  1. *Capabilities come from tensor presence, never from a name or a flag.*
 *     Converters do not agree on how to advertise a head -- the reference
 *     implementation infers `has_sparse` from whether `mlm_transform.weight`
 *     exists and documents no metadata key at all -- so an enum or a
 *     `has_sparse` field would mean trusting each converter in turn. Presence
 *     cannot lie.
 *
 *  2. *A missing key is reported, never defaulted.* A default of 30522 for the
 *     vocabulary reads as a valid SPLADE vector over the wrong vocabulary and
 *     returns a success code. The reference implementation ships exactly that
 *     trap: a hardcoded default with the strict check disabled.
 *
 *  3. *The two sparse formulas are different algorithms.* SPLADE takes
 *     max-over-tokens of log(1 + relu(vocab_logits)); the BGE-M3 scalar form
 *     scatters a per-token weight onto its own token id with no log and no
 *     vocabulary projection. They share a name and nothing else.
 *
 * The head table is data, not code: adding a head is a row, so this file does
 * not grow a switch per head type and no caller has to be edited.
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_SPEC_HPP
#define LIBEMBEDDING_GGUF_SPEC_HPP

#include "../../gguf_inspect.h"
#include "../../types.h"
#include "gguf_probe.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace lembed {
namespace gguf {

/* =========================================================================
 * Capability table
 * ========================================================================= */

/* One row per recognised head. `probe` is the tensor whose presence establishes
 * the capability; `companion` is an optional tensor that refines it without
 * being required (a tied decoder is recorded, but its absence is not an error).
 *
 * Adding a head means adding a row and a bit. It does not mean touching the
 * classification, the report or any caller. */
struct CapabilityRule {
    const char* tensor;          /* presence establishes the capability */
    const char* companion;       /* optional, may be null */
    lembed_gguf_capabilities_t  cap;
};

/* Ordered by specificity, not by bit: the first match wins when a rule is
 * satisfied by more than one row, so a file carrying several heads reports all
 * of them via the accumulate loop below. */
inline constexpr CapabilityRule kCapabilityRules[] = {
    {"mlm_transform.weight", "mlm_bias",            LEMBED_GGUF_CAP_SPLADE},
    {"sparse_linear.weight", nullptr,               LEMBED_GGUF_CAP_SPARSE_LINEAR},
    {"colbert_linear.weight", nullptr,              LEMBED_GGUF_CAP_COLBERT},
    {"classifier.dense.weight", "classifier.out_proj.weight",
                                                    LEMBED_GGUF_CAP_RERANKER},
};

/* Encoder-side tensors that are always required for any of this to mean
 * anything. Their absence means the file is not a text embedding model and no
 * amount of head-hunting will make it one. */
inline constexpr const char* kRequiredEncoderTensors[] = {
    "token_embd.weight",
};

/* =========================================================================
 * Hyper-parameters
 * ========================================================================= */

struct ArchField {
    const char* field;    /* suffix after "<arch>." */
    int* has_out;         /* where to publish presence */
    int32_t* value_out;   /* where to publish the value */
};

/* Reported under "<architecture>.<field>". Presence is tracked per field
 * because a caller must be able to tell "the file says 768" from "we would have
 * guessed 768".
 *
 * Converters spell the same fact differently and there is no reliable spelling:
 * llama.cpp's BERT converter writes `bert.hidden_size` and
 * `bert.num_hidden_layers`, other exports carry `bert.embedding_length` and
 * `bert.block_count`, and the context length appears as either
 * `context_length` or `max_position_embeddings`. A field with known synonyms
 * therefore accepts any of them, first spelling found wins. Requiring one
 * spelling would report a perfectly good file as missing its width or depth. */
inline void read_arch_fields(const Probe& probe, const char* arch,
                             lembed_gguf_desc_t& d) {
    if (!arch || !*arch) return;

    if (probe.get_arch_i32(arch, "vocab_size", d.vocab_size)) d.has_vocab_size = 1;

    if (!probe.get_arch_i32(arch, "embedding_length", d.embedding_length)) {
        if (probe.get_arch_i32(arch, "hidden_size", d.embedding_length))
            d.has_embedding_length = 1;
    } else {
        d.has_embedding_length = 1;
    }

    if (!probe.get_arch_i32(arch, "block_count", d.block_count)) {
        if (probe.get_arch_i32(arch, "num_hidden_layers", d.block_count))
            d.has_block_count = 1;
    } else {
        d.has_block_count = 1;
    }

    if (!probe.get_arch_i32(arch, "context_length", d.context_length)) {
        if (probe.get_arch_i32(arch, "max_position_embeddings", d.context_length))
            d.has_context_length = 1;
    } else {
        d.has_context_length = 1;
    }

    if (probe.get_arch_i32(arch, "colbert_dim", d.colbert_dim)) d.has_colbert_dim = 1;
}

/* Tokenizer special ids. Read from the metadata because the ids are vocabulary
 * specific: BERT has [CLS]=101, XLM-R has <s>=0, and a hardcoded list silently
 * filters the wrong tokens -- or, worse, filters nothing and leaks <pad> and
 * </s> into the sparse vector. */
inline void read_special_tokens(const Probe& probe, lembed_gguf_desc_t& d) {
    struct Field { const char* key; int32_t* id; int* has; };
    const Field fields[] = {
        {"tokenizer.ggml.pad_token_id",      &d.pad_token_id,      &d.has_pad_token_id},
        {"tokenizer.ggml.bos_token_id",      &d.bos_token_id,      &d.has_bos_token_id},
        {"tokenizer.ggml.eos_token_id",      &d.eos_token_id,      &d.has_eos_token_id},
        {"tokenizer.ggml.cls_token_id",      &d.cls_token_id,      &d.has_cls_token_id},
        {"tokenizer.ggml.separator_token_id", &d.separator_token_id,
                                                 &d.has_separator_token_id},
    };
    bool any = false;
    for (const Field& f : fields) {
        int32_t v = 0;
        if (probe.get_i32(f.key, v)) {
            *f.id = v;
            *f.has = 1;
            any = true;
        }
    }
    if (any) d.capabilities |= LEMBED_GGUF_CAP_SPECIAL_TOKENS;
}

/* =========================================================================
 * Classification
 * ========================================================================= */

/* Records what was looked for and not found, as a '|'-joined list the caller
 * can read directly. This is the mechanism behind rule 2: the answer to "can I
 * use this file" must never depend on a value this library chose. */
inline void note_missing(lembed_gguf_desc_t& d, const char* name) {
    const size_t used = strlen(d.missing_hparams);
    const size_t add = strlen(name);
    if (used == 0) {
        snprintf(d.missing_hparams, sizeof(d.missing_hparams), "%s", name);
        return;
    }
    if (used + add + 2 >= sizeof(d.missing_hparams)) return; /* keep the list valid */
    snprintf(d.missing_hparams + used, sizeof(d.missing_hparams) - used, "|%s", name);
}

/* Decides which sparse formula applies, refusing when the file supports both or
 * neither. Returning UNKNOWN is a valid outcome and the caller is expected to
 * refuse the file rather than pick one. */
inline lembed_gguf_sparse_formula_t resolve_formula(const Probe& probe,
                                                    lembed_gguf_capabilities_t caps,
                                                    lembed_gguf_desc_t& d) {
    const bool splade = (caps & LEMBED_GGUF_CAP_SPLADE) != 0;
    const bool linear = (caps & LEMBED_GGUF_CAP_SPARSE_LINEAR) != 0;

    if (splade && linear) {
        /* sparse_linear has two shapes and the two need different maths, so the
         * shape decides which one this is. */
        int64_t out_dim = 0;
        if (probe.tensor_numel("sparse_linear.weight", out_dim) &&
            d.has_vocab_size && out_dim == 1) {
            note_missing(d, "sparse_linear.out_dim");
            return LEMBED_GGUF_SPARSE_FORMULA_SCALAR;
        }
        if (probe.tensor_numel("sparse_linear.weight", out_dim) &&
            d.has_vocab_size && out_dim == (int64_t)d.vocab_size) {
            return LEMBED_GGUF_SPARSE_FORMULA_SPLADE;
        }
        return LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN;
    }
    if (splade) return LEMBED_GGUF_SPARSE_FORMULA_SPLADE;
    if (linear) return LEMBED_GGUF_SPARSE_FORMULA_SCALAR;
    return LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN;
}

/* Fills desc from the file. Exposed separately from the C entry point so tests
 * can drive it with a synthetic header and no filesystem. */
inline void classify(const Probe& probe, lembed_gguf_desc_t& d) {
    memset(&d, 0, sizeof(d));
    d.sparse_formula = LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN;

    if (!probe.valid()) {
        snprintf(d.diagnostic, sizeof(d.diagnostic),
                 "not a readable GGUF file");
        return;
    }

    std::string arch;
    if (probe.get_str("general.architecture", arch)) {
        snprintf(d.architecture, sizeof(d.architecture), "%s", arch.c_str());
    } else {
        note_missing(d, "general.architecture");
    }
    std::string name;
    if (probe.get_str("general.name", name)) {
        snprintf(d.name, sizeof(d.name), "%s", name.c_str());
    }

    d.n_tensors = probe.n_tensors();
    read_arch_fields(probe, arch.c_str(), d);

    /* Without these, "vocab_size" is a number with no meaning attached, so they
     * are required for a sparse claim and recorded as missing otherwise. */
    if (!d.has_vocab_size) note_missing(d, "<arch>.vocab_size");
    if (!d.has_embedding_length) note_missing(d, "<arch>.embedding_length");

    for (const char* t : kRequiredEncoderTensors) {
        if (!probe.has_tensor(t)) {
            note_missing(d, t);
            snprintf(d.diagnostic, sizeof(d.diagnostic),
                     "missing encoder tensor '%s'", t);
            return;
        }
    }

    d.capabilities |= LEMBED_GGUF_CAP_DENSE;

    for (const CapabilityRule& rule : kCapabilityRules) {
        if (!probe.has_tensor(rule.tensor)) continue;
        d.capabilities |= rule.cap;
        if (rule.companion && probe.has_tensor(rule.companion)) {
            if (rule.cap == LEMBED_GGUF_CAP_SPLADE) {
                d.capabilities |= LEMBED_GGUF_CAP_SPLADE_BIAS;
            }
        }
    }

    /* The decoder projection is tied to the token embeddings when no separate
     * vocabulary matrix is stored, which is the normal case for a BERT MLM head
     * and the reason a SPLADE head is nearly free on disk. */
    if ((d.capabilities & LEMBED_GGUF_CAP_SPLADE) && !probe.has_tensor("mlm_head.weight")) {
        d.capabilities |= LEMBED_GGUF_CAP_TIED_DECODER;
    }

    read_special_tokens(probe, d);
    d.sparse_formula = resolve_formula(probe, d.capabilities, d);

    /* A sparse claim without the vocabulary is exactly the case that produces a
     * plausible vector over the wrong tokens, so it is refused here rather than
     * downstream. */
    if ((d.capabilities & (LEMBED_GGUF_CAP_SPLADE | LEMBED_GGUF_CAP_SPARSE_LINEAR)) &&
        !d.has_vocab_size) {
        d.capabilities &= ~(lembed_gguf_capabilities_t)(LEMBED_GGUF_CAP_SPLADE |
                                                        LEMBED_GGUF_CAP_SPARSE_LINEAR);
        d.sparse_formula = LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN;
        snprintf(d.diagnostic, sizeof(d.diagnostic),
                 "sparse head present but <arch>.vocab_size is absent");
    }
}

} /* namespace gguf */
} /* namespace lembed */

#endif /* LIBEMBEDDING_GGUF_SPEC_HPP */