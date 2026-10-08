/*
 * libembedding - gguf_inspect entry points
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_INSPECT_ENTRY_HPP
#define LIBEMBEDDING_GGUF_INSPECT_ENTRY_HPP

#include "../../gguf_inspect.h"
#include "gguf_probe.hpp"
#include "gguf_spec.hpp"

#ifdef __cplusplus
extern "C" {
#endif

lembed_status_t lembed_gguf_inspect(const char* path, lembed_gguf_desc_t* desc) {
    if (!desc) return LEMBED_ERROR_INVALID_ARGUMENT;
    memset(desc, 0, sizeof(*desc));
    if (!path) {
        snprintf(desc->diagnostic, sizeof(desc->diagnostic), "null path");
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }

    lembed::gguf::Probe probe;
    if (!probe.open(path)) {
        /* Not an error the caller has to handle specially: an unreadable or
         * malformed file is a refusal reported in the description, so a caller
         * walking a directory keeps going and still gets the files it can read.
         * The distinction matters for `nul`, a directory, or a truncated file. */
        desc->sparse_formula = LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN;
        snprintf(desc->diagnostic, sizeof(desc->diagnostic),
                 "cannot read a GGUF header from '%s'", path);
        return LEMBED_ERROR_IO;
    }

    lembed::gguf::classify(probe, *desc);
    return LEMBED_OK;
}

const char* lembed_gguf_capability_name(lembed_gguf_capabilities_t cap) {
    if (cap & LEMBED_GGUF_CAP_SPLADE) return "sparse";
    if (cap & LEMBED_GGUF_CAP_SPARSE_LINEAR) return "sparse-linear";
    if (cap & LEMBED_GGUF_CAP_COLBERT) return "colbert";
    if (cap & LEMBED_GGUF_CAP_RERANKER) return "reranker";
    if (cap & LEMBED_GGUF_CAP_DENSE) return "dense";
    return "unknown";
}

const char* lembed_gguf_capability_summary(lembed_gguf_desc_t* desc) {
    if (!desc) return "unsupported";
    if (desc->diagnostic[0] && desc->capabilities == LEMBED_GGUF_CAP_NONE) {
        return "unsupported";
    }
    const bool sparse = (desc->capabilities &
                         (LEMBED_GGUF_CAP_SPLADE | LEMBED_GGUF_CAP_SPARSE_LINEAR)) != 0;
    if (sparse && desc->sparse_formula != LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN) {
        return "sparse";
    }
    return (desc->capabilities & LEMBED_GGUF_CAP_DENSE) ? "dense" : "unsupported";
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_GGUF_INSPECT_ENTRY_HPP */