/*
 * libembedding - llamacpp_backend.h
 * llama.cpp backend C API for GGUF embedding models
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_LLAMACPP_BACKEND_H
#define LIBEMBEDDING_LLAMACPP_BACKEND_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Check if llama.cpp backend is available at runtime */
int lembed_llama_backend_available(void);

/* Get llama.cpp version string */
const char* lembed_llama_version(void);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_LLAMACPP_BACKEND_IMPL
#define LIBEMBEDDING_LLAMACPP_BACKEND_IMPL

#ifdef __cplusplus
extern "C" {
#endif

int lembed_llama_backend_available(void) {
    return 1;
}

const char* lembed_llama_version(void) {
    return "llama.cpp enabled";
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_LLAMACPP_BACKEND_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_LLAMACPP_BACKEND_H */




