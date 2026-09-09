/*
 * libembedding - C/C++ Embedding Library (header-only on Linux/macOS, shared lib/DLL on Windows)
 * config.h - Version and feature configuration
 *
 * Auteur: David Orel
 * Version: 1.5.8
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CONFIG_H
#define LIBEMBEDDING_CONFIG_H

#define LIBEMBEDDING_VERSION_MAJOR 1
#define LIBEMBEDDING_VERSION_MINOR 5
#define LIBEMBEDDING_VERSION_PATCH 8
#define LIBEMBEDDING_VERSION_STRING "1.5.8"

/* Feature toggles (can be defined before including headers) */

/* Define to disable model downloading (offline-only mode) */
/* #define LIBEMBEDDING_NO_DOWNLOAD */

/* Define to disable image embedding support */
/* #define LIBEMBEDDING_NO_IMAGE */

/* Default configuration values */
#define LEMBED_DEFAULT_BATCH_SIZE    256
#define LEMBED_DEFAULT_MAX_LENGTH    512
#define LEMBED_DEFAULT_CACHE_SUBDIR  "libembedding_cache"
#define LEMBED_MAX_DIM               4096

/* Version string accessor (C API stable) */
#ifdef __cplusplus
extern "C" {
#endif
const char* lembed_version(void);
#ifdef __cplusplus
}
#endif

#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_VERSION_IMPL
#define LIBEMBEDDING_VERSION_IMPL

#ifdef __cplusplus
extern "C" {
#endif

const char* lembed_version(void) {
    return LIBEMBEDDING_VERSION_STRING;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_VERSION_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_CONFIG_H */




