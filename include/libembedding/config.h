/*
 * libembedding - C/C++ Embedding Library (header-only on Linux/macOS, shared lib/DLL on Windows)
 * config.h - Version and feature configuration
 *
 * Auteur: David Orel
 * Version: 1.10.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CONFIG_H
#define LIBEMBEDDING_CONFIG_H

#define LIBEMBEDDING_VERSION_MAJOR 1
#define LIBEMBEDDING_VERSION_MINOR 10
#define LIBEMBEDDING_VERSION_PATCH 0
#define LIBEMBEDDING_VERSION_STRING "1.9.0"

/* Feature toggles (can be defined before including headers) */

/* Define to disable model downloading (offline-only mode) */
/* #define LIBEMBEDDING_NO_DOWNLOAD */

/* Define to disable image embedding support */
/* #define LIBEMBEDDING_NO_IMAGE */

/* =========================================================================
 * Header-only implementation guard
 *
 * On Linux/macOS the library is an INTERFACE (header-only) target: the user
 * must define LIBEMBEDDING_IMPLEMENTATION in EXACTLY ONE .cpp file:
 *
 *     // libembedding_impl.cpp  (the only file that defines it)
 *     #define LIBEMBEDDING_IMPLEMENTATION
 *     #include <libembedding/libembedding.h>
 *
 * Every other translation unit just includes <libembedding/libembedding.h>.
 * Defining it twice causes duplicate symbols; forgetting it causes link
 * errors. Both mistakes are caught at compile time here and by the
 * `#error` guards at the top of every detail/*_impl.hpp header.
 * On Windows the DLL is prebuilt, so nothing has to be defined.
 * ========================================================================= */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(__cplusplus)
#error "LIBEMBEDDING_IMPLEMENTATION must be defined in a C++ file (.cpp), not in a C file (.c)"
#endif

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

/* ONNX Runtime version the library is linked against, e.g. "1.22.0".
 *
 * Benchmarks and bug reports must record this. Quantized throughput depends on
 * the runtime as much as on the weights: the QDQ fusion that turns
 * DequantizeLinear + MatMulInteger into QLinearMatMul is a runtime feature, so
 * two runs of the same model on different ORT versions are not comparable.
 * Returns a static string; never NULL. Implemented where onnxruntime_c_api.h is
 * available, which is why the declaration lives here but not the definition. */
const char* lembed_onnxruntime_version(void);

/* Name of an execution provider, e.g. "CPU". Returns "unknown" for a value that
 * is not a lembed_execution_provider_t. Benchmarks record it so a result cannot
 * be read as a CPU number when a GPU provider served it. */
const char* lembed_execution_provider_name(int provider);
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

const char* lembed_execution_provider_name(int provider) {
    switch (provider) {
        case 0: return "CPU";              /* LEMBED_PROVIDER_CPU */
        case 1: return "CUDA";             /* LEMBED_PROVIDER_CUDA */
        case 2: return "CoreML";           /* LEMBED_PROVIDER_COREML */
        case 3: return "DirectML";         /* LEMBED_PROVIDER_DIRECTML */
        case 4: return "TensorRT";         /* LEMBED_PROVIDER_TENSORRT */
        case 5: return "llama.cpp";        /* LEMBED_PROVIDER_LLAMACPP */
        default: return "unknown";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_VERSION_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_CONFIG_H */




