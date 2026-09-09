/*
 * libembedding - detail/session_factory.hpp
 * Unified session creation for ONNX and llama.cpp backends.
 *
 * Provides high-level factory functions that encapsulate the common pattern of
 * backend selection, model resolution, download, and context initialization.
 *
 * Include this header in detail/text_embedding_impl.hpp and detail/reranker implementation.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_SESSION_FACTORY_HPP
#define LIBEMBEDDING_DETAIL_SESSION_FACTORY_HPP

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "This header must be included only when LIBEMBEDDING_IMPLEMENTATION is defined"
#endif

#include "detail/downloader_impl.hpp"
#include "detail/tokenizer_impl.hpp"
#include "detail/onnx_session_impl.hpp"
#include "detail/llama_session_impl.hpp"
#include "detail/model_loader_impl.hpp"
#include "model_registry.h"
#include "downloader.h"
#include "gguf_registry.h"

namespace lembed { namespace detail {

/* =========================================================================
 * Resolve a model's cache directory (ONNX path).
 * Calls the appropriate ensure_*_model function and converts to std::string.
 * Throws std::runtime_error on failure.
 * ========================================================================= */

template<typename ModelEnum>
inline std::string ensure_onnx_model_dir(
        ModelEnum model_enum,
        const char* cache_dir,
        int show_progress,
        int offline,
        const char* const* additional_files = nullptr) {

    /* Resolve model info from registry */
    lembed_model_info_t info;
    lembed_status_t s = lembed_get_text_model_info((lembed_text_model_t)model_enum, &info);
    if (s != LEMBED_OK) {
        /* Try reranker registry as fallback */
        s = lembed_get_reranker_model_info((lembed_reranker_model_t)model_enum, &info);
    }
    if (s != LEMBED_OK) {
        char errbuf[256];
        snprintf(errbuf, sizeof(errbuf),
                 "Unknown model enum value: %d", (int)model_enum);
        throw std::runtime_error(errbuf);
    }

    char* model_dir_cstr = nullptr;
    s = lembed_ensure_text_model((lembed_text_model_t)model_enum, cache_dir,
                                 show_progress, offline, &model_dir_cstr);
    if (s != LEMBED_OK) {
        /* Try reranker registry */
        s = lemembed_ensure_reranker_model((lembed_reranker_model_t)model_enum,
                                          cache_dir, show_progress, offline, &model_dir_cstr);
    }
    if (s != LEMBED_OK) {
        throw std::runtime_error("Failed to ensure model in cache (download may have failed)");
    }

    std::string model_dir(model_dir_cstr);
    lembed_free_string(model_dir_cstr);

    /* If additional files are needed, verify they exist */
    if (additional_files) {
        for (int i = 0; additional_files[i]; i++) {
            std::string add_file = model_dir + "/" + additional_files[i];
            if (!file_exists(add_file)) {
                throw std::runtime_error(
                    "Additional model file missing: " + add_file);
            }
        }
    }

    return model_dir;
}

/* Resolve a GGUF model path from a name/path/URL.
 * Throws std::runtime_error on failure. */
inline std::string resolve_gguf_path(
        const char* gguf_name_or_path,
        const char* cache_dir,
        int offline) {

    char* resolved = nullptr;
    lembed_status_t s = lembed_resolve_gguf_path(
        gguf_name_or_path, cache_dir, offline, &resolved);
    if (s != LEMBED_OK) {
        throw std::runtime_error("Failed to resolve GGUF model path");
    }
    std::string result(resolved);
    lembed_free_string(resolved);
    return result;
}

/* Resolve GGUF model from registry name (e.g. "BGE-Reranker-v2-M3")
 * to its full path via ensure_gguf_model. */
inline std::string resolve_gguf_model(
        const char* model_name,
        const char* cache_dir,
        int show_progress,
        int offline) {

    const lembed_gguf_model_info_t* info = lembed_find_gguf_model(model_name);
    if (!info || !info->gguf_url) {
        char errbuf[256];
        snprintf(errbuf, sizeof(errbuf),
                 "No GGUF model found matching '%s'", model_name);
        throw std::runtime_error(errbuf);
    }

    char* cached_path = nullptr;
    lembed_status_t s = lembed_ensure_gguf_model(
        info->model_code, info->gguf_url, cache_dir,
        show_progress, offline, &cached_path);
    if (s != LEMBED_OK) {
        throw std::runtime_error("Failed to download GGUF model");
    }

    std::string result(cached_path);
    lembed_free_string(cached_path);
    return result;
}

/* Unified backend type resolution.
 * Given an explicit backend and a model name/path, determine the effective
 * backend to use. Returns LEMBED_BACKEND_ONNX or LEMBED_BACKEND_LLAMACPP. */
inline lembed_backend_t resolve_effective_backend(
        lembed_backend_t requested,
        const char* model_name_or_path) {

    if (requested == LEMBED_BACKEND_ONNX || requested == LEMBED_BACKEND_LLAMACPP)
        return requested;

    /* AUTO: detect from path */
    if (model_name_or_path) {
        std::string p = model_name_or_path;
        std::transform(p.begin(), p.end(), p.begin(), ::tolower);
        if (p.size() > 4 && p.substr(p.size() - 4) == ".gguf")
            return LEMBED_BACKEND_LLAMACPP;
        if (p.size() > 5 && p.substr(p.size() - 5) == ".onnx")
            return LEMBED_BACKEND_ONNX;
    }

    /* Default to ONNX for registry models */
    return LEMBED_BACKEND_ONNX;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_SESSION_FACTORY_HPP */
