/*
 * libembedding - detail/download_manager.hpp
 * Unified download manager for all model types (ONNX + GGUF).
 *
 * Centralizes model resolution, additional file lookup, cache management,
 * and download-or-reuse logic so that all ensure_*_model() C API wrappers
 * share a single code path.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_DOWNLOAD_MANAGER_HPP
#define LIBEMBEDDING_DETAIL_DOWNLOAD_MANAGER_HPP

#include <cstring>
#include <string>
#include <stdexcept>

#include "types.h"
#include "model_registry.h"
#include "downloader.h"
#include "detail/downloader_impl.hpp"

namespace lembed { namespace detail {

/* Progress callback for GGUF downloads (shared) */
inline void gguf_download_progress(float fraction, void* userdata) {
    (void)userdata;
    int bar_width = 40;
    int pos = (int)(bar_width * fraction);
    fprintf(stderr, "\r[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) fprintf(stderr, "=");
        else if (i == pos) fprintf(stderr, ">");
        else fprintf(stderr, " ");
    }
    fprintf(stderr, "] %3.0f%%", fraction * 100.0f);
    if (fraction >= 1.0f) fprintf(stderr, "\n");
    fflush(stderr);
}

/* Lookup additional files for a model from the registry.
 * Returns nullptr if no additional files are needed. */
inline const char* const* get_additional_files(int model_enum, int model_type) {
    for (int i = 0; i < lembed__additional_files_count; i++) {
        if (lembed__additional_files[i].model_enum == model_enum &&
            lembed__additional_files[i].model_type == model_type) {
            return lembed__additional_files[i].files;
        }
    }
    return nullptr;
}

/* =========================================================================
 * Unified model cache manager
 * ========================================================================= */

/* Ensure a text/sparse/image/reranker model is available in cache.
 *
 * Resolves the model from its registry enum, downloads missing files
 * (unless offline), and returns the cache directory containing the model.
 *
 * model_type: one of LEMBED__MODEL_TYPE_TEXT, SPARSE, IMAGE, RERANKER
 * model_enum: the model enum value cast to int
 */
inline std::string ensure_model_by_kind(
        int model_enum,
        int model_type,
        const char* cache_dir,
        bool show_progress,
        bool offline) {

    /* Resolve model info from appropriate registry based on type */
    lembed_model_info_t info;
    lembed_status_t s;

    switch (model_type) {
        case LEMBED__MODEL_TYPE_TEXT:
            s = lembed_get_text_model_info((lembed_text_model_t)model_enum, &info);
            break;
        case LEMBED__MODEL_TYPE_SPARSE:
            s = lembed_get_sparse_model_info((lembed_sparse_model_t)model_enum, &info);
            break;
#ifndef LIBEMBEDDING_NO_IMAGE
        case LEMBED__MODEL_TYPE_IMAGE:
            s = lembed_get_image_model_info((lembed_image_model_t)model_enum, &info);
            break;
#endif
        case LEMBED__MODEL_TYPE_RERANKER:
            s = lembed_get_reranker_model_info((lembed_reranker_model_t)model_enum, &info);
            break;
        default:
            throw std::runtime_error("Unknown model type in download manager");
    }

    if (s != LEMBED_OK) {
        throw std::runtime_error("Unknown model enum: " + std::to_string(model_enum));
    }

    /* Get additional files for this model (ONNX data shards, etc.) */
    const char* const* addl = get_additional_files(model_enum, model_type);

    /* Ensure cache directory and download */
    return ensure_model(
        info.model_code,
        info.model_file,
        addl,
        get_cache_dir(cache_dir),
        show_progress,
        offline);
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_DOWNLOAD_MANAGER_HPP */
