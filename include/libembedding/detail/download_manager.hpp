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
#include <utility>
#include <vector>

#include "libembedding/types.h"
#include "libembedding/model_registry.h"
#include "libembedding/downloader.h"
#include "downloader_impl.hpp"

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
 * Mobile model limit
 *
 * On Android and iOS a registry model above lembed_mobile_model_limit_mb() is
 * refused before anything is downloaded or loaded: a phone runs out of memory
 * long before the model fails on its own, and the process is then killed with
 * no error at all. The message names the light models of the same kind, so the
 * caller learns what to use instead. Local paths never come through here.
 * ========================================================================= */

/* Distinct type so the C wrappers report LEMBED_ERROR_UNSUPPORTED, not a
 * download failure: a caller that retries or skips on download errors must
 * not do so here. */
struct model_too_large_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void check_mobile_model_limit(const lembed_model_info_t& info, int model_type) {
    const int limit = lembed_mobile_model_limit_mb();
    if (limit <= 0) return;
    const int size = lembed_model_download_size_mb(info.model_code, info.model_file);
    if (size <= limit) return;  /* light, or unknown (-1): not refused */

    const lembed_model_info_t* models = nullptr;
    int count = 0;
    const char* kind = "";
    switch (model_type) {
        case LEMBED__MODEL_TYPE_TEXT:     lembed_list_text_models(&models, &count);     kind = "text"; break;
        case LEMBED__MODEL_TYPE_SPARSE:   lembed_list_sparse_models(&models, &count);   kind = "sparse"; break;
#ifndef LIBEMBEDDING_NO_IMAGE
        case LEMBED__MODEL_TYPE_IMAGE:    lembed_list_image_models(&models, &count);    kind = "image"; break;
#endif
        case LEMBED__MODEL_TYPE_RERANKER: lembed_list_reranker_models(&models, &count); kind = "reranker"; break;
        default: break;
    }
    /* One entry per repo, at its smallest light variant: the thread-local
     * error buffer holds 1024 bytes, and the advice comes first so that a
     * truncation only shortens the list. */
    std::vector<std::pair<std::string, int>> light;
    for (int i = 0; i < count; i++) {
        const int s = lembed_model_download_size_mb(models[i].model_code, models[i].model_file);
        if (s < 0 || s > limit) continue;
        bool seen = false;
        for (auto& entry : light) {
            if (entry.first == models[i].model_code) {
                if (s < entry.second) entry.second = s;
                seen = true;
            }
        }
        if (!seen) light.emplace_back(models[i].model_code, s);
    }
    std::string list;
    for (const auto& entry : light) {
        if (!list.empty()) list += ", ";
        list += entry.first + " (" + std::to_string(entry.second) + " MB)";
    }
    throw model_too_large_error(
        std::string(info.model_code) + " (" + info.model_file + ", " + std::to_string(size) +
        " MB) exceeds the " + std::to_string(limit) + " MB limit for models on mobile devices." +
        " Set LIBEMBEDDING_ALLOW_LARGE_MODELS=1 to load it anyway, or pass a local model path. " +
        (list.empty() ? std::string("No ") + kind + " model of the registry fits that limit."
                      : std::string("Light ") + kind + " models: " + list + "."));
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

    check_mobile_model_limit(info, model_type);

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
