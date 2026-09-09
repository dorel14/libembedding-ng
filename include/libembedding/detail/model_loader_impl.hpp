/*
 * libembedding - detail/model_loader_impl.hpp
 * Local model loading implementation helpers (create_from_path utilities)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_MODEL_LOADER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_MODEL_LOADER_IMPL_HPP

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "This header must be included only when LIBEMBEDDING_IMPLEMENTATION is defined"
#endif

#include <string>

#include "cJSON.h"
#include "detail/downloader_impl.hpp"

namespace lembed { namespace detail {

/*
 * Parse config.json blob to extract model dimension and max position embeddings.
 * Uses cJSON (bundled). Returns true if the blob was parsed successfully.
 */
inline bool parse_config_json(const std::string& config_blob,
                              int* out_dim, int* out_max_length) {
    cJSON* root = cJSON_Parse(config_blob.c_str());
    if (!root) return false;

    cJSON* hs = cJSON_GetObjectItem(root, "hidden_size");
    if (cJSON_IsNumber(hs) && hs->valueint > 0) {
        *out_dim = hs->valueint;
    }

    cJSON* mpe = cJSON_GetObjectItem(root, "max_position_embeddings");
    if (cJSON_IsNumber(mpe) && mpe->valueint > 0) {
        *out_max_length = mpe->valueint;
    }

    cJSON_Delete(root);
    return true;
}

/*
 * Infer pooling strategy from the directory path basename.
 * Heuristic: BGE, Snowflake Arctic, GTE, MXBAI models use CLS pooling.
 * All others default to MEAN.
 */
inline lembed_pooling_t infer_pooling_from_path(const std::string& path) {
    std::string basename = path;
    size_t pos = basename.find_last_of("/\\");
    if (pos != std::string::npos) basename = basename.substr(pos + 1);
    for (auto& c : basename) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }

    if (basename.find("bge") != std::string::npos ||
        basename.find("snowflake") != std::string::npos ||
        basename.find("arctic") != std::string::npos ||
        basename.find("gte") != std::string::npos ||
        basename.find("mxbai") != std::string::npos) {
        return LEMBED_POOLING_CLS;
    }
    return LEMBED_POOLING_MEAN;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_MODEL_LOADER_IMPL_HPP */
