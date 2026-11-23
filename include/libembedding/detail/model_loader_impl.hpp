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
#include "downloader_impl.hpp"

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
 * Parse a sentence_bert_config.json blob (SentenceTransformers) and return its
 * max_seq_length, the truncation length SentenceTransformers and FastEmbed use
 * for the model. Returns 0 when the blob is not JSON, has no usable
 * max_seq_length, or the value is outside [1, 131072].
 */
inline int parse_sentence_bert_max_seq_length(const std::string& blob) {
    cJSON* root = cJSON_Parse(blob.c_str());
    if (!root) return 0;
    int out = 0;
    cJSON* msl = cJSON_GetObjectItem(root, "max_seq_length");
    if (cJSON_IsNumber(msl) && msl->valuedouble >= 1.0 && msl->valuedouble <= 131072.0) {
        out = (int)msl->valuedouble;
    }
    cJSON_Delete(root);
    return out;
}

/*
 * Max token length of an ONNX text model, first non-zero of:
 *   1. the length the caller asked for (options.max_length),
 *   2. max_seq_length of <model_dir>/sentence_bert_config.json, when present,
 *   3. fallback (registry max_tokens, or config.json max_position_embeddings).
 * Returns 0 when all three are unknown (the loader then uses its own default).
 */
inline int resolve_text_max_length(int user_max_length,
                                   const std::string& model_dir,
                                   int fallback) {
    if (user_max_length > 0) return user_max_length;
    std::string path = model_dir + "/sentence_bert_config.json";
    if (file_exists(path)) {
        int sbert = parse_sentence_bert_max_seq_length(read_file_to_string(path));
        if (sbert > 0) return sbert;
    }
    return fallback > 0 ? fallback : 0;
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
