/*
 * libembedding - cpp/embedding_provider.hpp
 * Inline factory for EmbeddingProvider (header-only)
 *
 * Author: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/cpp/provider.hpp>
#include <libembedding/cpp/embedding_model.hpp>
#include <libembedding/cpp/llama_provider.hpp>

#include <algorithm>
#include <cctype>
#include <memory>

namespace lembed
{

/* Implementation of the create_embedding_provider function */
inline std::unique_ptr<EmbeddingProvider> create_embedding_provider(
    const std::string& model,
    const EmbeddingOptions& opts) {

    std::string lower_model = model;
    std::transform(lower_model.begin(), lower_model.end(), lower_model.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    bool use_llama = false;
    if (lower_model.find('/') != std::string::npos) use_llama = true;
    if (lower_model.find("\\") != std::string::npos) use_llama = true;
    auto ends_with = [&](const std::string& s, const std::string& suffix) {
        if (s.size() < suffix.size()) return false;
        return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with(lower_model, ".gguf") || ends_with(lower_model, ".gguf.q4_k_m") ||
        ends_with(lower_model, ".gguf_q4_k_m") || ends_with(lower_model, ".gguf.q8_0")) {
        use_llama = true;
    }

    if (use_llama) {
        return std::make_unique<LlamaEmbeddingProvider>(model, opts);
    } else {
        return std::make_unique<EmbeddingModel>(model, opts);
    }
}

} /* namespace lembed */


