/*
 * libembedding - cpp/provider.hpp
 * C++ wrapper for embedding providers (ONNX vs llama.cpp backend selection).
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CPP_PROVIDER_HPP
#define LIBEMBEDDING_CPP_PROVIDER_HPP

#include <libembedding/types.h>
#include <libembedding/text_embedding.h>
#include <libembedding/model_registry.h>
#include <libembedding/detail/status_helper.hpp>

#include <libembedding/cpp/embedding_model.hpp>
#include <libembedding/cpp/llama_provider.hpp>

namespace lembed
{

/* Parse provider string to enum */
inline lembed_execution_provider_t parse_provider(const std::string& name) {
    if (name == "cpu" || name.empty()) return LEMBED_PROVIDER_CPU;
    if (name == "cuda") return LEMBED_PROVIDER_CUDA;
    if (name == "coreml") return LEMBED_PROVIDER_COREML;
    if (name == "directml") return LEMBED_PROVIDER_DIRECTML;
    if (name == "tensorrt") return LEMBED_PROVIDER_TENSORRT;
    throw std::invalid_argument("Unknown execution provider: " + name);
}

/* Options for creating an EmbeddingProvider */
struct EmbeddingOptions {
    std::string                  model_path;              /* HF name or local dir path */
    int                          threads = 0;
    int                          batch_size = LEMBED_DEFAULT_BATCH_SIZE;
    bool                         offline = false;
    std::string                  cache_dir;
    std::string                  provider = "cpu";
    int                          device_id = 0;
    int                          max_length = 0;
    bool                         show_download_progress = true;
    lembed_pooling_t             pooling = LEMBED_POOLING_MEAN;
    int                          dim = 0;
};

/* Abstract base for embedding providers (ONNX or llama.cpp backend) */
class EmbeddingProvider {
public:
    virtual ~EmbeddingProvider() = default;

    virtual std::vector<std::vector<float>> embed(const std::vector<std::string>& texts,
                                                  int batch_size = 0) = 0;
    virtual int dimension() const = 0;
    virtual std::string name() const = 0;
    virtual void close() = 0;
};

/* Forward declaration of LlamaEmbeddingProvider */
class LlamaEmbeddingProvider;

/* Create an EmbeddingProvider with automatic backend detection
 * Rule: if model_path ends with ".gguf" or contains a "slash" (HF path), choose llama.cpp.
 * Otherwise use ONNX (model registry or local ONNX dir). */
std::unique_ptr<EmbeddingProvider> create_embedding_provider(
    const std::string& model,
    const EmbeddingOptions& opts = {});

} /* namespace lembed */

#endif /* LIBEMBEDDING_CPP_PROVIDER_HPP */