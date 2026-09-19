---
nav_exclude: true
---

# Home

> **Français:** [Accueil](index.html)

libembedding is a C/C++ and Python library for generating dense, sparse, and image embeddings from ONNX and GGUF (llama.cpp) models.

## Features

| Feature | Description |
|---------|-------------|
| **Text embeddings** | 44 text models (quantized and FP32) |
| **Sparse embeddings** | SPLADE++ and BGE-M3 sparse |
| **Image embeddings** | CLIP, ResNet, Unicom, Nomic Vision |
| **Reranking** | BGE, Jina rerankers |
| **Auto-tuning** | Find optimal config (workers, threads, batch) |
| **LRU Cache** | Thread-safe cache for frequent embeddings |
| **Multi-backend** | ONNX Runtime + llama.cpp |
| **Similarity** | Cosine, dot product, Euclidean |

## Navigation

- [Getting Started](getting_started.html) — Installation and first usage
- [Python API Reference](api_reference.html) — Complete Python API reference
- [Available Models](models.html) — List of supported models
- [Performance Tuning](performance_tuning.html) — Optimizations and configs
- [Advanced Usage](advanced_usage.html) — Cache, modes, workers, autotune
- [Similarity](similarity.html) — Vector comparison
- [Embedding Cache](python/cache.html) — Embedding cache
- [Benchmark](python/benchmark.html) — Backend comparison
- [Backend Detection](python/backend.html) — Auto-detection
- [Runtime Statistics](python/stats.html) — Runtime metrics
- [C API](c_api/similarity.html) — C API reference
- [Documentation française](index.html)
