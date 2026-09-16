---
nav_exclude: true
---

# libembedding Documentation

Welcome to the **libembedding** documentation, a fast embedding library with both **C/C++** and **Python** APIs powered by ONNX Runtime.

> **Forked from [pacifio/libembedding](https://github.com/pacifio/libembedding).**
> This fork adds Windows support (native DLL), PyPI packaging as `libembedding-ng`, local model loading, runtime introspection, similarity helpers, streaming, a multi-worker pool, an autotuner and automatic model selection, llama.cpp/GGUF backend support, length bucketing, LRU cache, dynamic scheduler, and FAST/BALANCED/QUALITY modes.

## Documentation structure

| Section | File | Description |
|---------|------|-------------|
| **Getting started** | [getting_started.html](getting_started.html) | Installation, prerequisites and quick start |
| **Python API** | [api_reference.html](api_reference.html) | Complete reference of Python classes |
| **Models** | [models.html](models.html) | Catalog of available models (text, image, sparse, reranker) |
| **Performance** | [performance_tuning.html](performance_tuning.html) | Session pool, auto-tuning, automatic model selection |
| **Advanced usage** | [advanced_usage.html](advanced_usage.html) | Local models, providers, cache, offline mode, context managers, GGUF/llama.cpp models |
| **Error handling** | [api_reference.html#error-handling](api_reference.html#error-handling) | Python exception hierarchy |
| **Français** | [../index.html](../index.html) | Documentation française |

## Overview

```python
from libembedding import TextEmbedding, SparseTextEmbedding, Reranker
import numpy as np

# Dense embeddings
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])
print(embeddings.shape)  # (2, 384)

# Sparse embeddings
sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning"])
print(results[0].indices.shape, results[0].values.shape)

# Reranking
reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank("What is deep learning?", [
    "Deep learning uses neural networks",
    "The weather is sunny today",
])
print(ranked[0].score, ranked[0].index)
```

**Performance**: 5-8x faster than fastembed, 3.5x less memory.
