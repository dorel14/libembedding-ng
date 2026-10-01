---
nav_exclude: true
---

# Getting started

## Installation

```bash
pip install libembedding-ng
```

> The PyPI package is **`libembedding-ng`** (imported in Python as `libembedding`).

## Prerequisites

libembedding requires **ONNX Runtime** to be installed on your system:

```bash
# macOS
brew install onnxruntime

# Ubuntu / Debian
sudo apt install libonnxruntime-dev

# Windows (via vcpkg or direct download)
# Set ONNXRUNTIME_ROOT if necessary
```

> **Note:** If ONNX Runtime is not found automatically, set the `ONNXRUNTIME_ROOT` environment variable to your installation path.

## llama.cpp backend

The llama.cpp backend is enabled by default and allows loading GGUF models:

```bash
cmake .. -DLIBEMBEDDING_BUILD_SHARED=ON
cmake --build . --parallel
```

Then install the Python bindings:

```bash
pip install libembedding-ng --no-binary :all:  # builds from source with llama.cpp
# or, if a prebuilt wheel with llama.cpp is available:
pip install libembedding-ng
```

Check availability at runtime:

```python
from libembedding import TextEmbedding

# The constructor detects .gguf and routes to the llama.cpp backend.
# Either a bare local path...
model = TextEmbedding("/path/to/all-MiniLM-L6-v2-Q4_K_M.gguf")

# ...or "repo/filename.gguf" to fetch from HuggingFace.
model = TextEmbedding(
    "Xenova/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2-Q4_K_M.gguf"
)
```

## Verifying the installation

```python
import libembedding
print(libembedding.__version__)

# List available models
for m in libembedding.list_text_models()[:3]:
    print(m.model_name, m.dim)
```

## First steps

### Dense text embeddings

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed([
    "The cat sat on the mat",
    "A kitten was sitting on a rug",
    "Quantum physics describes subatomic particles",
])

print(embeddings.shape)   # (3, 384)
print(embeddings.dtype)   # float32

# Cosine similarity
similarity = np.dot(embeddings[0], embeddings[1])  # ≈ 0.82
```

### Sparse embeddings

```python
from libembedding import SparseTextEmbedding

sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
    # (N,) (N,) where N is the number of active tokens
```

### Image embeddings

```python
from libembedding import ImageEmbedding

img_model = ImageEmbedding()
embeddings = img_model.embed_files(["photo.jpg", "diagram.png"])
print(embeddings.shape)  # (2, 512)
```

### Reranking

```python
from libembedding import Reranker

reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank(
    "What is deep learning?",
    [
        "Deep learning uses neural networks with many layers",
        "The weather is sunny today",
        "Neural networks are inspired by biological brains",
    ],
)

for r in ranked:
    print(f"doc[{r.index}] score={r.score:.4f}")
```

## Using a context manager

All classes support the context manager protocol for automatic resource management:

```python
with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
    embeddings = model.embed(["Hello world"])
# C resources are released automatically
```

## Full example

See [examples/python/example.py](https://github.com/dorel14/libembedding/blob/main/examples/python/example.py) for a detailed example with semantic search.
