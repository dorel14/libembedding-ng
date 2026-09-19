# libembedding

> **Forked from [pacifio/libembedding](https://github.com/pacifio/libembedding).**
> This fork adds Windows support (native DLL), PyPI packaging under the name
> **`libembedding-ng`**, local model loading, runtime introspection, similarity
> helpers, streaming, a multi-worker pool, an autotuner and automatic model
> selection. Imported as `libembedding`.

Fast ONNX-based text, image, and sparse embeddings for Python. **5-8x faster than fastembed** with 3.5x less memory.

Built on a C/C++ backend using ONNX Runtime, exposed to Python via zero-overhead cffi bindings. The wheel bundles the **compiled shared library** (`libembedding.so` / `.dylib` / `.dll`) together with ONNX Runtime and libcurl, so no system ONNX Runtime install is required at runtime. Supports 44 text embedding models, 5 image models, 2 sparse models, and 4 rerankers with automatic model downloading from HuggingFace Hub.

## Installation

```bash
pip install libembedding-ng
```

> The PyPI package is named **`libembedding-ng`** (to avoid clashing with the original
> `libembedding` on PyPI); it is imported as `libembedding`. The wheel already bundles the
> native shared library and ONNX Runtime, so no separate system install is required.

**Requirements:** ONNX Runtime must be installed on your system **only if** you build the package from source.

```bash
# macOS
brew install onnxruntime

# Ubuntu/Debian
apt install libonnxruntime-dev

# Or set ONNXRUNTIME_ROOT to your installation path
```

## Quick Start

### Text Embeddings

```python
from libembedding import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])

print(embeddings.shape)  # (2, 384)
print(embeddings.dtype)  # float32
```

### Sparse Embeddings

```python
from libembedding import SparseTextEmbedding

model = SparseTextEmbedding()
results = model.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
```

### Image Embeddings

```python
from libembedding import ImageEmbedding

model = ImageEmbedding()
embeddings = model.embed_files(["photo.jpg", "diagram.png"])
```

### Reranking

```python
from libembedding import Reranker

reranker = Reranker("BAAI/bge-reranker-base")
results = reranker.rerank(
    "What is deep learning?",
    [
        "Deep learning uses neural networks with many layers",
        "The weather is sunny today",
        "Neural networks are inspired by biological brains",
    ],
)
for r in results:
    print(f"doc[{r.index}] score={r.score:.4f}")
```

### Model Discovery

```python
import libembedding

for m in libembedding.list_text_models():
    print(f"{m.model_name:45} dim={m.dim:<5} {m.pooling}")
```

## API Reference

### Provider

```python
TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # HuggingFace model name, repo code, or local dir path
    provider="cpu",  # "cpu", "cuda", "coreml", "directml", "tensorrt", "llamacpp"
    device_id=0,
    cache_dir=None,  # None = ~/.cache/libembedding
    max_length=0,  # 0 = model default
    threads=0,  # 0 = auto
    batch_size=256,  # internal batch size for embedding
    offline=False,  # True = use cache only, never download
    show_download_progress=True,
    dim=0,  # embedding dim for local models without config.json
    pooling="mean",  # "cls" or "mean" for local models
    num_threads=0,  # deprecated, use threads
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(texts, batch_size=None)` | `np.ndarray (n, dim)` | L2-normalized dense embeddings |
| `embed_stream(texts, batch_size=None)` | generator | Yields one embedding at a time (low memory) |
| `dim` | `int` | Embedding dimension |
| `name` | `str` | Model name or local path |
| `info()` | `ModelDesc` | Runtime model descriptor |
| `max_length()` | `int` | Max token length for the model |
| `stats()` | `Stats` | Runtime statistics (texts, batches, latency) |
| `close()` | `None` | Release resources |
| `from_gguf(repo, filename=None, provider="llamacpp", ...)` | `TextEmbedding` | Load a GGUF model from HuggingFace (requires llama.cpp build) |
| `supports_llamacpp()` | `bool` | (static) Check if llama.cpp backend is available |

### SparseTextEmbedding

```python
SparseTextEmbedding(model_name="prithvida/SPLADE_PP_en_v1", ...)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(texts, batch_size=0)` | `list[SparseEmbedding]` | Sparse vectors with `.indices` and `.values` |
| `embed_stream(texts, batch_size=None)` | generator | Yields one embedding at a time |
| `dim` | `int` | Embedding dimension (0 = dynamic) |
| `name` | `str` | Model name or local path |
| `info()` | `ModelDesc` | Runtime model descriptor |
| `max_length()` | `int` | Max token length |
| `stats()` | `Stats` | Runtime statistics |

### ImageEmbedding

```python
ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision", ...)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed_files(paths, batch_size=0)` | `np.ndarray (n, dim)` | Embed from file paths |
| `embed_bytes(images, batch_size=0)` | `np.ndarray (n, dim)` | Embed from raw bytes |
| `dim` | `int` | Embedding dimension |
| `name` | `str` | Model name or local path |
| `info()` | `ModelDesc` | Runtime model descriptor |
| `stats()` | `Stats` | Runtime statistics |

### Reranker

```python
Reranker(model_name="BAAI/bge-reranker-base", ...)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `rerank(query, documents, batch_size=0)` | `list[RerankResult]` | Sorted by score descending |
| `name` | `str` | Model name or local path |
| `info()` | `ModelDesc` | Runtime model descriptor |
| `max_length()` | `int` | Max token length |
| `stats()` | `Stats` | Runtime statistics |

All classes support context managers (`with TextEmbedding(...) as model:`).

### Similarity Functions

```python
from libembedding import cosine_similarity, dot_product, euclidean_distance
import numpy as np

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([1.0, 2.0, 3.0], dtype=np.float32)

print(cosine_similarity(a, b))  # 1.0 (identical)
print(dot_product(a, b))  # 14.0
print(euclidean_distance(a, b))  # 0.0
```

### Streaming Embeddings

Process large document sets without allocating a single result array:

```python
from libembedding import TextEmbedding

with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
    for embedding in model.embed_stream(["doc1", "doc2", ...], batch_size=32):
        # Each iteration yields a single (dim,) numpy array
        process(embedding)
```

### Runtime Statistics

```python
with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
    model.embed(["text 1", "text 2", "text 3"])
    stats = model.stats()
    print(
        f"Embedded {stats.texts_embedded} texts "
        f"({stats.batches_run} batches), "
        f"avg latency {stats.avg_latency_ms:.2f}ms"
    )
```

### Data Types

**`ModelDesc`** — runtime model descriptor (returned by `info()`):

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Model name or local path |
| `dimension` | `int` | Embedding dimension |
| `max_length` | `int` | Max token length |
| `pooling` | `str` | "cls" or "mean" |
| `num_threads` | `int` | Threads configured |
| `batch_size` | `int` | Batch size configured |
| `provider` | `str` | Execution provider ("cpu", "cuda", "directml", "coreml", "llamacpp") |
| `device_id` | `int` | Device ID |

**`Stats`** — runtime statistics (returned by `stats()`):

| Field | Type | Description |
|---|---|---|
| `texts_embedded` | `int` | Total texts processed |
| `batches_run` | `int` | Total ONNX inference batches |
| `avg_latency_ms` | `float` | Average milliseconds per embed call |

### Local Model Loading

```python
from libembedding import TextEmbedding

# Load from a local directory containing model.onnx + tokenizer.json (+ config.json)
model = TextEmbedding("/path/to/model_dir")
```

## Available Models

**44 text models** including BGE, MiniLM, Nomic, E5, CLIP, Jina, GTE, Snowflake, ModernBERT (with quantized variants).

**6 image models** including CLIP ViT-B/32 (FP32 + INT8 quantized), ResNet-50, Unicom, Nomic Vision.

**2 sparse models**: SPLADE++, BGE-M3.

**5 reranker models**: BGE Reranker, Jina Reranker (including INT8 quantized variant).

**GGUF/llama.cpp models**: Run quantized GGUF models (Q4_K_M, Q8_0) via the llama.cpp backend.

## Benchmarks

Measured on Apple M-series with `all-MiniLM-L6-v2` (384-dim). Median of 10 runs.

| Metric                   | libembedding | fastembed | Speedup |
|--------------------------|-------------|-----------|---------|
| Single text latency (ms) | **4.4**     | 38.0      | **8.6x**|
| Batch 8 (texts/sec)      | **641**     | 92        | **7.0x**|
| Batch 32 (texts/sec)     | **581**     | 89        | **6.5x**|
| Peak RSS (MB)            | **567**     | 1,981     | **3.5x less**|

### Unified Auto-Tuning

Auto-tune threads, batch_size, and workers for any task type:

```python
from libembedding import (
    autotune,
    autotune_unified,
    LEMBED_TASK_RERANKING,
    LEMBED_AUTOTUNE_QUICK,
    LEMBED_AUTOTUNE_FULL,
)

# Quick text embedding tune (5-15s)
result = autotune("BAAI/bge-small-en-v1.5", LEMBED_AUTOTUNE_QUICK)

# Unified API for any task type
result = autotune_unified(
    task=LEMBED_TASK_RERANKING,
    model_name="BAAI/bge-reranker-base",
    mode=LEMBED_AUTOTUNE_QUICK,
)
```

Results are cached by hardware fingerprint — subsequent runs load instantly.

### Error Handling

```python
from libembedding import LembedError, LlamaError
from libembedding.exceptions import (
    ModelNotFoundError,
    OnnxRuntimeError,
    DownloadError,
)

try:
    model = TextEmbedding("nonexistent/model")
except ModelNotFoundError as e:
    print(f"Model not found: {e.detail}")
except LlamaError as e:
    print(f"llama.cpp error: {e.detail}")
except LembedError as e:
    print(f"Error [{e.status_code}]: {e.message}")
```

### GGUF Models (llama.cpp)

Load quantized GGUF models:

```python
from libembedding import TextEmbedding

# Check llama.cpp availability
if TextEmbedding.supports_llamacpp():
    model = TextEmbedding.from_gguf(
        repo="Xenova/all-MiniLM-L6-v2-GGUF",
        filename="all-MiniLM-L6-v2-Q4_K_M.gguf",
        provider="llamacpp",
    )
    embeddings = model.embed(["Hello world"])
```

## Configuration

| Environment Variable | Purpose |
|---------------------|---------|
| `LIBEMBEDDING_CACHE_DIR` | Override model cache directory |
| `FASTEMBED_CACHE_DIR` | Alternative cache dir (fastembed compatibility) |
| `HF_ENDPOINT` | Custom HuggingFace Hub endpoint |

## Building & Publishing

### Prerequisites

```bash
pip install build twine
```

### Build the shared library + wheel

```bash
cd python/

# Step 1: Build the C/C++ shared library and copy it into the package
./setup.sh --build-only

# Step 2: Build sdist and wheel
python -m build
```

This produces files in `dist/` (versions follow the `libembedding-ng` package, e.g. `1.4.0`):
```
dist/
  libembedding_ng-1.1.1.tar.gz                                # source distribution
  libembedding_ng-1.1.1-cp39-abi3-manylinux_2_28_x86_64.whl  # platform wheel (bundles libembedding.so + onnxruntime)
```

### Upload to PyPI

```bash
# Upload to TestPyPI first to verify
twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ libembedding

# Upload to production PyPI
twine upload dist/*
```

### One-liner (build + upload)

```bash
./setup.sh --build-only && python -m build && twine upload dist/*
```

### Platform-specific wheels

The wheel is **platform-tagged** (e.g. `manylinux`, `macosx`, `win_amd64`) because it bundles the compiled shared library (`libembedding.so` / `.dylib` / `.dll`) along with ONNX Runtime and libcurl. `python -m build` produces a wheel for the current build platform. To build wheels for other platforms, build on that platform or use cibuildwheel:

```bash
# macOS (current arch)
./setup.sh --build-only
python -m build

# For other platforms, build on that platform or use cibuildwheel:
pip install cibuildwheel
cibuildwheel --platform linux   # builds manylinux wheels
cibuildwheel --platform macos   # builds macOS wheels
```

## License

MIT
