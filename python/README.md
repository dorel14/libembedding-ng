# libembedding

> **Forked from [pacifio/libembedding](https://github.com/pacifio/libembedding).**
> This fork adds Windows support (native DLL), PyPI packaging under the name
> **`libembedding-ng`**, local model loading, runtime introspection, similarity
> helpers, streaming, a multi-worker pool, an autotuner and automatic model
> selection. Imported as `libembedding`.

Fast ONNX-based text, image, and sparse embeddings for Python. Dynamic INT8 ONNX models are ~4x smaller, ~2x lighter in memory and ~1.7-2.0x faster than FP32 on CPU (measured; see Benchmarks below).

Built on a C/C++ backend using ONNX Runtime, exposed to Python via zero-overhead cffi bindings. The wheel bundles the **compiled shared library** (`libembedding.so` / `.dylib` / `.dll`) together with ONNX Runtime (and, except on macOS, libcurl), so no system ONNX Runtime install is required at runtime. The licence texts of the bundled native code are in the wheel's `.dist-info/licenses/`. Supports 44 text embedding models, 5 image models, 2 sparse models, and 4 rerankers with automatic model downloading from HuggingFace Hub.

## Installation

```bash
pip install libembedding-ng
```

> The PyPI package is named **`libembedding-ng`** (to avoid clashing with the original
> `libembedding` on PyPI); it is imported as `libembedding`. The wheel already bundles the
> native shared library and ONNX Runtime, so no separate system install is required.

**Platforms:** wheels exist for Windows x86-64, Linux x86-64 (manylinux), Linux aarch64
(`manylinux_2_31`: Raspberry Pi 3/4/5 under 64-bit Raspberry Pi OS, Ubuntu 20.04+, Graviton)
and macOS arm64 (Apple Silicon, macOS 14 or later). The x86-64 wheels run on any x86-64 CPU,
no AVX/AVX2 required: GGUF models load the ggml CPU library matching the machine at run time
(`libembedding.ggml_cpu_features()` names it), and the aarch64 wheel does the same across
eight ARM levels (armv8.0 to armv9.2). 32-bit ARM (armv7l) is not supported yet. On Linux,
downloads go through a libcurl linked statically over mbedTLS, and HTTPS uses the system CA bundle
(`SSL_CERT_FILE` or `CURL_CA_BUNDLE` override it). The macOS wheel bundles its dependencies
(ONNX Runtime, llama.cpp/ggml, built without Metal) under `libembedding/.dylibs/` and uses the system libcurl.

**Android:** wheels for CPython 3.13 on Android 7.0+ (`android_24_arm64_v8a`,
`android_24_x86_64`, PEP 738: BeeWare, Chaquopy). cffi and numpy come from Chaquopy's index:
`pip install --extra-index-url https://chaquo.com/pypi-13.1 libembedding-ng`. Models download
normally (static libcurl over mbedTLS, system certificate store). The x86_64 wheel is tested
on an emulator in CI; the arm64 wheel has not been run on a device yet.

**Mobile model limit:** on Android and iOS, only light registry models load by name. A model
above 150 MB raises `UnsupportedError` before any download, with the light alternatives in
the message. `list_text_models()` and the other listing functions return only the light
models, and `ModelInfo.download_size_mb` gives each model's size. No sparse model is light
(SPLADE++ is 532 MB). `LIBEMBEDDING_ALLOW_LARGE_MODELS=1` lifts the limit, and a local model
path is never checked. `mobile_model_limit_mb()` returns the active limit (0 = none), and
`LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT=1` applies the limit on a desktop for testing.

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
    provider="cpu",  # "cpu", "coreml" (Apple), "llamacpp"; "cuda"/"directml"/"tensorrt" run on CPU (GPU planned after 1.x)
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
| `provider` | `str` | Execution provider requested ("cpu", "coreml", "llamacpp"; "cuda", "directml", "tensorrt" are accepted but run on CPU) |
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

### Comparison with fastembed — withdrawn

An earlier version of this file published a table claiming libembedding is
6.5-8.6x faster than `fastembed`, stated as "measured on Apple M-series". **It has
been removed.** All four measurement harnesses hardcoded
`"platform": "macOS arm64"` in their output, so no run could be traced to a
machine, and the maintainer has no Mac to reproduce it on. The harnesses now
report the real platform. See
`docs/archive/benchmarks/cross-implementation-macos-claim/` for the withdrawn
figures and the reasoning.

Treat any cross-implementation speed claim about libembedding as unsubstantiated
until it is re-measured on an identified machine.

### Quantization — measured and reproducible

The quantization benchmark in the main README reports its full environment (CPU,
ONNX Runtime version, execution provider) and reads the weight dtypes from the
ONNX files, so its provenance is verifiable.

### Unified Auto-Tuning

Auto-tune threads and batch_size (plus workers for text, `max_tokens` for the
reranker) for any task type:

```python
from libembedding import autotune, autotune_unified, clear_autotune_cache_unified

# Text embedding tune (QUICK by default; ~13 s cold on an i7-1065G7, indicative)
result = autotune("BAAI/bge-small-en-v1.5")

# Unified API for any task: "embedding", "reranking", "sparse", "image"
result = autotune_unified("embedding", "BAAI/bge-small-en-v1.5")

# Objective, custom workload and constraints (reranking: last text = the query)
result = autotune_unified(
    "reranking",
    "jinaai/jina-reranker-v1-turbo-en-quantized",
    objective="latency",              # "latency" | "throughput" | "balanced" | "memory"
    texts=[*documents, query],
    max_latency_ms=300.0,             # p95 budget for one tuning call (0 = none)
    min_tokens=128,                   # reranking only: floor for max_tokens
)
# Over budget? The best unconstrained config is returned: check result.p95_latency_ms.

clear_autotune_cache_unified("reranking")   # purge one task's cache
```

Results are cached per model, hardware, runtime, workload, objective, mode and
constraints — a repeated call returns in milliseconds. For sparse models, `top_k`,
`min_weight` and `storage_format` are not tuned (they change the output); the
result reports the model defaults. See `docs/performance_tuning.md` for grids,
methodology and measurements.

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
| `LIBEMBEDDING_DOWNLOAD_STALL_TIMEOUT` | Seconds without any byte received before a download is aborted and retried (default 30; up to 3 attempts) |
| `ORT_DISABLE_TELEMETRY` | ONNX Runtime telemetry: set to `1` by libembedding when unset (no data sent to Microsoft); `0` keeps ORT's telemetry |

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
