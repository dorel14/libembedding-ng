---
nav_exclude: true
---

# Advanced usage

## Local models (Bring Your Own ONNX)

You can use your own ONNX models without downloading them. Simply place your files in a local directory:

```
my_model/
├── model.onnx
├── tokenizer.json
└── config.json       # optional but recommended
```

Then load it by passing the path:

```python
from libembedding import TextEmbedding

model = TextEmbedding("/path/to/my_model")
embeddings = model.embed(["Hello world"])
```

### Parameters for local models

If your model does not have a `config.json`, you must specify manually:

```python
model = TextEmbedding(
    "/path/to/my_model",
    dim=768,           # embedding dimension
    pooling="mean",    # "cls" or "mean"
    max_length=512,    # max token length
)
```

## Execution providers

The `provider` parameter controls which ONNX Runtime backend is used for inference:

```python
from libembedding import TextEmbedding

# CPU (default, always available)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="cpu")

# NVIDIA CUDA
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="cuda", device_id=0)

# Apple CoreML (macOS/iOS)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="coreml")

# DirectML (Windows)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="directml")

# NVIDIA TensorRT
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="tensorrt")
```

**Supported providers:**

| Provider | Backend | Platform |
|----------|---------|----------|
| `cpu` | CPU | All |
| `cuda` | CUDA | NVIDIA GPU |
| `coreml` | CoreML | macOS, iOS |
| `directml` | DirectML | Windows |
| `tensorrt` | TensorRT | NVIDIA GPU |
| `llamacpp` | llama.cpp | All |

### GGUF models (llama.cpp)

Load quantized `.gguf` models via the llama.cpp backend:

```python
from libembedding import TextEmbedding

# Load GGUF model from HuggingFace
model = TextEmbedding(
    "Xenova/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2-Q4_K_M.gguf"
)

# Or load from local file
model = TextEmbedding("/path/to/model.Q4_K_M.gguf")

embeddings = model.embed(["Hello world"])
```

Routing happens inside the constructor: `_is_gguf_model()` recognises the `.gguf`
extension and switches to the llama.cpp backend. There is **no**
`TextEmbedding.supports_llamacpp()` in the Python API; to check whether the
backend is available at runtime, use `detect_backend(model_name)` or the C call
`lembed_llama_backend_available()`.

### Auto-tuning workers (llama.cpp)

For the llama.cpp backend, the optimal number of sessions/workers can be auto-detected:

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,      # auto-detect optimal session count
    cache_size=4096,        # optional LRU cache
)
```

## LRU Embedding Cache

Enable an LRU cache to avoid repeated inference on the same texts:

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_size=8192,  # max 8192 entries, 0 = disabled
)

# Embeddings are cached automatically
embeddings = model.embed(["Hello world", "Hello world"])  # 2nd call = cache hit
```

### Cache use cases

- **Benchmarks**: avoid recomputing the same embeddings
- **Incremental indexing**: detect duplicates
- **Development**: speed up iterations

### Default cache directory

Models are downloaded and cached in `~/.cache/libembedding` by default.

### Changing the cache directory

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_dir="/custom/cache/dir"
)
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `LIBEMBEDDING_CACHE_DIR` | Globally override the cache directory |
| `FASTEMBED_CACHE_DIR` | Alternative cache directory (fastembed compatibility) |
| `HF_ENDPOINT` | Custom HuggingFace Hub endpoint URL (e.g. mirror) |

```bash
# Usage examples
export LIBEMBEDDING_CACHE_DIR=/data/models/libembedding
export HF_ENDPOINT=https://hf-mirror.com
```

## Offline mode

Use offline mode to work without network access:

```python
from libembedding import TextEmbedding

# Only attempt to load from local cache
model = TextEmbedding("BAAI/bge-small-en-v1.5", offline=True)
```

Or set the environment variable:

```bash
export LIBEMBEDDING_NO_DOWNLOAD=1
```

## Context managers

All classes support the context manager protocol for automatic C resource management:

```python
from libembedding import TextEmbedding

with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
    embeddings = model.embed(["Hello", "World"])
# Resources are released automatically on exit
```

## Manual resource management

If you do not use a context manager, call `close()` explicitly:

```python
from libembedding import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
try:
    embeddings = model.embed(["Hello", "World"])
finally:
    model.close()
```

## Batch processing

Control batch size to balance speed and memory:

```python
from libembedding import TextEmbedding

# Batch size at construction time
model = TextEmbedding("BAAI/bge-small-en-v1.5", batch_size=64)

# Or at embed() call time
model = TextEmbedding("BAAI/bge-small-en-v1.5", batch_size=256)
embeddings = model.embed(texts, batch_size=128)  # temporary override
```

## Token limit

Some models have a maximum token length:

```python
from libembedding import TextEmbedding

# Limit to 128 tokens (default is 512 for BGE-small)
model = TextEmbedding("BAAI/bge-small-en-v1.5", max_length=128)
```

## Model lookup by code

libembedding automatically resolves models by HuggingFace name or ONNX repo code:

```python
from libembedding import TextEmbedding

# By full HuggingFace name
model = TextEmbedding("BAAI/bge-small-en-v1.5")

# By ONNX repo code
model = TextEmbedding("Xenova/bge-small-en-v1.5")

# By local path
model = TextEmbedding("/path/to/local/model")
```

## numpy integration

Returned embeddings are L2-normalized `float32` numpy arrays:

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["doc1", "doc2", "doc3"])

# Cosine similarity (dot product because embeddings are L2-normalized)
similarity = embeddings @ embeddings.T

# Nearest neighbor search
query = model.embed(["search query"])
scores = embeddings @ query.T
best_idx = scores.argmax()
```

## Full semantic search example

```python
from libembedding import TextEmbedding
import numpy as np

class SemanticSearch:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list[str]):
        self.documents.extend(documents)
        new_embeddings = self.model.embed(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
        query_emb = self.model.embed([query])
        scores = (self.embeddings @ query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i]), self.documents[i]) for i in top_indices]

    def close(self):
        self.model.close()
```

## Comparison with fastembed

libembedding is designed as a drop-in replacement for fastembed:

```python
# fastembed
from fastembed import TextEmbedding as FastEmbed
model = FastEmbed("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["Hello"]))

# libembedding — same API
from libembedding import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello"])  # returns np.ndarray directly
```

**Notable differences:**
- `embed()` returns a `np.ndarray` directly instead of a generator
- No separate `passage_embed()` method — everything goes through `embed()`
- Lighter classes, fewer Python dependencies

## Performance tips

For optimal performance:

1. **Reuse the model** — create it once and embed multiple batches
2. **Tune `batch_size`** — increase until memory saturation
3. **Use a GPU provider** — `cuda`, `coreml`, or `directml`
4. **Enable quantization** — choose `_Q` models to cut on-disk size by 2-4x (throughput effect is model-dependent: benchmark it, see [Performance](performance_tuning.html#quantization-fp32-vs-int8))
5. **Disable progress** — `show_download_progress=False` for scripts

## Dynamic batching scheduler (llama.cpp)

For the llama.cpp backend, the scheduler dynamically batches requests to improve throughput:

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,
)

# Scheduler is enabled automatically based on configuration
embeddings = model.embed(texts)
```

### Batching strategy

| Strategy | Description | Use case |
|----------|-------------|----------|
| `naive` | Sequential processing | Uniform length texts |
| `length_bucket` | Sort by length then batch | Heterogeneous corpus (short + long) |

```python
# Force bucketing for heterogeneous text
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    batch_strategy="length_bucket",
)
```

> **Note**: Bucketing is currently supported on the ONNX backend. llama.cpp support is under evaluation.
