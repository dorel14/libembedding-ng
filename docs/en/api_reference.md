---
nav_exclude: true
---

# Python API Reference

## Main classes

### TextEmbedding

Generates dense vector embeddings from text.

#### Constructor

```python
TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    provider="cpu",
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    cache_dir=None,
    max_length=0,
    dim=0,
    pooling="mean",
    auto_workers=False,
    cache_size=0,
    quantization=None,
    truncate_dim=None,
    num_threads=None,  # deprecated
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"BAAI/bge-small-en-v1.5"` | HuggingFace model name, ONNX repo code, or local directory path containing `model.onnx` + `tokenizer.json` |
| `provider` | `str` | `"cpu"` | Execution provider: `"cpu"`, `"coreml"` (Apple), `"llamacpp"`; `"cuda"`, `"directml"`, `"tensorrt"` are accepted but run on CPU (GPU planned after 1.x) |
| `threads` | `int` | `0` | Number of threads (`0` = auto) |
| `batch_size` | `int` | `256` | Internal batch size for inference |
| `offline` | `bool` | `False` | `True` = use cache only, never download |
| `show_download_progress` | `bool` | `True` | Show download progress bar |
| `cache_dir` | `str \| None` | `None` | Model cache directory (`None` = `~/.cache/libembedding`) |
| `max_length` | `int` | `0` | Max token length (`0` = model default) |
| `dim` | `int` | `0` | Embedding dimension for local models without `config.json` |
| `pooling` | `str` | `"mean"` | Pooling strategy for local models: `"cls"` or `"mean"` |
| `auto_workers` | `bool` | `False` | `True` = auto-detect optimal sessions/workers for llama.cpp backend |
| `cache_size` | `int` | `0` | LRU embedding cache size (`0` = disabled) |
| `quantization` | `str \| None` | `None` | Quantization mode: `"none"`, `"static"`, `"dynamic"`, `"fp16"` (None = model default). **Selects the registry entry providing that mode**, i.e. a different set of weights — it is not a session option. A mode the registry does not offer for this model raises `ModelNotFoundError` naming the modes that are available. ⚠️ `"auto"` is accepted by this parameter but **selects no weights**: it loads the default entry (FP32) and the `.quantization` property reports `"auto"`. For a measured auto-selection, use `preferred_quantization="auto"`. ⚠️ `"fp16"` is not INT8: those are `FLOAT16` weights served by an ORT-optimized graph. On a CPU without native FP16 arithmetic they are slower than FP32 while occupying less disk — useful for size, not for throughput. The four `Qdrant/*-onnx-Q` entries (including `bge-small-en-v1.5`) fall into that case; their `_Q` suffix and their historical "Quantized" description did not change that. |
| `preferred_quantization` | `str \| None` | `None` | `"auto"` = **measured selection**: the variants the registry actually ships for this model are benchmarked once per (model, machine, library version), the winner is cached, and later loads read the cache. Rule: no measurable FP32 baseline → FP32; a variant must beat FP32 by **more than 5 %**; and **dynamic INT8 is preferred** — another variant must beat it by **more than 15 %**. Cost: ~3.5 s on the very first load of a model, then free. See [Quantization](#quantization--fp32-vs-int8) for the measurements. |
| `truncate_dim` | `int \| None` | `None` | [Matryoshka](matryoshka.md) truncation: keeps the first `truncate_dim` components of every vector, then L2-renormalises them. Between `1` and the native dimension (otherwise `ValueError`). Meant for Matryoshka-trained models (`nomic-embed-text-v1.5`, `mxbai-embed-large-v1`, `embeddinggemma-300m`). |
| `num_threads` | `int \| None` | `None` | **Deprecated** — use `threads` instead |

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=None)` | `np.ndarray` | Embed texts. Returns array of shape `(n, dim)` with `float32` dtype. L2-normalized. |
| `embed_stream(texts, callback, batch_size=None)` | `None` | Embed as a stream — calls `callback(array, dim, userdata)` for each embedding |
| `embed_batched(texts, batch_size=None)` | `Generator` | Embed in batches, yielding an array per batch |
| `dim` | `int` (property) | Dimension of the returned vectors (`truncate_dim` when set) |
| `native_dim` | `int` (property) | Dimension produced by the model, before truncation |
| `truncate_dim` | `int \| None` (property) | Matryoshka truncation size, `None` when vectors are kept whole |
| `quantization` | `str` (property) | Quantization mode actually in use (`"none"`, `"static"`, `"dynamic"`, `"fp16"`). With `preferred_quantization="auto"`, reports the mode that was selected. |
| `quantization_reason` | `str` (property) | Why `preferred_quantization="auto"` picked that mode, e.g. `"1.69x the fp32 variant (15 -> 26 docs/s)"`. Empty when nothing was measured (explicit mode, or a model with no quantized sibling). |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name or local path |
| `model_name` | `str` (property) | Registry name of the loaded model |
| `backend` | `str` (property) | Runtime actually serving the model: `"onnxruntime"` or `"llama.cpp"` (see [Effective backend and provider](#effective-backend-and-provider)) |
| `provider` | `str` (property) | Execution provider actually in use, lowercase: `"cpu"`, `"cuda"`, `"coreml"`, `"directml"`, `"tensorrt"`; `"gpu"` for llama.cpp with offloaded layers |
| `info()` | `ModelDesc` | Loaded model descriptor |
| `stats()` | `Stats` | Runtime usage statistics |
| `close()` | `None` | Release resources |
| `__enter__()` | `self` | Context manager support |
| `__exit__()` | `None` | Calls `close()` automatically |
| `from_mode(mode="balanced", **kwargs)` | `TextEmbedding` | (classmethod) Create a TextEmbedding from a preset mode: `"fast"`, `"balanced"`, `"quality"` |

> **No `list_supported_models()` on `TextEmbedding`** — it only exists on
> `SparseTextEmbedding`, `ImageEmbedding` and `Reranker`. For text models use the
> module-level `list_text_models()`.

> **No `from_gguf()` and no `supports_llamacpp()`.** GGUF loading goes through
> the constructor: `_is_gguf_model()` recognises the `.gguf` extension and
> switches to the llama.cpp backend. For a HuggingFace repo pass
> `"repo/filename.gguf"`; for a local file pass the path.
> `detect_backend(model_name)` reports the backend that was selected.

#### Example

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])
print(embeddings.shape)   # (2, 384)
print(embeddings.dtype)   # float32

# Simple semantic search
query = model.embed(["What is AI?"])
scores = embeddings @ query.T
best_idx = scores.argmax()
```

---

### SparseTextEmbedding

Generates **sparse** embeddings (sparse vectors with token indices and weights).
`batch_size=None` uses the backend default: 256 for ONNX and 16 for GGUF. The
GGUF default is a measured compromise between throughput, stability and number
of batches; batch 2 can further reduce padding and memory at similar throughput.
Callers can override either setting.

#### Constructor

```python
SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    max_length=0,
    threads=0,
    batch_size=None,
    offline=False,
    show_download_progress=True,
    top_terms=0,
    min_weight=0.0,
    quantization=None,
    backend="auto",
    cache_size=0,
    num_threads=None,  # deprecated
)
```

**Available models:**

| HuggingFace name | Description |
|------------------|-------------|
| `prithivida/Splade_PP_en_v1` | SPLADE++ (default) |
| `BAAI/bge-m3` | Multilingual BGE-M3 |

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=0)` | `list[SparseEmbedding]` | Returns a list of `SparseEmbedding` objects |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name |
| `backend` | `str` (property) | `"onnxruntime"` or `"ggml-sparse"` (the library's own ggml runtime for GGUF SPLADE models, not llama.cpp) |
| `provider` | `str` (property) | Execution provider actually in use (`"cpu"` for `"ggml-sparse"`) |
| `info()` | `ModelDesc` | Model descriptor |
| `stats()` | `Stats` | Runtime usage statistics |
| `close()` | `None` | Release resources |
| `list_supported_models()` | `list[ModelInfo]` | (static) List sparse models |

#### Example

```python
from libembedding import SparseTextEmbedding

sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
    # indices: int32 array of token IDs
    # values: float32 array of weights
```

---

### ImageEmbedding

Generates dense vector embeddings from images.

#### Constructor

```python
ImageEmbedding(
    model_name="Qdrant/clip-ViT-B-32-vision",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    threads=0,
    batch_size=30,
    offline=False,
    show_download_progress=True,
    dim=0,
    num_threads=None,  # deprecated
)
```

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `embed_files(paths, batch_size=None)` | `np.ndarray` | Embed from file paths. Shape: `(n, dim)` |
| `embed_bytes(images, batch_size=None)` | `np.ndarray` | Embed from raw bytes (JPEG, PNG, etc.) |
| `dim` | `int` (property) | Embedding dimension |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name |
| `backend` | `str` (property) | Always `"onnxruntime"` |
| `provider` | `str` (property) | Execution provider actually in use |
| `info()` | `ModelDesc` | Model descriptor |
| `stats()` | `Stats` | Runtime usage statistics |
| `close()` | `None` | Release resources |
| `list_supported_models()` | `list[ModelInfo]` | (static) List image models |

#### Example

```python
from libembedding import ImageEmbedding

model = ImageEmbedding()
embeddings = model.embed_files(["photo.jpg", "diagram.png"])
print(embeddings.shape)  # (2, 512)
```

---

### MultiVectorEmbedding

Multi-vector (late interaction, ColBERT) embeddings: one float32 matrix
`[n_tokens, dim]` per text. ONNX Runtime only. The model is either a **local
directory** (`model.onnx` or `vespa_colbert.onnx`, `tokenizer.json`, and an optional
`onnx_config.json`) or a **registry entry**
(`"answerdotai/answerai-colbert-small-v1"`, downloaded into the cache on first use;
an existing directory wins). The library embeds; it does not index or search: score with
`maxsim` / `maxsim_batch`. Alias: `LateInteractionTextEmbedding` (FastEmbed
naming). C API: `multivector_embedding.h` (`lembed_multivector_*`).

#### Constructor

```python
MultiVectorEmbedding(
    model_dir, *, provider="cpu", threads=0, batch_size=0,
    query_maxlen=None, doc_maxlen=None, normalize=True,
    query_expansion=None, attend_to_expansion=None,
    cache_dir=None, offline=False, show_download_progress=False,
)
```

`cache_dir` / `offline` / `show_download_progress`: registry models only.
`MultiVectorEmbedding.list_supported_models()` lists the registry.
`query_maxlen` / `doc_maxlen`: length overrides (model defaults: 32 / 300).
`query_expansion`: pad queries with `[MASK]` (`None` = on).
`attend_to_expansion`: `None` = follow the model.

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `embed(texts, *, mode="document")` | `list[np.ndarray]` | One `[n_i, dim]` matrix per text; `mode` is `"document"` or `"query"` |
| `query_embed(query)` | `Iterator[np.ndarray]` | Query or queries (`str` or list), `query` mode |
| `passage_embed(texts)` | `Iterator[np.ndarray]` | Documents, `document` mode |
| `dim` | `int` (property) | Vector dimension |
| `backend` | `str` (property) | `"onnxruntime"` |
| `info()` / `stats()` | `ModelDesc` / `Stats` | Descriptor and usage statistics |
| `close()` | `None` | Release resources (idempotent); works with `with` |

Errors are `LembedError` subclasses (missing directory or file: `IOError`;
`.gguf` path: `UnsupportedError`; registry download failure: `DownloadError`).

#### Example

```python
from libembedding import MultiVectorEmbedding, maxsim

with MultiVectorEmbedding("/path/to/answerai-colbert-small-v1") as mv:
    (q,) = mv.query_embed("what is the capital of France?")   # (32, 96)
    docs = list(mv.passage_embed(["Paris is the capital of France.",
                                  "Bananas grow in tropical climates."]))
    scores = [maxsim(q, d) for d in docs]                      # first is highest
```

---

### Reranker

Scores and sorts documents by relevance to a query (cross-encoder).

On ONNX, query and document are encoded as a real sequence pair following the `tokenizer.json` `post_processor` (pair template, special tokens, `token_type_ids`) with `longest_first` truncation; BERT style `[CLS] q [SEP] d [SEP]` when none is declared. On GGUF (llama.cpp) the pair is built from the vocabulary's special tokens (BERT: `[CLS] q [SEP] d [SEP]`; XLM-R: `<s> q </s></s> d </s>`), same truncation.

#### Constructor

```python
Reranker(
    model_name="BAAI/bge-reranker-base",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    max_length=0,
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    backend="auto",
    quantization=None,
    cache_size=0,
    num_threads=None,  # deprecated
)
```

**Available models:**

| HuggingFace name | Description |
|------------------|-------------|
| `BAAI/bge-reranker-base` | BGE Reranker base (default) |
| `BAAI/bge-reranker-v2-m3` | Multilingual BGE Reranker v2 |
| `jinaai/jina-reranker-v1-turbo-en` | Jina Reranker v1 turbo |
| `jinaai/jina-reranker-v2-base-multilingual` | Jina Reranker v2 base multilingual |
| `jinaai/jina-reranker-v1-turbo-en-quantized` | Jina Reranker v1 turbo English (INT8) |

> **`quantization=` does not change the reranker weights.** Unlike
> `TextEmbedding`, it does not resolve a registry entry: it only sets ONNX
> Runtime session options. To get INT8, pass the **name** of the quantized entry
> — `jinaai/jina-reranker-v1-turbo-en-quantized`, which points at
> `onnx/model_quantized.onnx` of the `jinaai/jina-reranker-v1-turbo-en` repo.
> Unlike text models, a quantized reranker entry has a `model_name` **distinct**
> from its FP32 version, which is precisely why no per-mode resolution exists
> there today.

#### Methods and properties

| Member | Type | Description |
|--------|------|-------------|
| `rerank(query, documents, batch_size=0)` | `list[RerankResult]` | Returns results sorted by descending score |
| `batch_size` | `int` (property) | Configured batch size |
| `name` | `str` (property) | Model name |
| `backend` | `str` (property) | `"onnxruntime"` or `"llama.cpp"`: with `backend="auto"` a model without ONNX files may be served by its GGUF counterpart, and this is where that switch shows |
| `provider` | `str` (property) | Execution provider actually in use (`"cpu"` for llama.cpp) |
| `info()` | `ModelDesc` | Model descriptor |
| `stats()` | `Stats` | Runtime usage statistics |
| `close()` | `None` | Release resources |
| `list_supported_models()` | `list[ModelInfo]` | (static) List reranker models |
| `auto(profile="balanced", **kwargs)` | `Reranker` | (static) Create a Reranker auto-configured by profile |

#### Example

```python
from libembedding import Reranker

reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank(
    "What is deep learning?",
    [
        "Deep learning uses neural networks",
        "The weather is sunny today",
        "Neural networks are inspired by biological brains",
    ],
)

for r in ranked:
    print(f"doc[{r.index}] score={r.score:.4f}")
```

---

## Data types

### TextEmbeddingPool

Pool of ONNX sessions for inter-session parallelism. See [performance_tuning.html](performance_tuning.html).

```python
class TextEmbeddingPool:
    model_name: str
    workers: int = 0                  # number of sessions (0 = auto-detect)
    threads_per_worker: int = 1       # threads per session
    batch_size: int = 256
    provider: str = "cpu"
    offline: bool = False
    show_download_progress: bool = True
    cache_dir: str | None = None
    max_length: int = 0
    dim: int = 0
    pooling: str = "mean"
    autotune: bool = False            # auto-tune all parameters
    autotune_texts: list[str] = None  # corpus for autotune
    autotune_max_samples: int = 100   # max sample size
    cache_size: int = 0               # LRU cache per worker (0 = disabled)
    quantization: str | None = None   # quantization mode
```

**Methods:**

| Member | Type | Description |
|--------|------|-------------|
| `embed(texts)` | `np.ndarray` | Embed texts in parallel |
| `num_workers` | `int` (property) | Number of active workers |
| `dim` | `int` (property) | Embedding dimension |
| `close()` | `None` | Release resources |

### TuningResult

Result of autotune for a model.

```python
@dataclass(frozen=True)
class TuningResult:
    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
```

### ModelSelectionResult

Result of automatic model selection.

```python
@dataclass(frozen=True)
class ModelSelectionResult:
    model_code: str
    model_name: str
    dim: int
    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
    score: float
```

### Stats

Runtime usage statistics for an embedding context.

```python
@dataclass(frozen=True)
class Stats:
    texts_embedded: int    # total texts processed
    batches_run: int       # total ONNX batches executed
    avg_latency_ms: float  # average latency per embed() call (ms)
    cache_hits: int = 0    # LRU cache hits (0 if disabled)
    cache_misses: int = 0  # LRU cache misses (0 if disabled)
```

Available via the `.stats()` method on `TextEmbedding`, `Reranker`, `ImageEmbedding`, and `SparseTextEmbedding`.

### EmbeddingCache

Thread-safe LRU cache for dense embeddings. Can be used standalone or via the `cache_size` parameter of constructors.

```python
from libembedding import EmbeddingCache

cache = EmbeddingCache(capacity=4096, ttl_seconds=0, dim=0)
cache.put("text", np.array([...], dtype=np.float32))
vec = cache.get("text")  # np.ndarray | None
cache.clear()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `put(text, vec)` | `None` | Store an embedding |
| `get(text, dim=None)` | `np.ndarray \| None` | Look up a cached embedding by text key |
| `clear()` | `None` | Remove all entries |
| `capacity` | `int` | Maximum cache capacity |
| `current_size` | `int` | Current number of entries |
| `close()` | `None` | Release underlying C resources |

### Global functions

| Function | Description |
|----------|-------------|
| `autotune(model_name, full=False, texts=None, max_sample_size=100)` | Auto-tune a model. Returns `TuningResult`. `texts` supplies a measurement corpus (sampled down to `max_sample_size`); `None` uses a synthetic corpus. |
| `autotune_unified(task="embedding", model_name=None, *, full=False, objective=None, texts=None, max_latency_ms=0.0, min_tokens=0, max_sample_size=100)` | **Recommended.** Unified auto-tune for the four tasks (`"embedding"`, `"reranking"`, `"image"`, `"sparse"`). Returns `UnifiedTuningResult`. See [Unified Tuning](#unified-tuning). |
| `clear_autotune_cache_unified(task, model_name=None)` | Clear one task's autotune cache for a model (`None` = every model of the task). |
| `sparse_autotune(model_name, *, full=False)` | Legacy: auto-tune sparse embedding (threads × batch). Returns `SparseTuningResult`. |
| `image_autotune(model_name, *, full=False)` | Legacy: auto-tune image embedding. Returns `ImageTuningResult`. |
| `reranker_autotune(model_name, *, full=False, objective="balanced")` | Legacy: auto-tune reranker. Returns `RerankerTuningResult`. |
| `reranker_auto_config(model_name, target_latency_ms=500, objective="balanced")` | Auto-config reranker for a latency budget. |
| `reranker_auto_config_profile(profile="balanced", model_name=...)` | Auto-config reranker using a profile. |
| `reranker_autotune_constrained(model_name, *, full=False, objective="balanced", min_tokens=64, max_latency_ms=500.0)` | Legacy: auto-tune with constraints. |
| `clear_reranker_autotune_cache(model_name=None)` | Clear reranker autotune cache. |
| `auto_select_model(use_case="balanced")` | Select best model. Returns `ModelSelectionResult`. |
| `clear_autotune_cache(model_name=None)` | Clear the text autotune cache (with `None`, the **whole** autotune directory). |
| `cache_path()` | Get autotune cache file path. |
| `cosine_similarity(a, b)` | Cosine similarity between two vectors |
| `dot_product(a, b)` | Dot product between two vectors |
| `euclidean_distance(a, b)` | L2 distance between two vectors |
| `maxsim(query, doc)` | MaxSim (ColBERT) score: sum over `query` rows of the max dot product with `doc` rows (2-D float32 matrices) |
| `maxsim_batch(query, docs)` | One MaxSim score per document (`docs`: sequence of float32 matrices), in input order |
| `detect_backend(model_name, backend="auto")` | Detect which backend to use |

---

## Similarity Functions

Compare two embedding vectors natively without instantiation overhead.

```python
from libembedding import cosine_similarity, dot_product, euclidean_distance
import numpy as np

a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
b = np.array([0.4, 0.5, 0.6], dtype=np.float32)

sim = cosine_similarity(a, b)
dp = dot_product(a, b)
dist = euclidean_distance(a, b)
```

---

## Unified Benchmark

Unified comparison of ONNX and llama.cpp backends with auto-tuning.

| Class/Constant | Description |
|------------------|-------------|
| `Benchmark` | Benchmark with auto-tuning and multi-backend comparison |
| `BenchmarkResult` | Result for a model x backend |
| `ComparisonResult` | Comparison results with recommendation |
| `Metrics` | Benchmark metrics (throughput, latencies, memory) |
| `HardwareInfo` | Detected hardware information |
| `CorpusType` | Corpus categories (SHORT, MEDIUM, LONG, MIXED, MULTILINGUAL, EDGE_CASES) |
| `Objective` | Optimization objectives (LATENCY, THROUGHPUT, BALANCED, MEMORY) |

```python
from libembedding import Benchmark, Objective, detect_backend, cache_path, clear_cache

# Backend detection
backend = detect_backend("BAAI/bge-small-en-v1.5")

# Benchmark
bench = Benchmark()
result = bench.autotune("path/to/model.gguf", "llama.cpp", objective=Objective.BALANCED)
comparison = bench.compare_all(onnx_path="model.onnx", gguf_path="model.gguf")

# Utilities
path = cache_path()
clear_cache()
```

---

### Unified Tuning

The unified auto-tuner provides a single API for optimizing all task types (text embedding, sparse, image, reranking). It runs on `lembed_autotune_ex()` on the C side.

```python
from libembedding import autotune_unified, UnifiedTuningResult

result = autotune_unified(
    "reranking",
    "jinaai/jina-reranker-v1-turbo-en-quantized",
    objective="latency",                      # default: "throughput" (embedding, sparse, image), "balanced" (reranking)
    texts=[*documents, query],                # reranking: the LAST text is the query
    max_latency_ms=300.0,                     # p95 budget per tuning call (0 = none)
    min_tokens=128,                           # reranking only
)
print(result.threads, result.batch_size, result.max_tokens)
```

When no configuration fits `max_latency_ms`, the best unconstrained configuration is returned: compare `result.p95_latency_ms` with the budget. `texts` is rejected for `"image"`. For `"sparse"`, `top_k`, `min_weight` and `storage_format` are not tuned: they reflect the model defaults. Details, grids, methodology and measurements: [Performance](performance_tuning.html#unified-auto-tuning-api).

#### UnifiedTuningResult

```python
@dataclass(frozen=True)
class UnifiedTuningResult:
    task: str               # "embedding", "reranking", "image", or "sparse"
    threads: int
    batch_size: int
    workers: int            # embedding only
    max_tokens: int         # reranker only
    throughput_docs_sec: float
    latency_ms: float       # per call; amortised per text for "embedding"
    p95_latency_ms: float
    memory_mb: float        # estimate (RSS delta), not an absolute peak
    top_k: int = 0          # sparse: model default (not tuned)
    min_weight: float = 0.0 # sparse: same
    storage_format: int = 0 # sparse: same
```

### SparseTuningResult

```python
@dataclass(frozen=True)
class SparseTuningResult:
    top_k: int
    min_weight: float
    storage_format: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
```

### ImageTuningResult

```python
@dataclass(frozen=True)
class ImageTuningResult:
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
```

### RerankerTuningResult

```python
@dataclass(frozen=True)
class RerankerTuningResult:
    threads: int
    batch_size: int
    max_tokens: int
    throughput_docs_sec: float
    latency_ms: float
    p95_latency_ms: float
    memory_mb: float
```

---

### SparseEmbedding

```python
@dataclass(frozen=True)
class SparseEmbedding:
    indices: np.ndarray  # int32 — token IDs
    values:  np.ndarray  # float32 — token weights
```

### RerankResult

```python
@dataclass(frozen=True)
class RerankResult:
    index: int   # index of the document in the input list
    score: float # relevance score (higher = more relevant)
```

### ModelInfo

```python
@dataclass(frozen=True)
class ModelInfo:
    model_name:    str   # e.g. "BAAI/bge-small-en-v1.5"
    model_code:    str   # HF repo code, e.g. "Xenova/bge-small-en-v1.5"
    model_file:    str   # e.g. "onnx/model.onnx"
    description:   str
    dim:           int   # embedding dimension
    max_tokens:    int   # max token length
    pooling:       str   # "cls" or "mean"
    quantization:  str   # "none", "static", "dynamic"
    download_size_mb: int = -1  # download size in MB, -1 if unknown
```

### ModelDesc

```python
@dataclass(frozen=True)
class ModelDesc:
    name: str
    dimension: int
    max_length: int
    pooling: str   # "cls" or "mean"
    num_threads: int
    batch_size: int
    provider: str = "cpu"          # requested provider ("llama" for every GGUF context)
    device_id: int = 0
    quantization: str = "none"
    cache_size: int = 0
    backend: str = ""              # "onnxruntime", "llama.cpp" or "ggml-sparse"
    effective_provider: str = ""   # provider actually in use: "cpu", "cuda", ...
```

`backend` and `effective_provider` are appended last with a default value, so
code building a `ModelDesc` positionally keeps working. `info()` always fills
them.

### Effective backend and provider

Every model reports which runtime serves it and which provider it really runs
on (`backend` / `provider` properties, `info().backend` /
`info().effective_provider` fields):

| `backend` | Meaning | `provider` |
|---|---|---|
| `"onnxruntime"` | ONNX model | `"cpu"`, or the requested provider if it could be attached |
| `"llama.cpp"` | Dense or reranker GGUF model | `"cpu"`; `"gpu"` when `llama_n_gpu_layers != 0` and ggml has a GPU device (none in the shipped builds) |
| `"ggml-sparse"` | SPLADE GGUF model (in-house ggml runtime) | `"cpu"` |

**Explicitly requested but unavailable provider: error.** `"cpu"` is the
default and there is no "auto" provider, so any other value is an explicit
request. When it cannot be attached (build without CUDA, DirectML or
TensorRT, CoreML outside Apple, `"llama"` on an ONNX model, unknown value),
construction raises `UnsupportedError` with the reason, instead of running on
the CPU without saying so. The check happens before the model is parsed.

```python
from libembedding import TextEmbedding
from libembedding.exceptions import UnsupportedError

try:
    model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="cuda")
except UnsupportedError as e:
    print(e.detail)  # "Execution provider CUDA was requested but is not available: ..."

model = TextEmbedding("BAAI/bge-small-en-v1.5")
print(model.backend, model.provider)  # onnxruntime cpu
```

**`Reranker(backend="auto")` switch to GGUF: kept, but visible.** When the
requested model has no ONNX files and a GGUF counterpart exists, the reranker
is served by llama.cpp: `reranker.backend` is then `"llama.cpp"`, and
`LIBEMBEDDING_VERBOSE=1` writes the switch to stderr.

---

## Error handling

All exceptions inherit from `LembedError`. The `LlamaError` subclass is raised for llama.cpp/GGUF backend failures:

```python
from libembedding import LembedError
from libembedding.exceptions import (
    InvalidArgumentError,
    OutOfMemoryError,
    OnnxRuntimeError,
    TokenizerError,
    DownloadError,
    IOError,
    ModelNotFoundError,
    UnsupportedError,
    BatchSizeError,
    LlamaError,
)
```

**Hierarchy:**

```
LembedError (Exception)
├── InvalidArgumentError   — NULL pointer or out-of-range enum
├── OutOfMemoryError       — allocation failure
├── OnnxRuntimeError       — ONNX Runtime error
├── TokenizerError         — tokenizer loading/encoding error
├── DownloadError          — download failure
├── IOError                — file I/O error
├── ModelNotFoundError     — unknown model
├── UnsupportedError       — feature disabled at compile time, or registry model above the mobile limit (Android/iOS)
├── BatchSizeError         — incompatible batch size
└── LlamaError             — llama.cpp/GGUF backend error
```

`LlamaError` provides:
- `status_code` (`int`) — `LEMBED_ERROR_LLAMA`
- `message` (`str`) — `"llama.cpp backend error"`
- `detail` (`str`) — technical detail from llama.cpp

#### Example

```python
from libembedding import TextEmbedding
from libembedding.exceptions import ModelNotFoundError, OnnxRuntimeError

try:
    model = TextEmbedding("nonexistent/model")
except ModelNotFoundError as e:
    print(f"Model not found: {e.detail}")
except OnnxRuntimeError as e:
    print(f"ONNX error: {e.detail}")
except LembedError as e:
    print(f"libembedding error [{e.status_code}]: {e.message}")
```

Each exception provides:
- `status_code` (`int`) — C error code
- `message` (`str`) — generic message (e.g. `"ONNX Runtime error"`)
- `detail` (`str`) — technical detail (e.g. ORT error message)

---

## Utility functions

```python
import libembedding

# List models by category
libembedding.list_text_models()      # list[ModelInfo]
libembedding.list_sparse_models()    # list[ModelInfo]
libembedding.list_image_models()     # list[ModelInfo]
libembedding.list_reranker_models()  # list[ModelInfo]

# Size limit for registry models (MB): 150 on Android and iOS, 0 (no limit)
# elsewhere. On mobile, list_*_models() only returns models within the limit.
# See Platforms.
libembedding.mobile_model_limit_mb()  # int
```

See [Platforms](platforms.html) for the mobile limit (`UnsupportedError`,
`LIBEMBEDDING_ALLOW_LARGE_MODELS`, `LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT`).

### Auto-tuning workers (C API)

```c
#include <libembedding/worker_autotune.h>

lembed_worker_config_t cfg = lembed_detect_optimal_workers();
printf("Physical cores: %d\n", cfg.physical_cores);
printf("Optimal workers: %d\n", cfg.optimal_workers);
printf("Optimal threads: %d\n", cfg.optimal_threads);

int workers = lembed_recommended_workers_for_model("/path/to/model.gguf");
```

### Embedding modes (C API)

```c
#include <libembedding/embedding_mode.h>

lembed_embedding_mode_t mode = LEMBED_MODE_BALANCED;
lembed_text_model_t model = lembed_recommended_model_for_mode(mode);
const char* mode_str = lembed_mode_to_string(mode);  // "balanced"
```

### Embedding cache (C API)

```c
#include <libembedding/embedding_cache.h>

lembed_cache_config_t cfg = lembed_cache_config_default();
cfg.capacity = 4096;
cfg.ttl_seconds = 3600;  // 1 hour

lembed_cache_t* cache = lembed_cache_create(&cfg);
float* vec = NULL;
int dim = 0;
if (lembed_cache_get(cache, "Hello world", &vec, &dim)) {
    // cache hit
} else {
    // cache miss : calculate the embedding then store it
    float embedding[384];
    // ... calculation ...
    lembed_cache_put(cache, "Hello world", embedding, 384);
}

lembed_cache_free(cache);
```

### Effective backend and provider (C API)

Two accessors per context type. They return static strings (never freed), or
`NULL` when the context is `NULL`. No field is added to `lembed_model_desc_t`:
its size (ABI) does not change.

| Function | Values |
|---|---|
| `lembed_text_embedding_backend_name(ctx)` | `"onnxruntime"`, `"llama.cpp"` |
| `lembed_text_embedding_effective_provider(ctx)` | `"CPU"`, `"CUDA"`, `"CoreML"`, `"DirectML"`, `"TensorRT"`; llama.cpp: `"CPU"`, or `"GPU"` when layers are offloaded to a ggml GPU device |
| `lembed_sparse_text_embedding_backend_name(ctx)` | `"onnxruntime"`, `"ggml-sparse"` |
| `lembed_sparse_text_embedding_effective_provider(ctx)` | as for ONNX; `"CPU"` for `"ggml-sparse"` |
| `lembed_reranker_backend_name(ctx)` | `"onnxruntime"`, `"llama.cpp"` |
| `lembed_reranker_effective_provider(ctx)` | as for ONNX; `"CPU"` for llama.cpp |
| `lembed_image_embedding_backend_name(ctx)` | `"onnxruntime"` |
| `lembed_image_embedding_effective_provider(ctx)` | as for ONNX |

Provider names use the spelling of `lembed_execution_provider_name()`.

```c
lembed_text_options_t opts = lembed_text_options_default();
opts.provider = LEMBED_PROVIDER_CUDA;
lembed_text_embedding_t* ctx = NULL;
lembed_status_t s = lembed_text_embedding_create(&opts, &ctx);
if (s == LEMBED_ERROR_UNSUPPORTED) {
    /* "Execution provider CUDA was requested but is not available: ..." */
    fprintf(stderr, "%s\n", lembed_last_error());
    opts.provider = LEMBED_PROVIDER_CPU;
    s = lembed_text_embedding_create(&opts, &ctx);
}
printf("%s on %s\n", lembed_text_embedding_backend_name(ctx),
       lembed_text_embedding_effective_provider(ctx));  /* onnxruntime on CPU */
```

**Behaviour change (decision D3).** A provider other than
`LEMBED_PROVIDER_CPU` that cannot be attached to the ONNX Runtime session makes
the create function fail with `LEMBED_ERROR_UNSUPPORTED` and a message in
`lembed_last_error()`, instead of silently falling back to the CPU.
`LEMBED_PROVIDER_CPU` is the default of every options struct and there is no
"auto" provider, so any other value is an explicit request. This also applies
to `LEMBED_PROVIDER_LLAMACPP` given to an ONNX model and to an out-of-range
value. The GGUF paths (llama.cpp, `ggml-sparse`) ignore the provider as before.

The reranker `LEMBED_BACKEND_AUTO` switch (ONNX model not found, matching GGUF
entry) is kept: `lembed_reranker_backend_name()` then returns `"llama.cpp"`
and `LIBEMBEDDING_VERBOSE=1` logs the switch.

### Similarity (C API)

```c
#include <libembedding/similarity.h>

float a[] = {0.1f, 0.2f, 0.3f};
float b[] = {0.4f, 0.5f, 0.6f};
int dim = 3;

float sim = lembed_cosine_similarity(a, b, dim);
float dp  = lembed_dot_product(a, b, dim);
float dist = lembed_euclidean_distance(a, b, dim);
```
