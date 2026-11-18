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

# Apple CoreML (macOS/iOS), falls back to CPU if unavailable
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="coreml")
```

**Supported providers:**

| Provider | Backend | Platform |
|----------|---------|----------|
| `cpu` | CPU | All |
| `coreml` | CoreML | macOS, iOS |
| `llamacpp` | llama.cpp (CPU) | All |
| `cuda`, `directml`, `tensorrt` | reserved: run on CPU | — |

GPU acceleration is out of scope before 1.x. The `cuda`, `directml` and
`tensorrt` values are accepted to keep the API stable, but no build enables
them: the model then runs on the CPU.

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
| `LIBEMBEDDING_DOWNLOAD_STALL_TIMEOUT` | Seconds without any byte received before a download is aborted and retried (default 30; up to 3 attempts) |
| `ORT_DISABLE_TELEMETRY` | ONNX Runtime telemetry: libembedding sets it to `1` when unset (nothing sent to Microsoft); `0` keeps ORT's telemetry |
| `LIBEMBEDDING_VERBOSE` | `1` prints diagnostics on stderr (autotune progress, auto model selection, cache cleanup dry runs); silent by default |
| `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES` | `1` loads the registry models whose licence is not MIT-compatible (`jina-reranker-v2-base-multilingual`, CC-BY-NC-4.0; `embeddinggemma-300m`, Gemma), refused by default |

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

## Tokenizer fidelity (HuggingFace `tokenizers`)

The built-in tokenizer is compared with the HuggingFace `tokenizers` library on a
multilingual corpus (Greek, Cyrillic, Vietnamese, CJK, Arabic, Hebrew, Hindi, Thai,
Korean, emoji, accented Latin, mixed case, punctuation, very long words). The
references are JSON fixtures committed in `tests/data/tokenizer/`, produced by
`scripts/gen_tokenizer_fixtures.py`; the `test_tokenizer_fidelity` test needs neither
Python nor the `tokenizers` package.

| Family | Models | Status |
|---|---|---|
| WordPiece + BertNormalizer (uncased) | MiniLM, bge-*-en, snowflake, gte-en, nomic, mxbai, SPLADE | identical to HF, 0/54 texts differ |
| WordPiece, Chinese vocabulary | bge-*-zh | identical to HF, 0/54 |
| WordPiece, `<s>`/`</s>` | all-mpnet-base-v2 | identical to HF, 0/54 |
| Unigram (SentencePiece, `nmt_nfkc`) | multilingual-e5, paraphrase-multilingual, bge-m3, bge-reranker | not implemented in the built-in tokenizer: measured, not covered |
| Byte-level BPE | CLIP text, ModernBERT | not implemented in the built-in tokenizer: measured, not covered |

Normalisation coverage (strip accents + lowercase), checked codepoint by codepoint up
to U+2FFFF: Latin (Latin-1, Extended-A/B, Extended Additional including Vietnamese),
IPA, combining diacritics, Greek and Coptic, Greek Extended (polytonic), Cyrillic,
Armenian, Hebrew, Arabic, Devanagari, Thai, Hangul jamo and syllables (algorithmic
decomposition), general punctuation and symbols (up to U+2BFF), kana, CJK, fullwidth
forms, emoji: 0 mismatches. Still outside the tables (1.0% of the tested codepoints,
1,835 of 183,376): CJK compatibility ideographs (U+F900-FAFF and U+2F800-2FA1F),
Cherokee capitals, Glagolitic, Coptic, Cyrillic Extended-B, Deseret and other rare
alphabets, precomposed Indic letters outside Devanagari, Hebrew presentation forms.
These codepoints go through the tokenizer unchanged, which differs from HF.

The same work fixed three pre-tokenisation differences with HF: a word with an unknown
piece yields a single `[UNK]` (not the pieces already matched), a word longer than 100
characters yields `[UNK]`, and Unicode punctuation, CJK ideographs and format
characters (zero-width space, etc.) are handled as HF does.

`BertNormalizer` options are read independently, as in HF: `lowercase`, `strip_accents`
(absent or `null` follows `lowercase`), `clean_text` and `handle_chinese_chars`. A
`Sequence` normalizer is supported for `Lowercase`, `StripAccents` and `NFD`/`NFKD`;
accents are stripped only when both `StripAccents` and a decomposition are present. The
"lowercase only" and "strip accents only" modes are also checked codepoint by codepoint.
Byte-level BPE keeps its legacy behaviour (not HF-faithful) until it is rewritten; its
ids are pinned by `test_tokenizer`. The per-family id tests only run when the exact
`tokenizer.json` is in the machine's model cache (the family is skipped otherwise).

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

## Thread safety

The rule: **a context may be shared between threads, but calls on it are
serialised** (text, sparse, reranker). Image contexts run calls in parallel.
For parallelism, use one context per thread or `TextEmbeddingPool`. Never free
a context while another thread still uses it, and never call back into a
context from an `embed_stream` callback (it holds the context lock: deadlock).

| Type | Shared between threads? | Why (source) |
|---|---|---|
| Text, ONNX and GGUF | Yes, calls serialised | `call_mutex` in `detail/text_embedding_impl.hpp:125,169`; the tokenizer writes scratch strings and the GGUF backend has one `llama_context` |
| Sparse, ONNX and GGUF | Yes, calls serialised | `call_mutex` in `sparse_text_embedding.h:171,348`; GGUF workers reuse ggml graphs per call |
| Reranker, ONNX and GGUF | Yes, calls serialised | `call_mutex` in `detail/reranker_impl.hpp:91,341` |
| Image | Yes, calls run in parallel | no per-context scratch, `OrtApi::Run` is thread-safe; only the counters are locked (`image_embedding.h:116`) |
| Stats getters (`*_stats`, `*_stats_v2`) | Yes, never wait for an inference | separate `stats_mutex` (same lines) |
| `TextEmbeddingPool` | Yes | each worker owns a context (serialised), calls from several threads queue per worker (`cpp/embedding_pool.hpp`) |
| LRU cache (`lembed_cache_*`) | Yes | one mutex (`detail/embedding_cache_impl.hpp:142`). Use `lembed_cache_get_copy()`: `lembed_cache_get()` returns a borrowed pointer (deprecated) |
| Autotune cache entries | Yes in a process | unique temporary name then atomic rename (`detail/autotune_cache.hpp:392`); several processes: last writer wins |
| Legacy tune cache (`lembed_tune_cache_*`) | Yes in a process | process-wide mutex (`detail/autotune_cache_impl.hpp:91`); several processes: last writer wins |
| Model downloader | Yes | unique temporary file per download, atomic publish (`detail/downloader_impl.hpp:339`). Two threads may download the same file twice |
| `lembed_last_error()` | Yes | thread-local buffer (`error.h:79`): read it on the thread that got the failure |
| `create` / `free` | Create contexts from any thread; free from one thread once no call is in flight | |

Tests: `tests/test_concurrency.cpp` (also built with ThreadSanitizer by the
`tsan.yml` workflow) and `tests/test_repeated_calls.cpp` (10 000 repeated calls,
bounded RSS growth, `-DLIBEMBEDDING_STABILITY_TESTS=ON`).

## Performance tips

For optimal performance:

1. **Reuse the model** — create it once and embed multiple batches
2. **Tune `batch_size`** — increase until memory saturation
3. **Try CoreML on Apple** — `provider="coreml"` (benchmark it: the gain depends on the model)
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
