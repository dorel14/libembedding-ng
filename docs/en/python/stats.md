---
nav_exclude: true
---

# Runtime Statistics

The `Stats` type provides usage statistics for an active embedding context. Every embedding class (`TextEmbedding`, `Reranker`, `ImageEmbedding`, `SparseTextEmbedding`) exposes a `.stats()` method.

## `Stats` structure

| Field | Type | Description |
|-------|------|-------------|
| `texts_embedded` | `int` | Total texts processed since context creation |
| `batches_run` | `int` | Total ONNX inference batches executed |
| `avg_latency_ms` | `float` | Average wall-clock time per embed() call (ms) |
| `cache_hits` | `int` | LRU cache hits (0 if disabled) |
| `cache_misses` | `int` | LRU cache misses (0 if disabled) |

## Retrieving statistics

```python
from libembedding import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
model.embed(["hello world", "test text"])

stats = model.stats()
print(f"Texts: {stats.texts_embedded}")
print(f"Batches: {stats.batches_run}")
print(f"Avg latency: {stats.avg_latency_ms:.2f} ms")
print(f"Cache hits: {stats.cache_hits}")
print(f"Cache misses: {stats.cache_misses}")
```

## Statistics with LRU cache

```python
from libembedding import TextEmbedding, EmbeddingCache

cache = EmbeddingCache(capacity=1024, dim=384)
model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_size=1024)

model.embed(["hello world"])
model.embed(["hello world"])  # Should be a cache hit

stats = model.stats()
print(f"Hits: {stats.cache_hits}")   # 1
print(f"Misses: {stats.cache_misses}")  # 1
```

## See also

- [Embedding Cache](python/cache.html) — `EmbeddingCache` class
- [Python API Reference](api_reference.html) — Python API overview
