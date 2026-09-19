---
nav_exclude: true
---

# Embedding Cache (LRU)

The `EmbeddingCache` class provides a thread-safe LRU cache for dense embeddings. It can be used standalone or integrated automatically into `TextEmbedding` and `Reranker` via the `cache_size` parameter.

## Construction

```python
from libembedding import EmbeddingCache

cache = EmbeddingCache(capacity=4096, ttl_seconds=0, dim=0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity` | `4096` | Maximum number of entries |
| `ttl_seconds` | `0` | Time-to-live in seconds (0 = no expiry) |
| `dim` | `0` | Expected embedding dimensionality (for validation) |

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `put(text, vec)` | `None` | Store an embedding in the cache |
| `get(text, dim=None)` | `np.ndarray \| None` | Look up a cached embedding by text key |
| `clear()` | `None` | Remove all entries |
| `stats()` | `dict` | Cache statistics (capacity, current_size) |
| `close()` | `None` | Release underlying C resources |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `capacity` | `int` | Maximum cache capacity |
| `current_size` | `int` | Current number of entries |

## Context manager

```python
with EmbeddingCache(capacity=1024) as cache:
    cache.put("hello", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    vec = cache.get("hello")
```

## Default configuration

```python
from libembedding import cache_config_default

cfg = cache_config_default()
# {"capacity": 4096, "ttl_seconds": 0}
```

## Integration with TextEmbedding

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_size=2048,  # Enable internal LRU cache
)
```

## See also

- [Python API Reference](api_reference.html) — Python API overview
- [Runtime Statistics](python/stats.html) — `Stats` type
