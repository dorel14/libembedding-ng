---
nav_exclude: true
---

# Embedding Cache C API

This module provides a thread-safe LRU cache for dense embeddings in C.

## Types

| Type | Description |
|------|-------------|
| `lembed_cache_t` | Opaque cache handle |
| `lembed_cache_config_t` | Cache configuration |
| `lembed_cache_hardware_info_t` | Hardware information |

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `capacity` | `4096` | Maximum capacity |
| `ttl_seconds` | `0` | Time-to-live (0 = no expiry) |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_cache_create(config)` | `lembed_cache_t*` | Create a cache |
| `lembed_cache_free(cache)` | `void` | Free a cache |
| `lembed_cache_get_copy(cache, text, out_vec, capacity, out_dim)` | `int` (copied=1, miss=0, buffer too small=-1) | **Recommended** — copies under the lock into a caller-owned buffer |
| `lembed_cache_get(cache, text, out_vec, out_dim)` | `int` (hit=1, miss=0) | Deprecated — returns a **borrowed** pointer, invalidated on eviction |
| `lembed_cache_put(cache, text, vec, dim)` | `void` | Store embedding |
| `lembed_cache_clear(cache)` | `void` | Clear the cache |
| `lembed_cache_capacity(cache)` | `int` | Cache capacity |
| `lembed_cache_size(cache)` | `int` | Current size |
| `lembed_cache_config_default()` | `lembed_cache_config_t*` | Default configuration |
| `lembed_cache_detect_hardware(hw)` | `lembed_status_t` | Detect hardware |

## Pointer lifetime

`lembed_cache_get()` returns a pointer **owned by the cache**, not a copy. It
stays valid until the entry is overwritten (`lembed_cache_put`), evicted by LRU
pressure, cleared (`lembed_cache_clear`) or freed (`lembed_cache_free`). The
caller must copy what it needs and must **never** `free()` that pointer.

Since the cache releases its lock before returning, a concurrent eviction can
free the buffer while the caller is still copying it. Use
`lembed_cache_get_copy()`, which copies **under the lock**: no cache-owned
pointer ever leaves the library.

## Example

```c
#include <libembedding/embedding_cache.h>

lembed_cache_config_t cfg = {
    .capacity = 2048,
    .ttl_seconds = 3600,
};

lembed_cache_t *cache = lembed_cache_create(&cfg);

float vec[] = {0.1f, 0.2f, 0.3f};
lembed_cache_put(cache, "hello world", vec, 3);

// Retrieve: the copy happens under the cache lock
int dim = 0;
int rc = lembed_cache_get_copy(cache, "hello world", NULL, 0, &dim);
if (rc == -1) {
    // rc == -1: the key exists and `dim` is the size to allocate
    float *out_vec = (float *)malloc(dim * sizeof(float));
    if (lembed_cache_get_copy(cache, "hello world", out_vec, dim, &dim) == 1) {
        printf("Cache hit, dim=%d, out_vec[0]=%f\n", dim, out_vec[0]);
    }
    free(out_vec);
}

printf("Capacity: %d, Size: %d\n",
    lembed_cache_capacity(cache),
    lembed_cache_size(cache));

lembed_cache_free(cache);
```

## See also

- [Embedding Cache Python](python/cache.html) — Python version
- [Runtime Statistics](python/stats.html) — `Stats` type
