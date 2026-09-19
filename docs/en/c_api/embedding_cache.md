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
| `lembed_cache_get(cache, text, out_vec, out_dim)` | `int` (hit=1, miss=0) | Retrieve embedding |
| `lembed_cache_put(cache, text, vec, dim)` | `void` | Store embedding |
| `lembed_cache_clear(cache)` | `void` | Clear the cache |
| `lembed_cache_capacity(cache)` | `int` | Cache capacity |
| `lembed_cache_size(cache)` | `int` | Current size |
| `lembed_cache_config_default()` | `lembed_cache_config_t*` | Default configuration |
| `lembed_cache_detect_hardware(hw)` | `lembed_status_t` | Detect hardware |

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

float *out_vec = NULL;
int out_dim = 0;
int hit = lembed_cache_get(cache, "hello world", &out_vec, &out_dim);
if (hit) {
    printf("Cache hit, dim=%d\n", out_dim);
}

printf("Capacity: %d, Size: %d\n",
    lembed_cache_capacity(cache),
    lembed_cache_size(cache));

lembed_cache_free(cache);
```

## See also

- [Embedding Cache Python](python/cache.html) — Python version
- [Runtime Statistics](python/stats.html) — `Stats` type
