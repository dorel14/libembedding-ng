---
nav_exclude: true
---

# Autotune Cache C API

This module manages autotuning result caches (workers, threads, batch_size) with hardware/software/model fingerprinting.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_tune_cache_clear()` | `lembed_status_t` | Clear all tuning cache entries |
| `lembed_tune_cache_path()` | `const char*` | Cache file path |
| `lembed_tune_cache_fingerprint(model, hw, sw, result)` | `lembed_status_t` | Compute cache fingerprint |

## Example

```c
#include <libembedding/autotune_cache.h>

const char *path = lembed_tune_cache_path();
printf("Cache path: %s\n", path);

lembed_status_t status = lembed_tune_cache_clear();

char model[] = "BAAI/bge-small-en-v1.5";
char hw[] = "Intel i7-1065G7";
char sw[] = "Windows 11";
char fingerprint[256];
size_t fingerprint_len = 256;
status = lembed_tune_cache_fingerprint(
    model, hw, sw, fingerprint, &fingerprint_len);
```

## See also

- [Autotuner C API](autotuner.html) — Full C autotuning
- [Worker Auto-Tune C API](worker_autotune.html) — llama.cpp worker detection
