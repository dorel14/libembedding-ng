---
nav_exclude: true
---

# Autotune Cache C API

This module manages autotuning result caches (workers, threads, batch_size) with hardware/software/model fingerprinting.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_tune_cache_load(hw, sw, model, backend, entry)` | `lembed_status_t` | Load an entry (hit) or return `LEMBED_ERROR_CACHE_MISS` |
| `lembed_tune_cache_save(entry)` | `lembed_status_t` | Store the entry and all measured configs |
| `lembed_tune_cache_clear()` | `lembed_status_t` | Clear all tuning cache entries |
| `lembed_tune_cache_path()` | `const char*` | Cache file path |
| `lembed_tune_cache_key(hw, sw, model, backend, key_out)` | `void` | Compute the entry key |
| `lembed_tune_cache_add_config(entry, config)` | `void` | Append a measured config to the entry |
| `lembed_tune_cache_set_best(entry, idx)` | `void` | Set the best config index |
| `lembed_cache_detect_hardware(hw)` | `lembed_status_t` | Detect hardware (CPU, cores, OS, RAM, features) |
| `lembed_cache_detect_software(sw)` | `lembed_status_t` | Detect libembedding / llama.cpp versions |

## Key format

The key is the **FNV-1a 64-bit** hash of the full fingerprint (CPU, OS,
libembedding version, llama.cpp version, model id, backend), rendered as 16
hexadecimal characters. The readable fields are too long for a fixed buffer: a
truncated key would collide across different fingerprints and return a result
measured on another machine or model.

The caller buffer must hold at least `LEMBED_TUNE_CACHE_KEY_SIZE` bytes (17).
This format replaced the readable concatenation in September 2026:
`LEMBED_TUNE_CACHE_SCHEMA_VERSION` is now `2` and old entries are unreachable,
so those fingerprints are re-measured instead of being misread.

## Example

```c
#include <libembedding/autotune_cache.h>

const char *path = lembed_tune_cache_path();
printf("Cache path: %s\n", path);

lembed_status_t status = lembed_tune_cache_clear();

lembed_cache_hardware_info_t hw;
lembed_cache_software_info_t sw;
lembed_cache_model_info_t model;
lembed_cache_detect_hardware(&hw);
lembed_cache_detect_software(&sw);
snprintf(model.model_id, sizeof(model.model_id), "%s", "BAAI/bge-small-en-v1.5");

char key[LEMBED_TUNE_CACHE_KEY_SIZE];
lembed_tune_cache_key(&hw, &sw, &model, "llama.cpp", key);
printf("Cache key: %s\n", key);
```

## See also

- [Autotuner C API](autotuner.html) — Full C autotuning
- [Worker Auto-Tune C API](worker_autotune.html) — llama.cpp worker detection
