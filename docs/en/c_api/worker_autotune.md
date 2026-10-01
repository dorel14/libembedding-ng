---
nav_exclude: true
---

# Worker Auto-Tune C API

This module detects optimal worker and session configuration for the llama.cpp backend.

## Types

| Type | Description |
|------|-------------|
| `lembed_worker_config_t` | Detected optimal worker configuration |

| Field | Type | Description |
|-------|------|-------------|
| `optimal_workers` | `int` | Recommended worker count |
| `optimal_threads` | `int` | Recommended thread count |
| `physical_cores` | `int` | Detected physical cores |
| `logical_cores` | `int` | Detected logical cores |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_detect_optimal_workers()` | `lembed_worker_config_t` | Detect the optimal worker/session configuration |
| `lembed_recommended_workers_for_model(model_path)` | `int` | Recommended worker count for a model |

> `lembed_detect_optimal_workers()` takes **no argument** and returns a struct,
> not `void`: it inspects the machine's hardware. The variant that takes a model
> path is `lembed_recommended_workers_for_model()`, which returns a plain `int`.

## Example

```c
#include <libembedding/worker_autotune.h>

lembed_worker_config_t cfg = lembed_detect_optimal_workers();
printf("workers: %d, threads: %d (physical=%d, logical=%d)\n",
       cfg.optimal_workers, cfg.optimal_threads,
       cfg.physical_cores, cfg.logical_cores);

// Model-specific recommendation
int workers = lembed_recommended_workers_for_model("meta-llama/Llama-3-8B");
```

## See also

- [Autotuner C API](autotuner.html) — Full autotuning
- [llama.cpp Backend C API](llamacpp_backend.html) — llama.cpp backend
