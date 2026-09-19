---
nav_exclude: true
---

# Worker Auto-Tune C API

This module detects optimal worker and session configuration for the llama.cpp backend.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_detect_optimal_workers(model, n_threads, n_sessions)` | `void` | Detect optimal workers |
| `lembed_recommended_workers_for_model(model_name)` | `int` | Recommended workers for a model |

## Example

```c
#include <libembedding/worker_autotune.h>

int n_workers = lembed_recommended_workers_for_model("meta-llama/Llama-3-8B");
printf("Recommended workers: %d\n", n_workers);

int threads, sessions;
lembed_detect_optimal_workers("meta-llama/Llama-3-8B", &threads, &sessions);
```

## See also

- [Autotuner C API](autotuner.html) — Full autotuning
- [llama.cpp Backend C API](llamacpp_backend.html) — llama.cpp backend
