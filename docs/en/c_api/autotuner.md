---
nav_exclude: true
---

# Autotuner C API

This module provides comprehensive auto-tuning to find optimal configuration (workers, threads, batch_size).

## Types

| Type | Description |
|------|-------------|
| `lembed_tuning_result_t` | Autotuning result |
| `lembed_unified_tuning_result_t` | Unified autotuning result |
| `lembed_model_selection_t` | Model selection result |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `LEMBED_AUTOTUNE_QUICK` | `0` | Quick mode (5-15s) |
| `LEMBED_AUTOTUNE_FULL` | `1` | Exhaustive mode (30-120s) |

## Autotuning functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_autotune(model_code, mode, out)` | `lembed_status_t` | Autotune a text model |
| `lembed_autotune_custom(model_code, texts, n, mode, out)` | `lembed_status_t` | Autotune with custom corpus |
| `lembed_autotune_unified(task, model_code, mode, out)` | `lembed_status_t` | Unified autotune (all types) |
| `lembed_autotune_clear_cache(model_name)` | `void` | Clear autotune cache |
| `lembed_reranker_autotune(model_code, mode, objective, out)` | `lembed_status_t` | Autotune a reranker |
| `lembed_reranker_autotune_constrained(...)` | `lembed_status_t` | Constrained autotune |
| `lembed_reranker_auto_config(...)` | `lembed_status_t` | Latency-budget autoconfig |
| `lembed_reranker_auto_config_profile(...)` | `lembed_status_t` | Profile-based autoconfig |
| `lembed_sparse_autotune(model_code, mode, out)` | `lembed_status_t` | Autotune a sparse model |
| `lembed_image_autotune(model_code, mode, out)` | `lembed_status_t` | Autotune an image model |

## Unified tasks

| Constant | Description |
|----------|-------------|
| `LEMBED_TASK_EMBEDDING` | Text embedding task |
| `LEMBED_TASK_RERANKING` | Reranking task |
| `LEMBED_TASK_IMAGE` | Image embedding task |
| `LEMBED_TASK_SPARSE` | Sparse embedding task |

## Objectives

| Constant | Description |
|----------|-------------|
| `LEMBED_OBJECTIVE_LATENCY` | Minimize latency |
| `LEMBED_OBJECTIVE_THROUGHPUT` | Maximize throughput |
| `LEMBED_OBJECTIVE_BALANCED` | Balanced |
| `LEMBED_OBJECTIVE_MEMORY` | Minimize memory |

## Example

```c
#include <libembedding/autotuner.h>

lembed_tuning_result_t result;
lembed_status_t status = lembed_autotune(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    LEMBED_AUTOTUNE_QUICK,
    &result
);
if (status == LEMBED_OK) {
    printf("Workers: %d, Threads: %d, Batch: %d\n",
        result.workers, result.threads, result.batch_size);
}
```

## See also

- [Autotune Cache C API](autotune_cache.html) — Cache fingerprinting
- [Worker Auto-Tune C API](worker_autotune.html) — Worker detection
