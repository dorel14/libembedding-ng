---
nav_exclude: true
---

# Automatic Model Selection C API

This module selects the best model and configuration automatically based on available hardware.

## Types

| Type | Description |
|------|-------------|
| `lembed_model_selection_t` | Model selection result |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Select model automatically |
| `lembed_detect_hardware(hw)` | `lembed_status_t` | Detect hardware |

## `lembed_model_selection_t`

| Field | Type | Description |
|-------|------|-------------|
| `model_code` | `const char*` | HuggingFace model code |
| `model_name` | `const char*` | Model name |
| `dim` | `int` | Embedding dimension |
| `workers` | `int` | Number of workers |
| `threads` | `int` | Number of threads |
| `batch_size` | `int` | Batch size |
| `throughput_docs_sec` | `double` | Throughput (docs/s) |
| `latency_ms` | `double` | Latency (ms) |
| `memory_mb` | `double` | Memory (MB) |
| `score` | `double` | Quality score |

## Use cases

| Value | Description |
|-------|-------------|
| `"speed"` | Optimized for speed |
| `"quality"` | Optimized for quality |
| `"balanced"` | Balanced (default) |

## Example

```c
#include <libembedding/model_selector.h>

lembed_model_selection_t result;
lembed_status_t status = lembed_auto_select_model("balanced", &result);
if (status == LEMBED_OK) {
    printf("Model: %s\n", result.model_code);
    printf("Workers: %d, Threads: %d, Batch: %d\n",
        result.workers, result.threads, result.batch_size);
}
```

## See also

- [Benchmark C API](c_api/embedding_benchmark.html) — Benchmark with scoring
