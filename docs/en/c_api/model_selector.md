---
nav_exclude: true
---

# Automatic Model Selection C API

This module selects the best model and configuration automatically based on available hardware.

## Types

| Type | Description |
|------|-------------|
| `lembed_model_selection_t` | Model selection result (declared in `autotuner.h`) |
| `lembed_use_case_t` | Use case |
| `lembed_model_candidate_t` | Model candidate |
| `lembed_hardware_info_t` | **Deprecated** — detected hardware |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Select model automatically |
| `lembed_model_select(logical_cores, ram_mb, use_case, out_selected)` | `int` | Pick a candidate from a given hardware configuration |
| `lembed_detect_hardware(out_info)` | `int` | **Deprecated** — detect hardware |

> `lembed_detect_hardware()` returns an **`int`**, not a `lembed_status_t`: its
> implementation wraps `lembed_cache_detect_hardware()` and returns 0 on failure.
> Use `lembed_cache_detect_hardware(lembed_cache_hardware_info_t*)` instead --
> that is what the header documentation recommends.

## `lembed_use_case_t`

| Constant | Description |
|----------|-------------|
| `LEMBED_USE_CASE_SPEED` | Optimized for speed |
| `LEMBED_USE_CASE_QUALITY` | Optimized for quality |
| `LEMBED_USE_CASE_BALANCED` | Balanced (default) |

## `lembed_model_candidate_t`

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `char[256]` | Model name (e.g. `"BAAI/bge-small-en-v1.5"`) |
| `dim` | `int` | Embedding dimension |
| `max_length` | `int` | Max token length |
| `pooling` | `int` | `LEMBED_POOLING_CLS` or `LEMBED_POOLING_MEAN` |
| `estimated_ram_mb` | `int` | Estimated RAM (MB) |
| `estimated_throughput` | `double` | Estimated throughput (docs/s) |

## `lembed_hardware_info_t` (deprecated)

| Field | Type | Description |
|-------|------|-------------|
| `cpu_model` | `char[256]` | CPU model |
| `physical_cores` | `int` | Physical cores |
| `logical_cores` | `int` | Logical cores |
| `ram_mb` | `int` | Total RAM (MB) |

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

`lembed_auto_select_model()` takes a **string**, not the enum:

| String | Matching constant | Description |
|--------|-------------------|-------------|
| `"speed"` | `LEMBED_USE_CASE_SPEED` | Optimized for speed |
| `"quality"` | `LEMBED_USE_CASE_QUALITY` | Optimized for quality |
| `"balanced"` | `LEMBED_USE_CASE_BALANCED` | Balanced (default) |

## Example

```c
#include <libembedding/model_selector.h>
#include <libembedding/autotuner.h>

lembed_model_selection_t result;
lembed_status_t status = lembed_auto_select_model("balanced", &result);
if (status == LEMBED_OK) {
    printf("Model: %s\n", result.model_code);
    printf("Workers: %d, Threads: %d, Batch: %d\n",
        result.workers, result.threads, result.batch_size);
}

// Selection from a known hardware configuration, without probing the machine
lembed_model_candidate_t chosen;
if (lembed_model_select(8, 16384, LEMBED_USE_CASE_SPEED, &chosen) > 0) {
    printf("Chosen: %s (%d dim)\n", chosen.model_name, chosen.dim);
}
```

## See also

- [Benchmark C API](c_api/embedding_benchmark.html) — Benchmark with scoring
