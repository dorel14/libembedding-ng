---
nav_exclude: true
---

# Benchmark C API

This module provides model benchmarking with Pareto scoring, constraints, and recommendations.

## Types

| Type | Description |
|------|-------------|
| `lembed_benchmark_result_t` | Benchmark result |
| `lembed_backend_config_t` | Backend config for benchmark |
| `lembed_benchmark_metrics_t` | Benchmark metrics |
| `lembed_benchmark_hardware_info_t` | Hardware information |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_benchmark_run(model, backend, corpus, config, result)` | `lembed_status_t` | Run a benchmark |
| `lembed_benchmark_autotune(model, backend, objective, result)` | `lembed_status_t` | Benchmark with autotune |
| `lembed_benchmark_detect_hardware(hw)` | `lembed_status_t` | Detect hardware |

## `lembed_benchmark_metrics_t`

| Field | Type | Description |
|-------|------|-------------|
| `throughput_docs_sec` | `double` | Throughput (docs/s) |
| `latency_p50_ms` | `double` | Median latency (ms) |
| `latency_p95_ms` | `double` | P95 latency (ms) |
| `load_time_ms` | `double` | Load time (ms) |
| `peak_memory_mb` | `double` | Peak memory (MB) |
| `dim` | `int` | Dimension |
| `num_texts` | `int` | Number of texts |
| `num_errors` | `int` | Number of errors |

## Corpus types

| Constant | Description |
|----------|-------------|
| `0` | Short (< 20 tokens) |
| `1` | Medium (20-80 tokens) |
| `2` | Long (80-200 tokens) |
| `3` | Very long (200+ tokens) |
| `4` | Mixed |
| `5` | Multilingual |
| `6` | Edge cases |

## Example

```c
#include <libembedding/embedding_benchmark.h>

lembed_benchmark_result_t result;
lembed_backend_config_t config = {
    .backend = "llama.cpp",
    .num_threads = 4,
    .batch_size = 32,
    .workers = 1,
};

lembed_status_t status = lembed_benchmark_run(
    "path/to/model.gguf",
    "llama.cpp",
    4,  // corpus type MIXED
    &config,
    &result
);
```

## See also

- [Benchmark Python](python/benchmark.html) — Python version
- [Model Selection C API](model_selector.html) — Model selection
