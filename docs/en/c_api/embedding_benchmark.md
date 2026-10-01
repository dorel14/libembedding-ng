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
| `lembed_benchmark_weights_t` | Scoring weights (quality / throughput / cost) |
| `lembed_benchmark_constraints_t` | Hard limits |
| `lembed_corpus_type_t` | Test corpus category |

> The hardware type is not `lembed_benchmark_hardware_info_t`: it is
> `lembed_cache_hardware_info_t`, declared in `autotune_cache.h`, alongside
> `lembed_cache_detect_hardware()` and `lembed_cache_detect_software()`.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_benchmark_run(model_path, backend, corpus_type, config, result)` | `lembed_status_t` | Run a benchmark |
| `lembed_benchmark_compare(onnx_path, gguf_path, corpus_type, results)` | `int` | Same model on both backends (0 to 2 results) |
| `lembed_benchmark_autotune(model_path, backend, objective, result)` | `lembed_status_t` | Benchmark with autotune |
| `lembed_benchmark_select_model(model_dir, objective, constraints, weights, result)` | `lembed_status_t` | Hard constraints, then Pareto frontier, then scoring |
| `lembed_benchmark_detect_sessions(model_path, max_sessions, optimal, throughput)` | `lembed_status_t` | Detect the optimal session count |
| `lembed_benchmark_get_corpus(type, out_texts, out_count)` | `lembed_status_t` | Fetch the test corpus (static, do not free) |
| `lembed_benchmark_default_cache_dir()` | `const char*` | Default GGUF cache directory |
| `lembed_benchmark_profile_weights(obj)` | `lembed_benchmark_weights_t` | Normalized weights for an objective |
| `lembed_benchmark_custom_weights(q, t, c)` | `lembed_benchmark_weights_t` | Custom weights, renormalized |

> `lembed_benchmark_detect_hardware()` **does not exist**: hardware detection is
> `lembed_cache_detect_hardware(lembed_cache_hardware_info_t*)`.

## `lembed_benchmark_metrics_t`

| Field | Type | Description |
|-------|------|-------------|
| `throughput_docs_sec` | `float` | Throughput (docs/s) |
| `latency_p50_ms` | `float` | Median latency (ms) |
| `latency_p95_ms` | `float` | P95 latency (ms) |
| `load_time_ms` | `float` | Load time (ms) |
| `peak_memory_mb` | `float` | Peak memory (MB) |
| `dim` | `int` | Dimension |
| `num_texts` | `int` | Number of texts |
| `num_errors` | `int` | Number of errors |

> The metrics are **`float`**, not `double`.

## Corpus types

| Constant | Value | Description |
|----------|-------|-------------|
| `LEMBED_CORPUS_SHORT` | `0` | Short (< 20 tokens) |
| `LEMBED_CORPUS_MEDIUM` | `1` | Medium (20-80 tokens) |
| `LEMBED_CORPUS_LONG` | `2` | Long (80-200 tokens) |
| `LEMBED_CORPUS_VERY_LONG` | `3` | Very long (200+ tokens) |
| `LEMBED_CORPUS_MIXED` | `4` | Mixed |
| `LEMBED_CORPUS_MULTILINGUAL` | `5` | Multilingual |
| `LEMBED_CORPUS_EDGE_CASES` | `6` | Edge cases |

## Example

```c
#include <libembedding/embedding_benchmark.h>

lembed_backend_config_t config = {
    .backend = "onnx",     /* char[32], not a pointer */
    .num_threads = 4,
    .batch_size = 32,
    .workers = 1,
};

lembed_benchmark_result_t result;
lembed_status_t status = lembed_benchmark_run(
    "path/to/model.onnx",
    "onnx",
    LEMBED_CORPUS_MIXED,
    &config,
    &result
);
```

## See also

- [Benchmark Python](python/benchmark.html) — Python version
- [Model Selection C API](model_selector.html) — Model selection
