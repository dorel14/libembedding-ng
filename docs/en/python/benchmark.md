---
nav_exclude: true
---

# Unified Benchmark

The `Benchmark` module provides unified comparison of ONNX and llama.cpp backends on the same corpus, with auto-tuning. It enables hardware detection, benchmark execution, and results comparison.

## Hardware detection

```python
from libembedding import Benchmark

bench = Benchmark()
hw = bench.hardware
print(f"CPU: {hw.cpu_name}, Cores: {hw.logical_cores}, RAM: {hw.ram_mb} MB")
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `cpu_name` | `str` | Processor name |
| `physical_cores` | `int` | Number of physical cores |
| `logical_cores` | `int` | Number of logical cores |
| `os_name` | `str` | Operating system name |
| `ram_mb` | `int` | RAM in MB |
| `features` | `str` | Detected CPU features |

## Corpus categories

| Constant | Description |
|----------|-------------|
| `CorpusType.SHORT` | Short texts (< 20 tokens) |
| `CorpusType.MEDIUM` | Medium texts (20-80 tokens) |
| `CorpusType.LONG` | Long texts (80-200 tokens) |
| `CorpusType.VERY_LONG` | Very long texts (200+ tokens) |
| `CorpusType.MIXED` | Mixed lengths |
| `CorpusType.MULTILINGUAL` | Multilingual |
| `CorpusType.EDGE_CASES` | Edge cases |

## Optimization objectives

| Constant | Description |
|----------|-------------|
| `Objective.LATENCY` | Minimize latency |
| `Objective.THROUGHPUT` | Maximize throughput |
| `Objective.BALANCED` | Latency/throughput tradeoff |
| `Objective.MEMORY` | Minimize memory |

## Auto-tuning

```python
from libembedding import Benchmark, Objective

bench = Benchmark()
result = bench.autotune(
    "path/to/model.gguf",
    "llama.cpp",
    objective=Objective.BALANCED,
)
```

## Benchmark a model

```python
result = bench.run(
    "path/to/model.gguf",
    "llama.cpp",
    corpus=CorpusType.MIXED,
    sessions=4,
    threads=1,
)
```

## Multi-backend comparison

```python
comparison = bench.compare_all(
    onnx_path="path/to/model.onnx",
    gguf_path="path/to/model.gguf",
    objective=Objective.BALANCED,
)
print(comparison.summary())
```

## Multi-model sweep

```python
models = {
    "model_a": "path/to/model_a.onnx",
    "model_b": "path/to/model_b.gguf",
}
results = bench.sweep(models, corpora=[CorpusType.SHORT, CorpusType.MIXED])
```

## Utilities

```python
from libembedding import cache_path, clear_cache

path = cache_path()
clear_cache()
```

## See also

- [Python API Reference](api_reference.html) — Python API overview
- [Runtime Statistics](python/stats.html) — `Stats` type
