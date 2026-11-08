---
nav_exclude: true
---

# Performance Tuning

This guide covers libembedding's advanced performance features: session pooling, auto-tuning, automatic model selection, llama.cpp/GGUF tuning, bucketing, LRU cache, and FAST/BALANCED/QUALITY modes.

## Table of Contents

1. [Session Pool (EmbeddingPool)](#session-pool-embeddingpool)
2. [Auto-Tuning](#auto-tuning)
3. [Automatic Model Selection](#automatic-model-selection)
4. [llama.cpp / GGUF Performance](#llamacpp--gguf-performance)
5. [Benchmarks](#benchmarks)
6. [Best Practices](#best-practices)
7. [CPU Requirements](#cpu-requirements)

---

## CPU Requirements

The x86-64 wheels and release binaries run on any x86-64 CPU; no AVX or AVX2 is assumed:

- libembedding's own code (tokenizer, pooling, SPLADE pooling, similarity) is compiled for the
  x86-64 baseline; CI runs the unit tests under Intel SDE as Nehalem (SSE4.2), Sandy Bridge (AVX),
  Haswell (AVX2), Skylake-X (AVX-512) and Ice Lake (AVX-512 VNNI);
- GGUF models run on ggml, which ships one CPU library per instruction-set level (`ggml-cpu-x64` …
  `ggml-cpu-alderlake`) and loads the best one for the machine at run time;
  `libembedding.ggml_cpu_features()` (C: `lembed_ggml_cpu_features()`) names it. Record it with any
  GGUF benchmark;
- ONNX models run on ONNX Runtime, which selects its own kernels at run time.

The macOS wheel is arm64 (Apple Silicon, macOS 14 or later), built natively. For a local build that
will never leave the machine, `-DLIBEMBEDDING_ARCH_AVX2=ON` compiles libembedding's own code for
AVX2. The hot paths are in ONNX Runtime and ggml, and on an i7-1065G7 the difference stayed within
noise (dense, sparse ONNX, sparse GGUF: 0.98–1.01×).

---

## Session Pool (EmbeddingPool)

For small Transformer architectures (MiniLM, BGE-small, E5-small), **inter-session** parallelism (multiple independent ONNX sessions) is more efficient than **intra-session** parallelism (ORT threading).

### When to use it

| Scenario | Recommendation |
|----------|----------------|
| < 100 embeddings | Single `TextEmbedding` is sufficient |
| > 100 embeddings | `TextEmbeddingPool` recommended |
| Production / high throughput | `TextEmbeddingPool` + `autotune=True` |

### Usage

```python
from libembedding import TextEmbeddingPool

# Pool with 8 workers (independent ONNX sessions)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    workers=8,               # number of parallel sessions
    threads_per_worker=1,    # 1 thread per session (avoids contention)
    batch_size=64,
    offline=True,
)

embeddings = pool.embed(texts)
pool.close()
```

### Performance gain

| Configuration | Docs/s (short texts) | Speedup |
|---------------|------------------------|---------|
| 1 session × 4 threads | ~100 | 1.0x |
| 4 workers × 1 thread | ~265 | 2.6x |
| **8 workers × 1 thread** | **~360** | **3.6x** |

> **Golden rule**: `workers × threads ≤ CPU core count`

---

## Auto-Tuning

Auto-tuning automatically finds the optimal configuration (workers, threads, batch_size) for your hardware and corpus.

### Simple usage

```python
from libembedding import TextEmbeddingPool

# Autotune with synthetic corpus (default)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,           # enable auto-tuning
    offline=True,
)
# → benchmark on the first call (~13 s in QUICK on an i7-1065G7 4C/8T,
#   indicative: depends on the CPU and the model), then instant cache
```

### Autotune with your corpus (recommended)

For more accurate results, provide a sample of your actual texts:

```python
# Use a representative sample of your data
sample_texts = your_csv["text_column"].head(1000).tolist()

pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,    # your corpus
    autotune_max_samples=100,        # samples 100 representative texts
    offline=True,
)
```

### Autotune cache

Results are cached by machine + model, in a **single store** shared by the four
tasks: `%LOCALAPPDATA%\libembedding\autotune` (`~/.cache/libembedding/autotune`
outside Windows). Text sits at the root; `reranker/`, `sparse/` and `image/` have
their own subdirectory. The file name encodes the hardware fingerprint and a
short hash of the variant (workload, objective, mode).

| Event | Behavior |
|-------|----------|
| First call | Benchmark + save cache |
| Same machine + model + workload | Cache hit (a few ms) |
| Change CPU/ORT/library version/model | Cache miss → re-benchmark |
| Same model, different workload | Cache miss (the workload fingerprint is part of the key) |
| Same workload, different objective, mode or constraint | Cache miss |

A cache entry is reused only when **every** identity dimension matches: model,
hardware and software fingerprint, workload fingerprint, objective, mode and
constraints (p95 budget, `min_tokens`). A corrupt or truncated file is treated as
a miss, never as a result. Writes are atomic (temporary file then rename), so an
interruption cannot leave a partial result behind. Sparse, image and constrained
results are now cached too (previously a repeated sparse or image call re-ran a
95 to 170 s tuning). Older text-tuner entries are invalidated: they had been
measured on the default model whatever model was requested.

Tune on your own corpus:

```python
from libembedding import autotune

result = autotune(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    texts=my_documents,   # sampled down to max_sample_size (100 by default)
)
```

```python
from libembedding import clear_autotune_cache

# Clear cache for a model (all its variants: synthetic and custom)
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")

# Clear all cache
clear_autotune_cache()
```

### Full API

```python
# TextEmbedding with autotune
model = TextEmbedding(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)

# TextEmbeddingPool with autotune
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)
```

### Unified Auto-Tuning API

`autotune_unified()` is the single entry point for the four tasks
(`"embedding"`, `"reranking"`, `"sparse"`, `"image"`). It runs on
`lembed_autotune_ex()` on the C side ([C reference](c_api/autotuner.html)).

```python
from libembedding import autotune_unified, clear_autotune_cache_unified

# Defaults: QUICK, synthetic corpus, objective "throughput"
# (embedding, sparse, image) or "balanced" (reranking)
result = autotune_unified("embedding", "BAAI/bge-small-en-v1.5")

# Reranker on your real data: the LAST text is the query.
# p95 budget of 300 ms for one whole request, never truncate below 128 tokens.
result = autotune_unified(
    "reranking",
    "jinaai/jina-reranker-v1-turbo-en-quantized",
    objective="latency",
    texts=[*candidate_documents, user_query],
    max_latency_ms=300.0,
    min_tokens=128,
)
print(result.threads, result.batch_size, result.max_tokens)
if result.p95_latency_ms > 300.0:
    print("budget not reachable: best unconstrained configuration returned")

# Purge one task's cache for a model (None = every model of that task)
clear_autotune_cache_unified("reranking", "jinaai/jina-reranker-v1-turbo-en-quantized")
```

Parameters: `full=False` (QUICK) or `True` (FULL); `objective` (`"latency"`,
`"throughput"`, `"balanced"`, `"memory"`, `None` = task default); `texts`
(custom workload, sampled down to `max_sample_size=100`; rejected for `"image"`);
`max_latency_ms` (p95 budget per tuning call, `0` = none); `min_tokens`
(reranking only). `model_name` has a default only for `"embedding"` and
`"reranking"`.

**Objectives.** `latency` minimizes the tail cost P50 + ½(P95 − P50);
`throughput` maximizes throughput; `balanced` averages the two normalized costs;
`memory` minimizes the RSS delta (16 MB floor).

**Budget and fallback.** Only configurations whose p95 fits in `max_latency_ms`
are eligible. When none fits, the best **unconstrained** configuration is returned
(no error): compare `result.p95_latency_ms` with the budget. For the reranker the
fallback keeps the `min_tokens` floor. A tuning call is one request over the whole
workload (the documents of a reranking query, or the full batch); the latencies in
the result are **per call**, except for `"embedding"` where they are amortised
**per text**.

**`min_tokens` and truncation.** The reranker tuner chooses `max_tokens`
(256/64 in QUICK; 256/128/64/32 in FULL), which **truncates documents**: this is a
quality/speed trade-off, not a pure optimization. `min_tokens` sets the acceptable
floor. On equal performance (within 5 %), the configuration that truncates least
wins.

**What the tuner deliberately does not set.** For sparse, `top_k`, `min_weight`
and `storage_format` change the vectors produced, not only the speed: they are no
longer chosen by the tuner (only threads × batch are). `UnifiedTuningResult` still
exposes those three fields, filled with the model defaults, for information.

**Grids.** Text: splits of the *P* physical cores between workers and threads
(1×P, 2×P/2, P×1 in QUICK; + 1×logical and 4×P/4 in FULL) × batch (32 in QUICK;
16/32/64/128 in FULL), on a synthetic corpus of 32 (QUICK) or 256 (FULL) texts.
Reranker: threads (physical and logical cores in QUICK; + 1, 2, P/2 in FULL) ×
`max_tokens` × batch. Sparse and image: threads × batch.

### Measurement methodology

The four tuners share one engine:

- **Protocol.** QUICK: 1 warmup + 3 timed calls per configuration; FULL: 2 warmups
  + 10 calls. Every call is timed individually (monotonic clock); P50 and P95 are
  derived from them.
- **Racing.** A configuration whose **first two** timed calls both exceed 2.5 × the
  best P50 seen so far is stopped without finishing its calls (except for the
  `memory` objective, which does not rely on latency).
- **Session reuse.** Grid points that differ only by batch size share one ONNX
  Runtime session instead of rebuilding it.
- **Unit-free costs.** Every term is divided by the best value observed among the
  eligible configurations; a cost is "how many times worse than the best" (lower
  is better). Constraints (p95 budget, `min_tokens`) filter *before* normalization,
  so an ineligible configuration cannot distort the references.
- **5 % tie rule.** Costs within 5 % of the best are treated as a tie, resolved
  toward the preferred configuration (first grid point: fewer sessions, less
  truncation, smaller batch). The choice therefore does not flip on measurement
  noise.
- **Memory.** `memory_mb` is an **estimate**: RSS growth across "create + warmup",
  not an absolute peak. Peak RSS is a per-process high-water mark: do not read it
  as the exact footprint of one configuration.

### bench_autotune measurements

Measured with `benchmarks/bench_autotune.cpp` (public C API only), QUICK mode,
Release build, **i7-1065G7 (4C/8T)**, before → after the engine rework (commit
`e263a90` and following):

| Task (model) | Cold tuning | Repeated call (cache) | Throughput of chosen config vs library defaults |
|--------------|-------------|-----------------------|-------------------------------------------------|
| text (`all-MiniLM-L6-v2`) | 22.7 → 12.6 s | ~10–15 ms (already cached before) | ×1.77 |
| reranker (`jina-reranker-v1-turbo-en`) | 144 → 47 s | ~10–15 ms (already cached before) | ×3.26 |
| sparse (`Splade_PP_en_v1`) | 96 → 33 s | 95.6 s → 11 ms | ×1.01 |
| image (`clip-ViT-B-32-vision`) | 166 → 28 s | 168 s → 13 ms | ×1.01 |

How to read this table:

- Throughput is measured on an **evaluation workload distinct** from the tuner's,
  against the library defaults.
- **Text ×1.77**: the earlier ×1.23 was measured on the *wrong model* (the old
  tuner always measured the default model).
- **Reranker ×3.26**: obtained mostly by **truncating to 64 tokens**. It is a speed
  gain paid for with context seen by the model; use `min_tokens` if ranking quality
  matters.
- **Sparse ×1.01 and image ×1.01**: the defaults are already close to optimal on
  this machine; the tuner confirms it, it does not manufacture a gain.
- **Observed evaluation noise is 6 to 14 %** on this laptop: do not present small
  differences as significant. These durations depend on the CPU, the model and the
  workload: they are indicative, not guarantees.

To reproduce (the Windows CMake generator is multi-config; models must already be
in the local cache, nothing is downloaded):

```powershell
cmake -S . -B build -DLIBEMBEDDING_BUILD_BENCHMARKS=ON
cmake --build build --config Release --target bench_autotune
build/benchmarks/Release/bench_autotune.exe --task text   # text | rerank | sparse | image
# options: --model <code> (the task's model), --runs <n> (default 5, median reported)
```

The JSON report goes to stdout, progress to stderr.

### llama.cpp / GGUF Auto-Tuning

When building with llama.cpp support, the unified benchmark and autotuner can compare ONNX vs llama.cpp:

```python
from libembedding import Benchmark, CorpusType, Objective

bench = Benchmark()
comparison = bench.compare_all(
    onnx_path="/path/to/model.onnx",
    gguf_path="/path/to/model.Q4_K_M.gguf",
    corpus=CorpusType.MIXED,
    objective=Objective.BALANCED,
)
print(comparison.recommendation.backend)  # "onnx" or "llama.cpp"
```

---

## Automatic Model Selection

To automatically choose the best model for your hardware and use case:

```python
from libembedding import auto_select_model, TextEmbeddingPool

# Auto-select best model for your hardware + use case
result = auto_select_model("balanced")  # "speed", "quality", or "balanced"

print(f"Best model: {result.model_name}")
print(f"Config: {result.workers} workers × {result.threads} threads")
print(f"Throughput: {result.throughput_docs_sec:.0f} docs/s")

# Create pool with optimal config
pool = TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
)
embeddings = pool.embed(texts)
```

### Use cases

| Use case | Recommendation |
|----------|----------------|
| Real-time, latency critical | `"speed"` |
| Semantic search, max quality | `"quality"` |
| General production | `"balanced"` (default) |

---

## llama.cpp / GGUF Performance

For CPU-bound embedding workloads with small BERT-style models, the optimal configuration is:

### Recommendations

| Parameter | Recommended value | Reason |
|-----------|-------------------|--------|
| `threads` | **1** per session | Avoids contention on small BERT models |
| `workers` / `sessions` | **physical_cores × 2** (max 8) | Near-linear scaling until saturation |
| `batch_size` | 8-32 | Depends on average text length |
| `batch_strategy` | `LENGTH_BUCKET` | Reduces padding for heterogeneous corpora |

### Auto-tuning workers

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,      # auto-detect optimal session count
    cache_size=4096,        # optional LRU cache
)
```

### Preset modes

```python
# FAST : speed priority
model = TextEmbedding.from_mode("fast")

# BALANCED : speed/quality compromise (default)
model = TextEmbedding.from_mode("balanced")

# QUALITY : quality priority
model = TextEmbedding.from_mode("quality")
```

### LRU Cache

```python
# Enable LRU cache with 4096 entries
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    cache_size=4096,
)
```

### Baseline i7-1065G7 (MiniLM-L6-v2 Q4_K_M)

| Sessions | Threads | Docs/s |
|----------|---------|--------|
| 1 | 1 | 41.8 |
| 1 | 4 | 72.1 |
| 4 | 1 | 105.9 |
| **6** | **1** | **125.9** |
| 8 | 1 | 129.4 |
| 6 | 2 | 87.6 |

**Conclusion**: beyond 2 sessions, `threads=1` is always faster. Intra-session multithreading degrades performance on BERT embedding models.

---

## Benchmarks

### Test configuration

- CPU: Intel i7-1065G7 (4c/8t)
- OS: Windows 11
- Model: all-MiniLM-L6-v2 (384-dim)
- Backends: ONNX Runtime and llama.cpp (GGUF Q4_K_M)

### Impact of text length (ONNX)

| Tokens/text | Docs/s (8 workers) | ms/text |
|-------------|---------------------|---------|
| 16 | 696 | 6.0 |
| 64 | 150 | 16.9 |
| 128 | 62 | 32.6 |
| 256 | 15 | 68.6 |

> **Note**: Throughput is strongly dependent on text length. Benchmarks with short texts do not predict performance with long texts.

### ONNX vs llama.cpp comparison

| Backend | Config | Docs/s | RAM |
|---------|--------|--------|-----|
| ONNX | 8 workers × 1 thread | ~128 | ~500 MB |
| llama.cpp GGUF Q4_K_M | 6 sessions × 1 thread | ~126 | ~20 MB |

### llama.cpp configuration comparison

| Sessions | Threads | Docs/s | Efficiency |
|----------|---------|--------|------------|
| 1 | 1 | 41.8 | Low |
| 1 | 4 | 72.1 | Medium |
| 4 | 1 | 105.9 | Good |
| **6** | **1** | **125.9** | **Optimal** |
| 8 | 1 | 129.4 | Saturation |
| 6 | 2 | 87.6 | Degraded |

**Conclusion**: Inter-session parallelism (6×1) is optimal on this machine. Beyond 2 sessions, intra-session multithreading degrades performance.

### Model comparison

| Model | Docs/s (8w×1t) | RAM (8 workers) | Deterministic |
|-------|----------------|-----------------|---------------|
| MiniLM-L6-v2-Q (INT8) | 474-696 | 230 MB | ~1.6% variance |
| MiniLM-L6-v2 (FP32) | 305-361 | 740 MB | Yes |
| BGE-small-en (FP32) | 143-150 | 1.1 GB | Yes |

### Quantization: FP32 vs INT8

Measured by `benchmarks/quantization/bench_quantization.py` on 2026-10-01
(host `PC_Asus`, Windows 11 AMD64, CPU provider, Python 3.12.10, libembedding
1.8.0, commit `c2dbe95`). Corpus of 1,000 texts (50 warmup + 950 timed), models
pre-cached. Each variant runs in its own subprocess, so peak RSS is attributable
to the configuration that produced it. Every mode is measured at every batch size,
and each quantized variant is compared against the FP32 baseline **at the same
batch size** — comparing a batch-64 variant to a batch-8 baseline would measure the
batch size, not the quantization:

| Model | Variant | Weights | Throughput | Peak RAM | vs FP32 (same batch) |
|-------|---------|---------|------------|----------|---------|
| `all-MiniLM-L6-v2` | FP32 (`none`) | 86.2 MB | 90.5 docs/s @ batch 64 | 206 MB | -- |
| `all-MiniLM-L6-v2` | INT8 (`dynamic`) | **21.9 MB** (3.9x smaller) | **151.7 docs/s** @ batch 64 | **117 MB** | **1.7x faster**, 1.8x less RAM |
| `bge-small-en-v1.5` | FP32 (`none`) | 126.9 MB | 47.4 docs/s @ batch 64 | 249 MB | -- |
| `bge-small-en-v1.5` | INT8 (`dynamic`) | **32.2 MB** (3.9x smaller) | **95.5 docs/s** @ batch 64 | **120 MB** | **2.0x faster**, 2.1x less RAM |
| `bge-small-en-v1.5` | **FP16** (`LEMBED_QUANTIZATION_FP16`) | 63.4 MB (2.0x smaller) | 5.0 docs/s @ batch 64 | 177 MB | **9.5x slower**, 1.4x less RAM |

**Dynamic INT8 is the win on both models**, and it wins on every axis at once:
1.7x faster on MiniLM, 2.0x faster on BGE, 3.9x smaller on disk, and 1.8 to 2.1x
less resident memory. Nothing here is a trade-off except the accuracy, which this
benchmark does not measure.

> **One more trap, and it is a naming problem.** The four `Qdrant/*-onnx-Q`
> registry entries — `bge-small-en-v1.5` among them — ship **FP16** weights, not
> INT8: every one of their 149 initializers is `FLOAT16` and they carry an
> ORT-optimized graph (`Attention` + `SkipLayerNormalization` + `FastGelu` fused).
> They used to declare `LEMBED_QUANTIZATION_STATIC` and be described as
> "Quantized", so `quantization="static"` handed back float16 weights — 9.5x
> *slower* than FP32, which is what a reader of an earlier version of this table
> would conclude about "INT8". FP16 is the pathological case on a CPU without
> native FP16 arithmetic: it halves the file, then converts back to FP32 to
> compute. The registry now declares these `LEMBED_QUANTIZATION_FP16`, the
> descriptions say FP16, and `quantization="static"` correctly fails with
> `ModelNotFoundError` rather than silently returning float16.
>
> The genuine INT8 export of `bge-small-en-v1.5` is
> `onnx-community/bge-small-en-v1.5-ONNX` (`onnx/model_quantized.onnx`, 33 MB,
> 144 `INT8` + 6 `UINT8` initializers, 72 `MatMulInteger` fed by 48
> `DynamicQuantizeLinear`). It is registered as
> `LEMBED_TEXT_BGE_SMALL_EN_V15_INT8` and is what the INT8 row above measures.
>
> Run `python benchmarks/verify_quantization_modes.py` to check every registry
> entry's declared mode against the dtypes actually present in its file. It
> reports 0 mismatches; before the `fp16` fix it reported 4.

- **Accuracy is not measured here.** Dynamic INT8 embeddings vary slightly with
  batch composition (cosine ~0.984 vs the FP32 baseline). Validate on your own
  corpus before switching a quality-critical index.
- **Peak RAM is a real measurement now.** Each configuration ran alone, so the
  high-water mark is attributable. It exceeds the weight file because ONNX Runtime
  memory-maps the file and allocates its own arena on top — but the *difference*
  between configurations is real, and it moves in the same direction as the weight
  file for both models.
- **The reliable win is disk, cache and download footprint** -- the practical
  constraint for container images, air-gapped installs and cold starts.

#### Full per-batch results

`BAAI/bge-small-en-v1.5`:

| Batch | Variant | docs/s | ms/doc | Weights | Peak RAM | vs FP32 |
|-------|---------|--------|--------|---------|----------|---------|
| 8 | FP32 (`none`) | 48.2 | 20.75 | 126.9 MB | 249 MB | -- |
| 8 | INT8 (`dynamic`) | 98.0 | 10.21 | 32.2 MB | 120 MB | 2.0x faster |
| 8 | FP16 | 5.0 | 198.99 | 63.4 MB | 178 MB | 9.6x slower |
| 32 | FP32 (`none`) | 50.5 | 19.82 | 126.9 MB | 249 MB | -- |
| 32 | INT8 (`dynamic`) | 103.5 | 9.66 | 32.2 MB | 120 MB | 2.0x faster |
| 32 | FP16 | 5.0 | 201.47 | 63.4 MB | 177 MB | 10.1x slower |
| 64 | FP32 (`none`) | 47.4 | 21.10 | 126.9 MB | 249 MB | -- |
| 64 | INT8 (`dynamic`) | 95.5 | 10.47 | 32.2 MB | 120 MB | 2.0x faster |
| 64 | FP16 | 5.0 | 199.18 | 63.4 MB | 177 MB | 9.5x slower |
| 128 | FP32 (`none`) | 47.8 | 20.91 | 126.9 MB | 249 MB | -- |
| 128 | INT8 (`dynamic`) | 90.5 | 11.06 | 32.2 MB | 120 MB | 1.9x faster |
| 128 | FP16 | 5.0 | 200.67 | 63.4 MB | 178 MB | 9.6x slower |

`sentence-transformers/all-MiniLM-L6-v2`:

| Batch | Variant | docs/s | ms/doc | Weights | Peak RAM | vs FP32 |
|-------|---------|--------|--------|---------|----------|---------|
| 8 | FP32 (`none`) | 97.1 | 10.30 | 86.2 MB | 206 MB | -- |
| 8 | INT8 (`dynamic`) | 172.4 | 5.80 | 21.9 MB | 118 MB | 1.8x faster |
| 32 | FP32 (`none`) | 96.3 | 10.38 | 86.2 MB | 206 MB | -- |
| 32 | INT8 (`dynamic`) | 165.0 | 6.06 | 21.9 MB | 118 MB | 1.7x faster |
| 64 | FP32 (`none`) | 90.5 | 11.05 | 86.2 MB | 206 MB | -- |
| 64 | INT8 (`dynamic`) | 151.7 | 6.59 | 21.9 MB | 117 MB | 1.7x faster |
| 128 | FP32 (`none`) | 99.6 | 10.04 | 86.2 MB | 206 MB | -- |
| 128 | INT8 (`dynamic`) | 164.0 | 6.10 | 21.9 MB | 118 MB | 1.6x faster |

#### How to read these numbers

- **Peak RAM is measured, not dismissed.** Every configuration ran in its own
  process, so the resident high-water mark belongs to that configuration. It
  exceeds the weight file because ONNX Runtime memory-maps the file and
  allocates its own arena on top -- 118 MB for a 21.9 MB file, 249 MB for a
  126.9 MB file -- but the *difference* between configurations is real.
- **Run-to-run variance is around 10%.** The MiniLM FP32 batch-64 baseline has
  measured 82.7, 89.0, 90.5 and 99.6 docs/s across four runs on the same machine.
  Treat the small differences as noise and the order-of-magnitude ones
  (1.7x, 2.0x, 9.5x) as real; the tables print two decimals because the raw run
  does, not because the third digit is meaningful.
- **Every configuration ran on the same runtime.** The report records
  `onnxruntime` and `execution_provider` per configuration and fails loudly if
  they differ, because quantized throughput depends on the runtime as much as on
  the weights. It also records the initializer dtypes found in each weights file,
  so a mislabelled variant cannot enter the tables unnoticed.
- **The quantized path is stable across batch sizes here** (151.7 to 172.4 docs/s
  on MiniLM, 90.5 to 103.5 on BGE), so the batch size is not what drives the
  comparison. An earlier run showed MiniLM `dynamic` collapsing to 112.1 docs/s
  at batch 128; that has not reproduced, and it is the kind of outlier the
  ~10% variance note explains.
- **Not every mode exists for every model.** The registry now offers exactly
  `dynamic` (real INT8) for MiniLM, and `none` + `dynamic` + `fp16` for
  `bge-small-en-v1.5`. No model has a `static` (INT8-with-calibration) entry.
  Absent combinations are reported as *not measured* with the available modes
  listed, never as a failed run.
- **`quantization=` selects the weights, not just the session options.** Each
  quantized sibling is its own registry entry with its own `model_file`, and the
  requested mode is resolved *before* the session is created. When a model has no
  entry in that mode, creation fails with `LEMBED_ERROR_MODEL_NOT_FOUND` (Python:
  `ModelNotFoundError`) naming the modes the model actually has:

  ```python
  from libembedding import TextEmbedding

  # Resolved to the registry entry that provides the requested mode
  TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", quantization="dynamic")
  TextEmbedding("BAAI/bge-small-en-v1.5", quantization="dynamic")   # real INT8
  TextEmbedding("BAAI/bge-small-en-v1.5", quantization="fp16")      # float16

  # Naming the repo directly is equivalent and unambiguous -- this is what the
  # benchmark does
  TextEmbedding("Xenova/all-MiniLM-L6-v2", quantization="dynamic")
  TextEmbedding("onnx-community/bge-small-en-v1.5-ONNX", quantization="dynamic")
  ```

  Asking for a mode a model does not have is an error, not a silent fallback:

  ```python
  TextEmbedding("BAAI/bge-small-en-v1.5", quantization="static")
  # ModelNotFoundError: Model 'BAAI/bge-small-en-v1.5' has no 'static'
  # variant (available: dynamic, fp16, none)
  ```

  In C, the model enum selects the entry, or the `quantization` field of the v2
  options does:

  ```c
  opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15_INT8;  /* or ..._Q for the FP16 one */
  ```

  This resolution did not always exist: the mode used to be stamped on the context
  without changing the loaded file. That is why the archived 2026-09-27 benchmark is
  void -- it measured the same FP32 weights under three labels, and the 2026-09-29
  run published before the fix is void too. The 2026-10-01 run above is the first
  one that both resolves registry entries and measures a genuine INT8 file.

  Two caveats: `quantization="auto"` selects **no** weights (it loads the FP32
  default while `.quantization` reports `"auto"`) -- use
  `preferred_quantization="auto"` for a measured auto-selection. And
  `Reranker(quantization=...)` does not resolve a variant either: pass the name of
  the quantized entry, `jinaai/jina-reranker-v1-turbo-en-quantized`.

- **Accuracy is not measured here.** Dynamic INT8 embeddings vary slightly with
  batch composition (cosine ~0.984 vs the FP32 baseline). Validate on your own
  corpus before switching a quality-critical index; static INT8 (`_Q`) is the safer
  default.

Full report, machine block included, archived verbatim in
`docs/archive/benchmarks/quantization-2026-10-01/` — every figure published above
is traceable to it line by line.
`benchmarks/quantization/results.html` is only the benchmark's output path and is
overwritten by each run. The earlier runs are archived in
`quantization-2026-09-27/`, `quantization-2026-09-29/` and
`quantization-2026-09-30/`; all three are void — the first measured FP32 weights
under three labels, the second predates the variant-selection fix, and the third
measured FP16 files while calling them INT8. An incomplete run is archived in
`quantization-2026-09-30-partial/`.

```bash
# Each variant runs in its own subprocess: peak memory is attributable and one
# crashing configuration cannot take the benchmark down.
python benchmarks/quantization/bench_quantization.py --num-texts 1000
```

---

## Best Practices

### 1. For large corpora (> 100K texts)

```python
# Stratified sampling for large corpora
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=large_corpus,      # your 2M texts
    autotune_max_samples=100,         # samples 100 representative texts
    offline=True,
)
# → ~56s for autotune (once)
# → 66 docs/s in production
# → ~8h for 2M texts
```

### 2. For semantic search

```python
# Prioritize quality
pool = TextEmbeddingPool(
    "Xenova/bge-small-en-v1.5",      # better quality than MiniLM
    autotune=True,
    autotune_texts=documents,         # your documents
    offline=True,
)
```

### 3. For real-time

```python
# Prioritize speed
pool = TextEmbeddingPool(
    "Xenova/all-MiniLM-L6-v2",        # INT8 quantized version
    autotune=True,
    autotune_texts=queries,           # your short queries
    offline=True,
)
```

### 4. Batch size strategy

| Batch size | Usage |
|------------|-------|
| 8-16 | Long texts (> 100 tokens) |
| 32-64 | General use |
| 128-256 | Short texts (< 20 tokens) |

### 5. General best practices

- **Reuse the pool**: create it once, reuse it for all embeddings
- **Autotune once**: cache prevents re-benchmarking
- **Use your own texts** for autotune (more accurate than synthetic corpus)
- **Close the pool**: `pool.close()` or context manager `with`
- **Offline mode** in production: `offline=True` prevents downloads

### Complete example: RAG Pipeline

```python
from libembedding import TextEmbeddingPool, auto_select_model
import numpy as np

# 1. Automatic model selection (once)
result = auto_select_model("balanced")
print(f"Selected model: {result.model_name}")

# 2. Create pool with optimal config
with TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
    offline=True,
) as pool:

    # 3. Embed documents
    doc_embeddings = pool.embed(documents)

    # 4. Embed queries
    query_embeddings = pool.embed(queries)

    # 5. Nearest neighbor search
    scores = doc_embeddings @ query_embeddings.T
    top_k = np.argsort(scores, axis=0)[-5:]
```
