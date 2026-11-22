---
nav_exclude: true
---

# Quantization

> **Français:** [Quantification](../quantization.html)

This page describes the quantization modes the library knows, how to pick one, what the `auto` mode does, and which tensor types the sparse GGUF runtime accepts.

**Project rule: never assume a quantized model is faster. Measure** (see [Measuring](#measuring)).

## Available modes

C enum `lembed_quantization_t` (`include/libembedding/types.h`):

| C constant | Python name | Value | Meaning |
|---|---|---|---|
| `LEMBED_QUANTIZATION_NONE` | `"none"` | 0 | FP32 weights |
| `LEMBED_QUANTIZATION_STATIC` | `"static"` | 1 | Static INT8 |
| `LEMBED_QUANTIZATION_DYNAMIC` | `"dynamic"` | 2 | Dynamic INT8 |
| `LEMBED_QUANTIZATION_AUTO` | `"auto"` | 3 | Request for automatic selection (never a state: see below) |
| `LEMBED_QUANTIZATION_FP16` | `"fp16"` | 4 | float16 weights |

`FP16` is not INT8 quantization: it halves the file and the resident set, but on a CPU without native FP16 arithmetic it converts back to FP32 to compute, so it trades throughput for size (comment in `types.h`).

The numeric values are part of the public ABI; `FP16` was appended last so as not to shift them.

## A variant is a different file

In the registry, every quantized variant is **its own entry** with its own `model_file`, for example `onnx/model_quantized.onnx` next to the FP32 file (`include/libembedding/model_registry.h`). Choosing a quantization mode therefore selects different weights, not a different session configuration.

- Each registry entry's `quantization` field tells which mode it implements. In Python, `list_text_models()` returns `ModelInfo` objects whose `quantization` attribute is `"none"`, `"static"`, `"dynamic"`, `"auto"` or `"fp16"` (`python/src/libembedding/models.py`).
- Not every model has every variant. `lembed_find_text_model_variant(model_name, quantization)` returns the entry index, or `-1` when the model has no entry in that mode; the thread-local error then names the requested mode and the modes the model does have (`model_registry.h`).
- Matching is done on the canonical registry name (`model_name`), not on an arbitrary HuggingFace repo id.

## Choosing a mode

### Python

```python
from libembedding import TextEmbedding

# Explicit mode: loads the matching registry variant
emb = TextEmbedding("BAAI/bge-small-en-v1.5", quantization="dynamic")

# Automatic selection (measure, then cache the decision)
emb = TextEmbedding("BAAI/bge-small-en-v1.5", preferred_quantization="auto")
```

(`python/src/libembedding/text_embedding.py`.)

- `quantization=`: `None` (default) keeps the model's default mode. Accepted values are the keys of `_QUANTIZATION_ENUM`: `"none"`, `"static"`, `"dynamic"`, `"auto"`, `"fp16"` (case-insensitive). Any other value raises `ValueError`. Note: the `TextEmbedding` docstring only lists `"none"`, `"static"`, `"dynamic"`.
- `preferred_quantization=`: `None` by default; accepts the same names, and `"auto"` triggers automatic selection.
- `TextEmbedding`, `TextEmbeddingPool` and `Reranker` accept `quantization=`.
- If the model has no variant for the requested mode, creation fails with `LEMBED_ERROR_MODEL_NOT_FOUND`, which is `ModelNotFoundError` in Python.

### C

The `quantization` field of `lembed_text_options_v2_t` (type `lembed_quantization_t`) is read by `lembed_text_embedding_create_v2()` (`include/libembedding/detail/text_embedding_onnx_impl.hpp`). For an explicit mode that differs from the requested entry's mode, it resolves the variant before creating the model; with no such variant it returns `LEMBED_ERROR_MODEL_NOT_FOUND`.

## What `auto` does

`lembed_quantization_auto_select()` (`include/libembedding/autotuner.h`):

1. **Candidates**: only the variants the registry actually ships for that model are considered (`NONE`, `STATIC`, `DYNAMIC`, `FP16`, at most 4). FP32 is the baseline (`detail/quantization_auto_impl.hpp`, `quant_auto_collect_variants`).
2. **Single set of weights**: if the family has only one registry entry, nothing is measured and the requested model is loaded as is.
3. **Measurement**: the first time for a (model, machine, library version) triple, the variants are benchmarked and the decision is cached; later calls read the cache. Variants whose weights are absent are **skipped, never downloaded**.
4. **Parameters**: `num_docs` is clamped to [16, 256] (0 = default 16); `dry_run = 1` measures without writing the cache.
5. **Decision rule** (`quant_auto_pick`):
   - FP32 stays unless a variant is at least **1.05x** faster (`kQuantMinGain`);
   - if dynamic INT8 cleared that bar, it remains the default choice unless another variant is at least **1.15x** faster than it (`kQuantDynamicPreference`);
   - with no usable FP32 baseline measurement, FP32 is kept.
6. **Memory**: `file_mb` is the on-disk size of the weights (including the `_data` sidecar), not the resident set: peak RSS is a per-process high-water mark and cannot be attributed to a variant measured after another one.
7. **Failure**: if the selection fails during model creation, the requested entry is kept; `auto` is an optimisation, not a reason to refuse to load.

The result (`lembed_quantization_choice_t`, `types.h`) holds the winner, the entry index to load, the `measured[4]` table (docs/s, latency in ms, `file_mb`), `from_cache` and a `reason` sentence. In Python the selected mode is stored on the object (`self._quantization`) and the reason in `self._quantization_reason`.

For the cache file and the autotune cache format, see [C API — Autotune Cache](c_api/autotune_cache.html).

## Sparse GGUF runtime: tensor types

The in-house ggml runtime for sparse GGUF (see [Sparse GGUF runtime](gguf_sparse_runtime.html)) validates every required tensor at load time (`include/libembedding/detail/gguf/gguf_weights.hpp`, `validate_tensors`):

| Rule | Detail |
|---|---|
| Accepted types | `F32`, `F16`, `Q8_0` only. Any other type is refused by name: `tensor '<name>' has type <type>; this runtime supports f32, f16 and q8_0`. |
| Vectors (1-D) | Biases and LayerNorm weights must be `F32`: `tensor '<name>' is a vector of type <type>; vectors must be f32`. Converters keep 1-D tensors F32. |
| `token_type_embd.weight` | Must be `F32` (`token_type_embd.weight is not F32`). |
| Shape | Checked against what the graph expects; otherwise `tensor '<name>' has shape [...], expected [...]`. A missing tensor gives `missing tensor '<name>'`. |

These refusals are reported as `LEMBED_ERROR_UNSUPPORTED` with the detail in `lembed_last_error()` (`detail/sparse_text_embedding_gguf_impl.hpp`). See [Troubleshooting](troubleshooting.html).

This list applies to **sparse GGUF** only. It says nothing about the quantizations accepted by the dense llama.cpp backend, which reads GGUF files through llama.cpp itself.

## Quality by variant

Retrieval quality measured on BEIR SciFact (test: 5183 documents, 300 queries), brute-force cosine, no timing. "Cosine vs FP32" is the mean cosine between a document embedding and the FP32 registry entry's embedding of the same document.

| Model | Registry entry | Quantization | nDCG@10 | MRR@10 | Recall@100 | Cosine vs FP32 |
|---|---|---|---:|---:|---:|---:|
| bge-small-en-v1.5 | Xenova/bge-small-en-v1.5 | none | 0.7127 | 0.6822 | 0.9417 | 1.00000 |
| bge-small-en-v1.5 | onnx-community/bge-small-en-v1.5-ONNX | dynamic | 0.7077 | 0.6766 | 0.9467 | 0.99831 |
| bge-small-en-v1.5 | Qdrant/bge-small-en-v1.5-onnx-Q | fp16 | 0.7127 | 0.6822 | 0.9417 | 1.00000 |
| all-MiniLM-L6-v2 | Qdrant/all-MiniLM-L6-v2-onnx | none | 0.6541 | 0.6071 | 0.9317 | 1.00000 |
| all-MiniLM-L6-v2 | Xenova/all-MiniLM-L6-v2 | dynamic | 0.6519 | 0.6077 | 0.9367 | 0.99336 |

A registry variant is a different repository, so possibly a different export of the model: the table measures the entry a user gets, not the quantizer alone. Measured on an i7-1065G7 with ONNX Runtime 1.31.0, single run (no confidence interval: differences around 0.005 over 300 queries are not significant).

Reproduce:

```powershell
python benchmarks/quantization/bench_quality.py --models BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2
```

The script writes `quality_results.json` and `quality_results.md` to `benchmarks/quantization/` (option `--out`), with machine metadata.

## Measuring

A smaller quantization is not necessarily faster: it depends on the CPU, the model, the batch and the sequence length. Before choosing:

- Measure your own variants with the same method (one sub-process per configuration, warmup separate from the timed run, median over at least 10 rounds, a gap below 5% is noise): see [Benchmark methodology](benchmark_methodology.html), which also requires the metadata (machine, model, corpus, date).
- Let `auto` measure for you (above), or use the tools in [Performance Tuning](performance_tuning.html).
- Do not reuse a figure measured on another machine: the `auto` cache is keyed on a hardware fingerprint for that reason.
