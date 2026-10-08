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
| `lembed_autotune(model_name, mode, out)` | `lembed_status_t` | Autotune a text model |
| `lembed_autotune_custom(model_name, texts, n, mode, out)` | `lembed_status_t` | Autotune with custom corpus |
| `lembed_autotune_unified(task, model_name, mode, out)` | `lembed_status_t` | Unified autotune (all types) |
| `lembed_autotune_unified_config(task, model_name, target_latency_ms, out)` | `lembed_status_t` | Unified autotune with a latency budget |
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Pick the best model for the machine |
| `lembed_sparse_autotune(model_name, mode, out)` | `lembed_status_t` | Autotune a sparse model |
| `lembed_image_autotune(model_name, mode, out)` | `lembed_status_t` | Autotune an image model |
| `lembed_reranker_autotune(model_name, mode, objective, out)` | `lembed_status_t` | Autotune a reranker |
| `lembed_reranker_autotune_custom(model_name, texts, n, mode, objective, out)` | `lembed_status_t` | Autotune a reranker on a supplied corpus |
| `lembed_reranker_autotune_constrained(model_name, mode, objective, min_tokens, max_latency_ms, out)` | `lembed_status_t` | Constrained autotune |
| `lembed_reranker_auto_config(model_name, target_latency_ms, objective, out)` | `lembed_status_t` | Latency-budget autoconfig |
| `lembed_reranker_auto_config_profile(model_name, profile, out)` | `lembed_status_t` | Profile-based autoconfig |
| `lembed_quantization_auto_select(model, threads, batch_size, num_docs, dry_run, out)` | `lembed_status_t` | Pick the fastest quantization variant for this machine |

## Automatic quantization selection

`lembed_quantization_auto_select()` answers a question the caller cannot answer
alone: **on this machine, which weights file is fastest for this model?** Quantized
variants are separate registry entries with their own `model_file`, so choosing a
mode means choosing a file.

Only the variants **actually shipped** are considered: a model with no `static`
sibling is never compared against a phantom one. Variants whose weights are absent
are **skipped**, never downloaded just to be measured.

### Decision rule

1. no measurable FP32 baseline -> FP32;
2. a variant must beat FP32 by **more than 5 %** -- below that the difference sits
   inside the noise of a short benchmark and a weights swap is not worth it;
3. **biased toward dynamic INT8**: another variant must beat it by **more than
   15 %** to take over, because it is the safer of the quantized options quality
   wise.

### Cache

The cache key **is** the hardware fingerprint: it is baked into the file name (CPU,
ONNX Runtime version, library version) and the entry identity is re-checked on
read, so a decision measured on another machine or version can never be served.
Dedicated `quantization/` location, so a quantization decision can never overwrite
a threads/batch tuning entry.

Cost: **~3.5 s on the very first load** of a model, then **free**.

### Two-phase measurement

The variants are not comparable in price: FP16 measured **6x slower** than dynamic
INT8 on the same machine and would have dominated the whole selection without
teaching anything a few hundred milliseconds would not have.

| Phase | What it does | Bounds |
|---|---|---|
| 1 -- probe | Rejects clear losers. **4 documents max, 250 ms max** per variant. | Bounded at ~750 ms for three variants, whatever their relative speed |
| 2 -- measure | Full corpus, separate warmup, baseline and survivors only. | 16 documents, 2-document warmup |

The **time** budget is the important half: a document-count-only probe still pays
full price on a pathological variant, which is exactly the case worth avoiding.

The probe is a **rejector, not a ranker**: its first document is untimed (the
first embed pays kernel selection and buffer allocation), and it only eliminates a
variant that is more than 3x slower than the best seen so far.

### Measured (bge-small-en-v1.5, i7-1065G7)

| Variant | docs/s | Weights |
|---|---|---|
| FP32 | 15.1 | 126.9 MB |
| **Dynamic INT8** | **25.6** (1.7x) | **32.2 MB** (3.9x smaller) |
| FP16 | 2.5 (6x slower) | 63.4 MB |

> **Measurement note.** The size reported is the on-disk size of the weights, not
> resident memory: peak RSS is a per-process high-water mark, so it can never go
> down and cannot be attributed to a variant measured after another one. Counting
> only the graph file would be worse still -- an ONNX model past the protobuf size
> limit is a small graph plus a weights sidecar, and bge-small's INT8
> `model_quantized.onnx` is 413 KB next to 33 MB of weights. The sidecar is
> therefore counted, otherwise INT8 would report **smaller** than FP32.

### Example

```c
#include <libembedding/autotuner.h>

lembed_quantization_choice_t choice;
if (lembed_quantization_auto_select(LEMBED_TEXT_BGE_SMALL_EN_V15,
                                    /*num_threads*/ 0,
                                    /*batch_size*/  32,
                                    /*num_docs*/    0,   /* 0 = default */
                                    /*dry_run*/     0,
                                    &choice) == LEMBED_OK) {
    printf("quantization = %s\n", choice.reason);
    printf("from_cache   = %d\n", choice.from_cache);
}

/* Or let create_v2 resolve it: LEMBED_QUANTIZATION_AUTO in the v2 options runs
 * the exact same selection and persists the decision. */
lembed_text_options_v2_t opts = lembed_text_options_v2_default();
opts.base.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
opts.quantization = LEMBED_QUANTIZATION_AUTO;
lembed_text_embedding_t* ctx = NULL;
lembed_text_embedding_create_v2(&opts, &ctx);
```

`lembed_quantization_choice_t` is declared in `types.h` rather than
`autotuner.h`: the creation path has to resolve `AUTO` without pulling in the
autotune machinery.

The decision rule is a **pure** function, separated from the measurement, which
makes it testable in milliseconds without a model or network
(`tests/test_quantization_auto.cpp`).

> **Both forms of the name are accepted.** The registry gives every model two
> different strings -- the HuggingFace repo (`model_code`,
> `Qdrant/all-MiniLM-L6-v2-onnx`) and the canonical name (`model_name`,
> `sentence-transformers/all-MiniLM-L6-v2`). Every autotune entry point resolves
> either one through `lembed_resolve_text_model()` /
> `lembed_resolve_reranker_model()`, and the cache identity is always the
> `model_code` form. A call using either form finds the same model and purges the
> same entries.
>
> Resolution is exact `model_code` first, then exact `model_name`. One useful
> consequence: a repo belongs to exactly one entry, so
> `Xenova/all-MiniLM-L6-v2` (the INT8 repo) selects the quantized entry, while the
> canonical name -- which carries no mode -- selects the first entry declaring it,
> the FP32 one. An unknown name returns -1 with a diagnostic, never entry 0.
>
> This aligns the C API with what the Python bindings already do in
> `resolve_text_model()`, and with what the reranker benchmark already did on its
> own. Previously `lembed_autotune` required the HF repo while the note announced
> the canonical name, so the note and the example below contradicted each other.
>
> `lembed_find_text_model_variant()` (below) stays deliberately strict: it matches
> the **canonical name**, because that is the only way to name "this model, in
> this quantization mode" without ambiguity.
>
> Both fields are described by `lembed_model_info_t` in
> `include/libembedding/model_registry.h`.

## Cache clearing

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_autotune_clear_cache(model_name)` | `void` | Clear the text autotune cache; `NULL` = whole cache |
| `lembed_reranker_autotune_clear_cache(model_name)` | `void` | Clear the reranker autotune cache; `NULL` = whole reranker cache |
| `lembed_autotune_unified_clear_cache(task, model_name)` | `void` | Dispatch to the cache of the given task |

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

## Autotune cache

Results are stored under `%LOCALAPPDATA%\libembedding\autotune`
(`$HOME/.cache/libembedding/autotune` outside Windows, and
`./libembedding_autotune_cache` as a last resort when none of those variables is
set). Rerankers use a separate `reranker/` subdirectory.

An entry is reused only when its identity matches **in full**: model, variant,
corpus fingerprint, objective and mode. A result measured on one corpus is
therefore never returned for another, and an unreadable or corrupt file is
treated as a cache miss.

| Variant label | Used by |
|---------------|---------|
| `synthetic` | Default text tuning, on a synthetic corpus |
| `custom` | Text or reranker tuning on a supplied corpus |
| `default` | Default reranker tuning, on a synthetic corpus |

Note that the default label is **not** the same for text (`synthetic`, in
`autotune_bench_text.hpp`) and for reranker (`default`, via
`default_reranker_identity()` in `autotune_bench_reranker.hpp`). No variant
selection depends on those labels: `lembed_find_text_model_variant()` compares the
quantization mode and the `model_name`, not `synthetic`/`custom`.

The identity also includes a hardware and software fingerprint (CPU brand,
logical and physical cores, ONNX Runtime major.minor version, libembedding
major.minor version), so a result measured on another machine or after a version
bump is never served again.

`lembed_autotune_clear_cache(model_name)` removes every entry for a model (all
variants); `lembed_reranker_autotune_clear_cache(model_name)` does the same for
rerankers. Passing `NULL` empties the matching cache entirely. Entries are
matched on the `model` field read **from inside** the file, not on the file
name, which is a hash of the hardware fingerprint.

> **Do not confuse this** with the lower-level `tune_cache.json` file and the
> `lembed_tune_cache_*` API (`lembed_tune_cache_load`, `lembed_tune_cache_save`,
> `lembed_tune_cache_clear`, see [Autotune cache](autotune_cache.md)), which are
> keyed by hardware + software + model + backend fingerprint and are not driven
> by `lembed_autotune_clear_cache`.

## Related function: quantization variant resolution

`lembed_find_text_model_variant(model_name, quantization)` is **not** an
autotuning function: it belongs to the registry API
(`include/libembedding/model_registry.h`) and does not yet have a dedicated page
in this documentation.

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_find_text_model_variant(model_name, quantization)` | `int` | Resolve a quantization request to the registry entry that provides it. Returns the entry index, or `-1` with `lembed_last_error()` naming the requested mode and the modes available. |
| `lembed_resolve_text_model(model)` | `int` | Resolve a model given by its HF repo **or** its canonical name to its entry. `-1` plus a diagnostic if unknown. |
| `lembed_resolve_reranker_model(model)` | `int` | Same, for rerankers. |

These two resolvers are the ones the autotune functions above use.

The match is done on `model_name` (the canonical name), not `model_code`:
passing a HuggingFace repo matches no entry.

`LEMBED_QUANTIZATION_AUTO` is not a weights selection: no registry entry carries
that mode, so the function always returns `-1` -- and **still sets a thread-local
error**, like any unknown mode. `lembed_text_embedding_create_v2` only calls it
for an explicitly requested mode, never for `AUTO`.

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
