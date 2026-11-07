---
nav_exclude: true
---

# Autotuner C API

This module provides auto-tuning to find the best configuration on the current machine: threads, batch_size, and depending on the task workers (text) or `max_tokens` (reranker). The four tasks (text, reranker, sparse, image) run on **one shared measurement engine** (see [measurement methodology](../performance_tuning.html#measurement-methodology)) and share a single cache store.

> **Availability.** Like every auto-tuner entry point, these functions are compiled into the **shared library** (`python/csrc/libembedding_shlib.cpp`: the DLL on Windows, the Python wheel), not into the header-only `LIBEMBEDDING_IMPLEMENTATION` build.

## Types

| Type | Description |
|------|-------------|
| `lembed_tuning_result_t` | Text autotuning result (legacy) |
| `lembed_unified_tuning_result_t` | Unified autotuning result (all tasks) |
| `lembed_autotune_options_t` | Options of `lembed_autotune_ex()`: task, mode, objective, workload, budget, `min_tokens` |
| `lembed_model_selection_t` | Model selection result |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `LEMBED_AUTOTUNE_QUICK` | `0` | Quick mode: reduced grid, 1 warmup + 3 timed calls per configuration |
| `LEMBED_AUTOTUNE_FULL` | `1` | Exhaustive mode: extended grid, 2 warmups + 10 timed calls per configuration |

**Indicative** cold durations (empty cache), measured in QUICK mode on an i7-1065G7 laptop (4C/8T): text ~13 s, reranker ~47 s, sparse ~33 s, image ~28 s (see [Performance](../performance_tuning.html#bench_autotune-measurements)). They depend on the CPU, the model and the workload; FULL mode takes longer, with no guaranteed duration. A repeated call (warm cache) returns in a few milliseconds.

## Recommended entry point: `lembed_autotune_ex`

`lembed_autotune_ex()` exposes the objective, a custom workload and constraints for **every** task. The legacy functions (listed below) remain available and are facades over the same code; new code should prefer `lembed_autotune_ex()`. (Documentation-only deprecation: no compiler attribute is applied.)

```c
lembed_autotune_options_t lembed_autotune_options_default(lembed_task_t task);

lembed_status_t lembed_autotune_ex(
    const char* model_name,
    const lembed_autotune_options_t* options,   /* NULL = defaults of LEMBED_TASK_EMBEDDING */
    lembed_unified_tuning_result_t* result);
```

Always start from `lembed_autotune_options_default(task)`, then override fields:

| Field | Default | Semantics |
|-------|---------|-----------|
| `task` | argument of `_default()` | `LEMBED_TASK_EMBEDDING`, `_RERANKING`, `_IMAGE` or `_SPARSE` |
| `mode` | `LEMBED_AUTOTUNE_QUICK` | QUICK or FULL |
| `objective` | `THROUGHPUT` (text, sparse, image); `BALANCED` (reranking) | What to optimize (see [Objectives](#objectives)) |
| `texts`, `n_texts` | `NULL`, `0` | Custom workload; `NULL`/`0` = built-in synthetic workload. Text, sparse: the texts to embed. **Reranking: the last text is the query**, the others are the documents. **Image: rejected** (`LEMBED_ERROR_INVALID_ARGUMENT`). |
| `max_latency_ms` | `0` (none) | **p95 budget per tuning call**. One call = one request over the whole workload: the documents of a reranking query, or the full text/image batch. |
| `min_tokens` | `0` (none) | Reranking only: smallest acceptable `max_tokens`. |

Behavior:

- **Budget and fallback.** Only configurations whose p95 meets `max_latency_ms` are eligible. When none fits, the best **unconstrained** configuration is returned with `LEMBED_OK`: compare `result.p95_latency_ms` with the budget to detect that case. For the reranker, the fallback keeps the `min_tokens` floor.
- **Latencies.** `latency_ms` and `p95_latency_ms` are **per call**, except for `LEMBED_TASK_EMBEDDING` where they are amortised **per text** (as in `lembed_tuning_result_t`).
- **Cache.** The result is cached (see [Autotune cache](#autotune-cache)); a repeated call returns in a few milliseconds.
- **Errors.** `LEMBED_ERROR_MODEL_NOT_FOUND` (unknown model), `LEMBED_ERROR_INVALID_ARGUMENT` (inconsistent options), `LEMBED_ERROR_ONNX_RUNTIME` (no configuration could run). `lembed_last_error()` is set on every autotune error path.

Example: a reranker with a p95 budget and a truncation floor.

```c
#include <libembedding/libembedding.h>
#include <stdio.h>

int main(void) {
    /* Real workload: candidate documents, the query LAST. */
    const char* texts[] = {
        "Candidate document 1 ...",
        "Candidate document 2 ...",
        "Candidate document 3 ...",
        "my user query",                   /* last text = query */
    };

    lembed_autotune_options_t opts = lembed_autotune_options_default(LEMBED_TASK_RERANKING);
    opts.mode           = LEMBED_AUTOTUNE_QUICK;
    opts.objective      = LEMBED_OBJECTIVE_LATENCY;
    opts.texts          = texts;
    opts.n_texts        = 4;
    opts.max_latency_ms = 300.0;   /* p95 budget for one whole request */
    opts.min_tokens     = 128;     /* never truncate documents below 128 tokens */

    lembed_unified_tuning_result_t r;
    lembed_status_t st = lembed_autotune_ex(
        "jinaai/jina-reranker-v1-turbo-en-quantized", &opts, &r);
    if (st != LEMBED_OK) {
        fprintf(stderr, "autotune: %s\n", lembed_last_error());
        return 1;
    }
    printf("threads=%d batch=%d max_tokens=%d p95=%.1f ms\n",
           r.threads, r.batch_size, r.max_tokens, r.p95_latency_ms);
    if (r.p95_latency_ms > opts.max_latency_ms)
        printf("budget not reachable: best unconstrained configuration returned\n");
    return 0;
}
```

### What the tuner sets, per task

| Task | Grid | Notes |
|------|------|-------|
| `LEMBED_TASK_EMBEDDING` | Splits of the *P* physical cores between workers and threads: 1×P, 2×P/2, P×1 in QUICK (+ 1×logical and 4×P/4 in FULL) × batch (32 in QUICK; 16/32/64/128 in FULL) | Measures the model you asked for. Synthetic corpus: 32 texts (QUICK) / 256 (FULL). |
| `LEMBED_TASK_RERANKING` | threads = physical and logical cores in QUICK (+ 1, 2, P/2 in FULL) × `max_tokens` (256/64 in QUICK; 256/128/64/32 in FULL) × batch | `max_tokens` **truncates documents**: it is a quality/speed trade-off, to be bounded with `min_tokens`. On a tie (5 %), the configuration that truncates least wins. |
| `LEMBED_TASK_SPARSE` | threads × batch | `top_k`, `min_weight` and `storage_format` are **no longer** tuned: they change the vectors produced, not only the speed. The result returns the model defaults (informational fields). |
| `LEMBED_TASK_IMAGE` | threads × batch | Custom workload not supported. |

## Autotuning functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_autotune_ex(model_name, options, out)` | `lembed_status_t` | **Recommended.** Every task, objective, workload, p95 budget, `min_tokens` |
| `lembed_autotune_options_default(task)` | `lembed_autotune_options_t` | Default options of a task |
| `lembed_autotune(model_name, mode, out)` | `lembed_status_t` | Legacy: autotune a text model |
| `lembed_autotune_custom(model_name, texts, n, mode, out)` | `lembed_status_t` | Legacy: autotune with custom corpus |
| `lembed_autotune_unified(task, model_name, mode, out)` | `lembed_status_t` | Facade over `lembed_autotune_ex` with default options |
| `lembed_autotune_unified_config(task, model_name, target_latency_ms, out)` | `lembed_status_t` | Facade with a latency budget; honors the budget for **every** task, sparse included (it used to return `UNSUPPORTED` for sparse) |
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Pick the best model for the machine |
| `lembed_sparse_autotune(model_name, mode, out)` | `lembed_status_t` | Legacy: autotune a sparse model (threads × batch) |
| `lembed_image_autotune(model_name, mode, out)` | `lembed_status_t` | Legacy: autotune an image model |
| `lembed_reranker_autotune(model_name, mode, objective, out)` | `lembed_status_t` | Legacy: autotune a reranker |
| `lembed_reranker_autotune_custom(model_name, texts, n, mode, objective, out)` | `lembed_status_t` | Legacy: autotune a reranker on a supplied corpus |
| `lembed_reranker_autotune_constrained(model_name, mode, objective, min_tokens, max_latency_ms, out)` | `lembed_status_t` | Legacy: constrained autotune |
| `lembed_reranker_auto_config(model_name, target_latency_ms, objective, out)` | `lembed_status_t` | Legacy: latency-budget autoconfig (now honors `objective`) |
| `lembed_reranker_auto_config_profile(model_name, profile, out)` | `lembed_status_t` | Profile-based autoconfig |
| `lembed_quantization_auto_select(model, threads, batch_size, num_docs, dry_run, out)` | `lembed_status_t` | Pick the fastest quantization variant for this machine |

Functions marked "Legacy" are kept for source and binary compatibility (no signature changed); they expose fewer settings than `lembed_autotune_ex()`.

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
| `lembed_autotune_unified_clear_cache(task, model_name)` | `void` | Purge the cache of one task (text, reranker, sparse **and** image) for a model; `NULL` = every model of that task (for `LEMBED_TASK_EMBEDDING`, the root-level text entries only). |

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
| `LEMBED_OBJECTIVE_LATENCY` | Minimize latency (cost = P50 + ½(P95 − P50)) |
| `LEMBED_OBJECTIVE_THROUGHPUT` | Maximize throughput |
| `LEMBED_OBJECTIVE_BALANCED` | Balanced: mean of the normalized latency and throughput costs |
| `LEMBED_OBJECTIVE_MEMORY` | Minimize memory (RSS delta, 16 MB floor) |

Costs are **normalized and unit-free** (each term is divided by the best value observed on the grid), so docs/s and milliseconds are never mixed. Defaults of `lembed_autotune_options_default()`: `THROUGHPUT` for text, sparse and image; `BALANCED` for reranking. Details in [Performance — measurement methodology](../performance_tuning.html#measurement-methodology).

## Autotune cache

A **single store** for the four tasks: `%LOCALAPPDATA%\libembedding\autotune`
(`$HOME/.cache/libembedding/autotune` outside Windows, and
`./libembedding_autotune_cache` as a last resort when none of those variables is
set). **Text** entries sit at the root; the other tasks have their own
subdirectory: `reranker/`, `sparse/`, `image/`. Quantization decisions live in
`quantization/`.

An entry is reused only when its identity matches **in full**: resolved model,
hardware and software fingerprint, workload fingerprint, objective, mode and
constraints (p95 budget, `min_tokens`). A result measured on one workload is
therefore never returned for another, and an unreadable or corrupt file is
treated as a cache miss. **Constrained** results are cached too, as are sparse
and image results (they were not: a repeated call re-ran a 95 to 170 s tuning).

The model identity is always the resolved form (for text, the `model_code`; see
the note on the two forms of the name above). The hardware and software
fingerprint covers the CPU brand, logical and physical cores, the ONNX Runtime
major.minor version and the libembedding major.minor version, so a result
measured on another machine or after a version bump is never served again.

> **Invalidation on upgrade.** Older text-tuner cache entries are invalidated:
> they had been measured on the default model whatever model was requested.

`lembed_autotune_clear_cache(model_name)` removes every entry for a model in the
root store and its subdirectories (an unknown name is a no-op);
`lembed_reranker_autotune_clear_cache(model_name)` does the same for
rerankers; `lembed_autotune_unified_clear_cache(task, model_name)` targets one
task (sparse and image included). Passing `NULL` empties that task's cache
only: for `LEMBED_TASK_EMBEDDING`, only the root-level text entries are removed.
Only `lembed_autotune_clear_cache(NULL)` deletes the **whole** autotune
directory, subdirectories included. Entries are
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

## Example (legacy API, text)

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
