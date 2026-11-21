---
nav_exclude: true
---

# API conventions

> **Français:** [Conventions de l'API](../api_conventions.html)

Rules shared by the public C API (`include/libembedding/*.h`) and its Python translation. This page describes what the code does; when in doubt, the headers are authoritative.

## Naming

- Public functions, types and enums use the `lembed_` prefix (types: `lembed_<module>_t`, enums: `LEMBED_<MODULE>_<VALUE>`).
- Handles are opaque: `lembed_text_embedding_t`, `lembed_sparse_embedding_ctx_t`, `lembed_image_embedding_t`, `lembed_reranker_t` (`types.h`). Only a pointer is ever handled.

## Options: defaults, then override

Each family has a `lembed_<family>_options_default()` function returning the struct by value: `lembed_text_options_default()`, `lembed_sparse_options_default()`, `lembed_image_options_default()`, `lembed_reranker_options_default()`. Start from that default and change only the fields you need; a hand-filled struct leaves forgotten fields at zero, which is not always the default (for example the reranker `backend`: `0` = `ONNX`, the default is `AUTO`).

```c
lembed_reranker_options_t o = lembed_reranker_options_default();
o.num_threads = 4;
lembed_reranker_t* rr = NULL;
lembed_status_t s = lembed_reranker_create(&o, &rr);
```

## Creation, release, ownership

| Object | Creation | Release |
|---|---|---|
| Text / sparse / image / reranker context | `lembed_<module>_create*()`: returns a `lembed_status_t`, the handle comes out through `**out` | `lembed_text_embedding_free`, `lembed_sparse_text_embedding_free`, `lembed_image_embedding_free`, `lembed_reranker_free` |
| LRU cache | `lembed_cache_create()`: returns the pointer (or `NULL`) | `lembed_cache_free` |
| Dense result (`lembed_embeddings_t`) | filled by `*_embed()` | `lembed_embeddings_free` |
| Sparse result (`lembed_sparse_embeddings_t`) | filled by `*_embed()` | `lembed_sparse_embeddings_free` (frees each `indices` / `values`, then `items`) |
| Reranking result (`lembed_rerank_results_t`) | filled by `*_rerank()` | `lembed_rerank_results_free` |

- The buffers of a result belong to the caller, who releases them with the matching `*_free` function, not with a bare `free()`. These functions reset the pointers to `NULL` and the counters to 0, and accept a `NULL` pointer (`types.h`).
- Introspection pointers (`lembed_*_desc()`, `lembed_*_desc_v2()`, `model_name()`) belong to the context: they are invalid after `*_free()`. Copy the fields you need, do not keep the pointer (`text_embedding.h`, `sparse_text_embedding.h`).
- `*_backend_name()` and `*_effective_provider()` return a static string that must not be freed.

## Return codes and detailed error

Fallible functions return a `lembed_status_t` (`error.h`):

| Code | Value |
|---|---|
| `LEMBED_OK` | 0 |
| `LEMBED_ERROR_INVALID_ARGUMENT` | 1 |
| `LEMBED_ERROR_OUT_OF_MEMORY` | 2 |
| `LEMBED_ERROR_ONNX_RUNTIME` | 3 |
| `LEMBED_ERROR_TOKENIZER` | 4 |
| `LEMBED_ERROR_DOWNLOAD` | 5 |
| `LEMBED_ERROR_IO` | 6 |
| `LEMBED_ERROR_MODEL_NOT_FOUND` | 7 |
| `LEMBED_ERROR_UNSUPPORTED` | 8 |
| `LEMBED_ERROR_BATCH_SIZE` | 9, reserved: never returned by the current implementation |
| `LEMBED_ERROR_LLAMA` | 10 |
| `LEMBED_ERROR_CACHE_MISS` | 11 |
| `LEMBED_ERROR_GGUF` | 12, the library's own GGUF runtime (sparse), distinct from `LLAMA` |

- `lembed_status_message(status)` returns a static string per code.
- `lembed_last_error()` returns the detail of the last error **on the calling thread** (a 1024-byte `thread_local` buffer, truncated beyond that). It is written on failure; a successful call does not clear it (only the autotune entry points reset it to empty at the start, `detail/autotune_result.hpp`). Read `lembed_last_error()` right after a failure, on the same thread, and do not use it to detect success: the `lembed_status_t` is authoritative.

## Concurrency

- Calls on the **same** text, sparse or reranker context are serialised by an internal mutex (`call_mutex`): sharing a context between threads is safe, but brings no parallelism (`detail/text_embedding_impl.hpp`, `detail/reranker_impl.hpp`, `sparse_text_embedding.h`). To run in parallel, use one context per thread (`TextEmbeddingPool`).
- Statistics counters have a separate lock: reading stats does not wait for an inference in flight.
- The LRU cache is thread-safe for its functions, with the caveat of `lembed_cache_get()` below.
- `lembed_last_error()` is per thread, see above.

## ABI stability

- The layout (order, size, type of fields) of a public struct does not change. A new field is not added to an existing struct: it goes into a **versioned** struct that embeds the old one as its first `base` field.
  - `lembed_text_options_v2_t` (`base` + `quantization`), `lembed_reranker_options_v2_t`, `lembed_model_desc_v2_t`, `lembed_stats_v2_t` (`types.h`), `lembed_sparse_tuning_result_v2_t` (`autotuner.h`).
  - The matching functions carry the `_v2` suffix: `lembed_text_embedding_create_v2()`, `lembed_reranker_create_v2()`, `lembed_sparse_autotune_v2()`, `lembed_*_stats_v2()`, `lembed_*_desc_v2()`.
- A public enum is neither reordered nor shortened: a value that fell out of use stays in place. `LEMBED_ERROR_BATCH_SIZE` is no longer returned but kept so that `LLAMA`, `CACHE_MISS` and `GGUF` do not change value; `LEMBED_ERROR_GGUF` was appended at the end (`error.h`).
- Any change to the C API is mirrored in `python/src/libembedding/_cdefs.h` and `libembedding.def`. `tests/test_struct_sizes.cpp` freezes the sizes on the C side, `python/tests/test_cdefs.py` on the cffi side.

## Deprecations

Deprecation is documentary (a comment in the header), with no compiler `deprecated` attribute; ADR-004 lists the removals planned for a future major version.

- `lembed_cache_get()`: returns a **borrowed** pointer into the cache, which a concurrent `put`, eviction or `clear` can free. Never `free()` it. Use `lembed_cache_get_copy(cache, text, out_vec, capacity, &out_dim)`: copies under the lock; returns 1 (success), 0 (missing or invalid argument) or -1 (`capacity` too small: `out_dim` gives the required size). See `embedding_cache.h`.

## Backend selection

The backend is chosen by file extension and entry point (`.gguf`: llama.cpp; `.onnx`, a HuggingFace identifier or an unknown name: ONNX Runtime). See [Model routing](model_routing.html).

The `backend` field (`LEMBED_BACKEND_ONNX`, `LEMBED_BACKEND_LLAMACPP`, `LEMBED_BACKEND_AUTO`) is an official option of the **reranker only**:

- `lembed_reranker_options_t.backend` (default `AUTO`) forces a backend. In `AUTO`, the reranker tries ONNX Runtime, then falls back to the GGUF equivalent only when the ONNX files are not found (`LEMBED_ERROR_MODEL_NOT_FOUND`) (`detail/reranker_impl.hpp`, `lembed_reranker_create`).
- Python: `Reranker(..., backend="auto" | "onnx" | "llama")`.
- In `lembed_text_options_t` the field exists, but `lembed_text_embedding_create()` only checks that the value lies in `[ONNX, AUTO]` (otherwise `LEMBED_ERROR_INVALID_ARGUMENT`) and then ignores it (`detail/text_embedding_onnx_impl.hpp`). There is no `backend=` parameter on `TextEmbedding` or `SparseTextEmbedding`.

## Python mapping

- Every failure status becomes an exception raised by `check_status()` (`python/src/libembedding/_status.py`). All inherit from `LembedError` and carry `status_code`, `message` (text of `lembed_status_message`) and `detail` (text of `lembed_last_error()`).

| C status | Python exception |
|---|---|
| `INVALID_ARGUMENT` | `InvalidArgumentError` |
| `OUT_OF_MEMORY` | `OutOfMemoryError` |
| `ONNX_RUNTIME` | `OnnxRuntimeError` |
| `TOKENIZER` | `TokenizerError` |
| `DOWNLOAD` | `DownloadError` |
| `IO` | `IOError` (class of `libembedding.exceptions`, not the builtin) |
| `MODEL_NOT_FOUND` | `ModelNotFoundError` |
| `UNSUPPORTED` | `UnsupportedError` |
| `BATCH_SIZE` | `BatchSizeError` |
| `LLAMA` | `LlamaError` |
| `CACHE_MISS`, `GGUF` | `LembedError` (no dedicated subclass in `_STATUS_MAP`) |

- `TextEmbedding`, `SparseTextEmbedding`, `ImageEmbedding` and `Reranker` expose `close()` (calls the C `*_free`) and the context-manager protocol (`with ... as m:`). After `close()` the internal handle is `None`.

## Output formats

From the structs in `types.h`:

- Dense: `lembed_embeddings_t` = flat `float` array of `num_embeddings * dim` elements (`float*`, 32-bit).
- Sparse: per text, `indices` as `int32_t*`, `values` as `float*`, `length` elements.
- Reranking: `lembed_rerank_result_t` = `{ int index; float score; }`.

Sort order and normalisation depend on the model and are not guaranteed here: see the documentation of the model concerned.
