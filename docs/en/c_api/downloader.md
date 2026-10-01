---
nav_exclude: true
---

# Model Downloading C API

This module handles model downloading and resolution (ONNX and GGUF) from HuggingFace or local cache.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_ensure_text_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Ensure a text model is cached |
| `lembed_ensure_sparse_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Same, for a sparse model |
| `lembed_ensure_image_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Same, for an image model |
| `lembed_ensure_reranker_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Same, for a reranker |
| `lembed_ensure_gguf_model(repo, filename, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Same, for a GGUF model |
| `lembed_resolve_gguf_path(name_or_path, cache_dir, offline, model_path_out)` | `lembed_status_t` | Resolve a `.gguf` path (file, URL or registry name) |
| `lembed_cleanup_model_cache(cache_dir, keep_models, dry_run, deleted_count, freed_bytes)` | `lembed_status_t` | Purge the cache, keeping a list of models |
| `lembed_cleanup_model_cache_except(cache_dir, active_model_dir, dry_run, deleted_count, freed_bytes)` | `lembed_status_t` | Purge the cache except the active model |
| `lembed_free_string(s)` | `void` | Free a string allocated by the library |

> There is **no** `lembed_download_model()`: downloading always goes through an
> `lembed_ensure_*` function. The `ensure` prefix means "download if absent,
> otherwise a no-op".
>
> `lembed_resolve_gguf_path()` is declared in this header and implemented inside
> the `LIBEMBEDDING_IMPLEMENTATION` block. It **is** exported by
> `libembedding.def` (line 156), so it works both against the Windows DLL and in
> a header-only build.

## `lembed_ensure_text_model()`

Ensures a text model is downloaded and available in cache.

| Parameter | Description |
|-----------|-------------|
| `model_idx` | Text model index (`lembed_text_model_t`) |
| `cache_dir` | Cache directory (NULL = default) |
| `progress` | Show progress bar (0/1) |
| `offline` | Offline mode (1) or download (0) |
| `out_path` | Resolved model path |

## See also

- [Models](models.html) — Available models list
- [GGUF Registry C API](gguf_registry.html) — C GGUF registry
