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

## HTTPS certificates

Certificates are resolved by `find_ca_source()`
(`include/libembedding/detail/downloader_impl.hpp`):

- **Linux, Android, iOS**: if `SSL_CERT_FILE` or `CURL_CA_BUNDLE` names an
  existing file, it is used (`CURLOPT_CAINFO`). These variables are not read on
  Windows or macOS, which keep libcurl's default.
- **Linux, Android**: otherwise, the first existing bundle among
  `/etc/ssl/certs/ca-certificates.crt`, `/etc/pki/tls/certs/ca-bundle.crt`,
  `/etc/ssl/ca-bundle.pem`, `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`,
  `/etc/ssl/cert.pem`; failing that, the Android directory
  `/system/etc/security/cacerts` (`CURLOPT_CAPATH`).
- **iOS**: with no variable set, Mozilla's CA bundle compiled into the library
  (CMake option `LIBEMBEDDING_CA_BUNDLE_EMBED`, macro
  `LIBEMBEDDING_EMBEDDED_CA_BUNDLE`) is handed to libcurl through
  `CURLOPT_CAINFO_BLOB`.

The default cache directory is `~/.cache/libembedding`, or
`$HOME/Library/Caches/libembedding` on iOS; `LIBEMBEDDING_CACHE_DIR` and
`FASTEMBED_CACHE_DIR` take precedence.

## Size limit on mobile

On Android and iOS, a registry model larger than `LEMBED_MOBILE_MODEL_LIMIT_MB`
(150 MB) is refused before any download: the `lembed_ensure_*` functions (text,
sparse, image, reranker) and `lembed_ensure_gguf_model` return
`LEMBED_ERROR_UNSUPPORTED`, and `lembed_last_error()` gives the size, the limit
and the light models of the same kind. A local path is never checked.
`lembed_mobile_model_limit_mb()` returns the active limit (0 = none) and
`lembed_model_download_size_mb()` a model's size (-1 if unknown); both are
declared in `model_registry.h`. `LIBEMBEDDING_ALLOW_LARGE_MODELS=1` lifts the
limit, `LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT=1` applies it on a desktop.
Details: [Platforms](../platforms.html).

## See also

- [Models](models.html) — Available models list
- [GGUF Registry C API](gguf_registry.html) — C GGUF registry
