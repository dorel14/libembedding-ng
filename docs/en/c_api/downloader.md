---
nav_exclude: true
---

# Model Downloading C API

This module handles model downloading and resolution (ONNX and GGUF) from HuggingFace or local cache.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_ensure_text_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Ensure a text model is cached |
| `lembed_resolve_gguf_path(model_name, out_path)` | `lembed_status_t` | Resolve GGUF model path |
| `lembed_download_model(model_code, dest_dir, progress)` | `lembed_status_t` | Download a model |

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
