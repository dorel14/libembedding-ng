---
nav_exclude: true
---

# Backend Detection

The `backend` module provides `detect_backend()`, a standalone helper that tells which backend (ONNX or llama.cpp) matches a model name or path. The constructors (`TextEmbedding`, `Reranker`, etc.) do not call it: actual routing happens in the C layer, by file extension (`.gguf`, case-insensitive, goes to llama.cpp) or by the entry point used; everything else (`.onnx`, HuggingFace ID, unknown) goes to ONNX Runtime.

> `backend=` is a constructor option of `Reranker` and `TextEmbedding` (`"auto"`, `"onnx"`, `"llama"`, default `"auto"`). `SparseTextEmbedding` and `ImageEmbedding` have no such parameter: the backend follows the model extension. See [Model routing](../model_routing.html).

## Reference

| Function | Returns | Description |
|----------|---------|-------------|
| `detect_backend(model_name, backend="auto")` | `str` | Detect which backend to use |
| `backend_to_enum(backend)` | `int` | Convert backend string to C enum value |

## `detect_backend()`

Detects which backend to use based on model and user preference.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Model name, path, or HuggingFace ID |
| `backend` | `"auto"` | `"auto"`, `"onnx"` or `"llama"` |

| Return | Description |
|--------|-------------|
| `"onnx"` | ONNX backend |
| `"llama"` | llama.cpp backend |

## `detect_backend()` logic

This logic belongs to the helper only; it does not describe how the constructors route.

1. If explicit `backend` → returns directly
2. If `.gguf` file → `"llama"`
3. If `.onnx` file → `"onnx"`
4. Local path → detect from directory contents
5. Otherwise (HuggingFace ID or unknown name) → `"onnx"`; no "known GGUF model" lookup is done

## Usage

```python
from libembedding import detect_backend

# Auto-detection
backend = detect_backend("BAAI/bge-small-en-v1.5")
print(backend)  # "onnx"

# Force llama.cpp
backend = detect_backend("meta-llama/Llama-3-8B", backend="llama")
print(backend)  # "llama"

# Detection by local file
backend = detect_backend("/path/to/model.gguf")
print(backend)  # "llama"
```

## See also

- [Python API Reference](api_reference.html) — Python API overview
- [Embedding Modes](c_api/embedding_mode.html) — C API model selection
