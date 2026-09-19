---
nav_exclude: true
---

# Backend Detection

The `backend` module provides automatic backend (ONNX or llama.cpp) detection based on model name or file path.

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

## Detection logic

1. If explicit `backend` → returns directly
2. If `.gguf` file → `"llama"`
3. If `.onnx` file → `"onnx"`
4. Local path → detect from directory contents
5. HuggingFace ID → GGUF if known GGUF model, else ONNX

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
