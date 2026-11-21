---
nav_exclude: true
---

# Model routing

> **Français:** [Routage des modèles](../model_routing.html)

How the library chooses the backend (ONNX Runtime or llama.cpp / GGUF) for a given model, and how to check what is actually running.

## Principle

**The backend follows the model format**, not a parameter (`include/libembedding/text_embedding.h`, doc of `lembed_text_embedding_backend_name`):

- a `.gguf` file is served by the GGUF backend;
- an ONNX file or directory, a registry model (HuggingFace identifier) or any other name goes through ONNX Runtime.

There is **no `backend=` parameter** on `TextEmbedding` or `SparseTextEmbedding` in Python. Two surface exceptions, described below: the reranker (`backend=`) and the helper function `detect_backend()`.

## Rules per entry point

### Dense text

| Entry | Rule |
|---|---|
| Python `TextEmbedding(name)` | Name ending in `.gguf` (case-insensitive): llama.cpp, either an existing local file or `<repo>/<file>.gguf` (downloaded). Otherwise: local path (existing, or ending in `.onnx` / `.json`) or registry model: ONNX Runtime. An unknown name raises `ModelNotFoundError` (`python/src/libembedding/text_embedding.py`, `models.py`). |
| C `lembed_text_embedding_create()` | Registry model (enum): ONNX Runtime. |
| C `lembed_text_embedding_create_from_path()` | Path ending in `.gguf` (exactly lowercase): llama.cpp, through `lembed_text_embedding_create_from_gguf_path()`. Otherwise an ONNX directory holding `model.onnx` and `tokenizer.json` (`config.json` optional) (`detail/text_embedding_impl.hpp`). |
| C `..._create_from_gguf_path()` / `..._create_from_gguf_model()` | llama.cpp, explicitly. |

### Sparse

`lembed_sparse_text_embedding_create_from_path()` sends a path ending in `.gguf` (case-insensitive) to the **in-house ggml runtime**, not to llama.cpp, whose loader cannot read these files (`sparse_text_embedding.h`). Everything else goes through ONNX Runtime. See [Sparse GGUF runtime](gguf_sparse_runtime.html).

### Reranker

The reranker is the only one that exposes a selector: the `backend` field of `lembed_reranker_options_t` (`LEMBED_BACKEND_ONNX`, `LEMBED_BACKEND_LLAMACPP`, `LEMBED_BACKEND_AUTO`, default `AUTO` in `lembed_reranker_options_default()`) and the `backend="auto"` argument of the Python `Reranker` (`"auto"`, `"onnx"`, `"llama"`).

- `LLAMACPP`: always GGUF.
- `ONNX`: always ONNX Runtime.
- `AUTO`: ONNX Runtime first; **only if the model's ONNX files cannot be found** (`LEMBED_ERROR_MODEL_NOT_FOUND`) does the reranker switch to the GGUF counterpart. An unavailable provider (`LEMBED_ERROR_UNSUPPORTED`) or a broken ONNX model stays an error: it does not trigger a switch (`detail/reranker_impl.hpp`, `lembed_reranker_create`).
- `lembed_reranker_create_from_path()` sends a path ending in `.gguf` (case-insensitive) to GGUF, and also accepts an ONNX directory (`reranker.h`).

`backend` is an official option of the reranker **only**. The field also exists in `lembed_text_options_t`, but `lembed_text_embedding_create()` only checks that the value is valid and then **ignores it** (`detail/text_embedding_onnx_impl.hpp`). For dense text, choose the format through the entry point as above. See [API conventions](api_conventions.html).

### `detect_backend()` (Python)

`libembedding.detect_backend(model_name, backend="auto")` (`python/src/libembedding/backend.py`) is a standalone helper: it returns `"onnx"` or `"llama"` from the extension (`.gguf`, `.onnx`), from the contents of a local directory, or `"onnx"` by default. The Python constructors do not call it, so it is not what decides the backend of a `TextEmbedding`. Details: [Backend auto-detection](python/backend.html).

## Execution provider and backend

The provider (`provider`) is independent from the backend: `"llama"` / `"llamacpp"` (value 5) is **not** an ONNX Runtime provider. Passing it with an ONNX model fails with `LEMBED_ERROR_UNSUPPORTED` and the message `it is not an ONNX Runtime execution provider; load a .gguf model to use the llama.cpp backend` (`detail/onnx_session_impl.hpp`). See [Troubleshooting](troubleshooting.html).

## Checking the backend in use

### C

| Function | Values |
|---|---|
| `lembed_text_embedding_backend_name(ctx)` | `"onnxruntime"` or `"llama.cpp"` |
| `lembed_sparse_text_embedding_backend_name(ctx)` | `"onnxruntime"` or `"ggml-sparse"` |
| `lembed_reranker_backend_name(ctx)` | `"onnxruntime"` or `"llama.cpp"` |
| `lembed_image_embedding_backend_name(ctx)` | always `"onnxruntime"` |
| `lembed_*_effective_provider(ctx)` | ONNX Runtime: `"CPU"`, `"CUDA"`, `"CoreML"`, `"DirectML"` or `"TensorRT"`; llama.cpp: `"CPU"`, or `"GPU"` when `llama_n_gpu_layers != 0` and ggml has a GPU device (the shipped builds have none); `ggml-sparse`: always `"CPU"` |

These functions return a static string that must not be freed, or `NULL` if `ctx` is `NULL`. A non-CPU provider that cannot be attached makes creation fail, so `effective_provider` never reports a provider that is not in use.

### Python

```python
from libembedding import TextEmbedding

emb = TextEmbedding("BAAI/bge-small-en-v1.5")
print(emb.backend)   # "onnxruntime" or "llama.cpp"
print(emb.provider)  # "cpu", "cuda"... (lowercase)
info = emb.info()    # descriptor: includes backend and effective_provider
```

`TextEmbedding`, `SparseTextEmbedding`, `ImageEmbedding` and `Reranker` expose the `backend` and `provider` properties. For llama.cpp, `provider` is `"cpu"`, or `"gpu"` when layers are offloaded to a ggml GPU device (`text_embedding.py`).

### Log

With `LIBEMBEDDING_VERBOSE=1` (read once at process start), the library logs to stderr, including the reranker's ONNX-to-GGUF switch in `AUTO` mode (`reranker.h`, `detail/log.hpp`).
