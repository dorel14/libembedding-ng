---
nav_exclude: true
---

# Troubleshooting

> **Français:** [Dépannage](../troubleshooting.html)

Common symptoms, cause, fix. Every entry comes from the code; where the code does not allow saying more, the page stops there.

## Reading the error

### C

Every failing function returns a `lembed_status_t` (`include/libembedding/error.h`). Two functions give the detail:

- `lembed_status_message(status)`: fixed label for the code (for example `"Unsupported operation"`).
- `lembed_last_error()`: **thread-local** detail set by the last failing call (1024-byte buffer). Read it on the same thread, right after the failure.

```c
lembed_status_t s = lembed_text_embedding_create(&opts, &ctx);
if (s != LEMBED_OK) {
    fprintf(stderr, "%s: %s\n", lembed_status_message(s), lembed_last_error());
}
```

Codes: `LEMBED_ERROR_INVALID_ARGUMENT`, `OUT_OF_MEMORY`, `ONNX_RUNTIME`, `TOKENIZER`, `DOWNLOAD`, `IO`, `MODEL_NOT_FOUND`, `UNSUPPORTED`, `BATCH_SIZE` (reserved, never returned), `LLAMA`, `CACHE_MISS`, `GGUF`.

### Python

`check_status()` raises a subclass of `LembedError` carrying `status_code`, `message` and `detail` (`python/src/libembedding/_status.py`, `exceptions.py`). The text is `"<message>: <detail>"`.

| C code | Python exception |
|---|---|
| `INVALID_ARGUMENT` | `InvalidArgumentError` |
| `OUT_OF_MEMORY` | `OutOfMemoryError` |
| `ONNX_RUNTIME` | `OnnxRuntimeError` |
| `TOKENIZER` | `TokenizerError` |
| `DOWNLOAD` | `DownloadError` |
| `IO` | `IOError` (the package's own class, not the builtin) |
| `MODEL_NOT_FOUND` | `ModelNotFoundError` |
| `UNSUPPORTED` | `UnsupportedError` |
| `BATCH_SIZE` | `BatchSizeError` |
| `LLAMA` | `LlamaError` |
| `CACHE_MISS`, `GGUF` | `LembedError` (no dedicated subclass in `_STATUS_MAP`); read `status_code` |

```python
from libembedding import LembedError
try:
    ...
except LembedError as e:
    print(e.status_code, e.message, e.detail)
```

To see what the library logs, set `LIBEMBEDDING_VERBOSE=1` (truthy values: `1`, `true`, `yes`, `on`) **before** starting the program: the variable is read once per process and the lines go to stderr (`detail/log.hpp`).

## Windows

### `ctest` reports "Timeout", or the program dies at startup with `0xC0000135`

- **Cause**: `STATUS_DLL_NOT_FOUND`. A runtime DLL is missing next to the executable; `ctest` reports the dead process as a timeout.
- **Fix**: run the executable directly and read its exit code (`$LASTEXITCODE` in PowerShell) to confirm. The runtime DLLs (ONNX Runtime, libcurl, libembedding) are copied next to executables by `copy_runtime_dlls()`; check that they are there (`AGENTS.md` section 3.5).

### `ONNX Runtime API version N is not supported by the loaded onnxruntime library (version X)`

- **Cause**: the library was built against newer ONNX Runtime headers than the `onnxruntime.dll` that was actually loaded. The code comment names the case of a `System32` copy shadowing ours (`detail/onnx_session_impl.hpp`, `ort_api()`).
- **Fix**: make the process load the `onnxruntime.dll` shipped with the library (the one next to `libembedding.dll`) instead of the system one. The message gives both versions (`N` = expected, `X` = loaded).
- In Python, `_binding.py` preloads the dependent DLLs (including `onnxruntime.dll`) from the library's directories and registers those directories for DLL resolution. If the error still appears, check which library is loaded (`LIBEMBEDDING_LIBRARY`, below).

### CMake stops with `libcurl runtime DLL not found`

- **Cause**: on Windows, `libembedding.dll` links against libcurl and the DLL (`libcurl-x64.dll`) is never committed to the repository.
- **Fix** (message from `CMakeLists.txt`): one of
  - `pwsh -File scripts/fetch_windows_libcurl.ps1` (installs the runtime in `third_party/curl/bin/`);
  - `-DCURL_LIBRARY=<vcpkg>\installed\x64-windows\lib\libcurl.lib`;
  - `-DCURL_RUNTIME_DLL=<path>\libcurl-x64.dll`;
  - `-DLIBEMBEDDING_NO_DOWNLOAD=ON` (no download code at all);
  - `-DLIBEMBEDDING_CURL_STATIC=ON` (runtime managed by the caller).

## Python: which native library is loaded?

- `LIBEMBEDDING_LIBRARY=<path>` forces the file to load; if the path does not exist you get `OSError: LIBEMBEDDING_LIBRARY points to a missing file: ...` (`_binding.py`).
- In a source checkout holding several builds, a warning `Loading <lib>, which is older than the build output <lib2>` signals that a newer copy exists; set `LIBEMBEDDING_LIBRARY` or remove the stale copy.
- On macOS, `_binding.py` prepends the library's directories to `DYLD_LIBRARY_PATH` before `dlopen` (`LD_LIBRARY_PATH` on Linux). If loading `libembedding.dylib` fails, start by setting `LIBEMBEDDING_LIBRARY` to the intended file to isolate the problem.

## Models and network

### `Model file not in cache (offline mode): <path>`

- **Cause**: `offline=True` (Python) / `offline = 1` (C) forbids any download and the file is not in the cache. The status is `LEMBED_ERROR_DOWNLOAD` (`DownloadError` in Python) (`detail/downloader_impl.hpp`, `downloader.h`).
- **Fix**: load the model once with network access, or point to the right cache. The cache directory is resolved in this order: explicit `cache_dir` argument, then `LIBEMBEDDING_CACHE_DIR`, then `FASTEMBED_CACHE_DIR`, then `~/.cache/libembedding` (`detail/downloader_impl.hpp`, `get_cache_dir`). A build with `-DLIBEMBEDDING_NO_DOWNLOAD=ON` can only read the existing cache.
- `LEMBED_ERROR_CACHE_MISS` is unrelated: it concerns the **autotune cache** (`autotune_cache.h`).

### `LEMBED_ERROR_UNSUPPORTED` with a licence message

- **Cause**: some registry models are under a licence that is not MIT-compatible (`jinaai/jina-reranker-v2-base-multilingual`, `onnx-community/embeddinggemma-300m-ONNX`); loading them is refused by default (`detail/download_manager.hpp`).
- **Fix**: the message says so. Set `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1` once you have read the licence, or pass a local model path.

### `ModelNotFoundError`: `Unknown model: <name>` (Python)

- **Cause**: the name is neither a local path, nor a `.gguf` file, nor a registry model (`text_embedding.py`, ONNX branch). On the Python side an unknown identifier is rejected before any loading.
- **Fix**: check the name with `list_text_models()`; see [Model routing](model_routing.html).

## Execution provider (GPU, etc.)

### `Execution provider X was requested but is not available: ...` (`LEMBED_ERROR_UNSUPPORTED`)

- **Cause**: a non-CPU provider was requested that this build cannot attach (`detail/onnx_session_impl.hpp`, `provider_never_available`):
  - CUDA: build without `USE_CUDA`;
  - CoreML: Apple platforms only;
  - DirectML: Windows with `USE_DML` only;
  - TensorRT: build without `USE_TENSORRT`;
  - `llama` / `llamacpp` (value 5): not an ONNX Runtime provider; for the llama.cpp backend, load a `.gguf` file;
  - any other value: `unknown lembed_execution_provider_t value`.
- **Fix**: use `LEMBED_PROVIDER_CPU` (`provider="cpu"`), as the message says. The check runs before any file is read and does not throw.
- A provider that is compiled in but fails to attach at runtime raises an error at load time (`apply_provider_config()`). To know what actually runs, see [Model routing](model_routing.html).

## GGUF refused at load

Sparse GGUF goes through the in-house ggml runtime, which refuses a file with a precise reason (`LEMBED_ERROR_UNSUPPORTED`, detail in `lembed_last_error()`; `UnsupportedError` in Python). Messages (`detail/gguf/gguf_weights.hpp`):

| Message | Cause | Fix |
|---|---|---|
| `missing tensor '<name>'` | A required tensor is absent from the file (see the [sparse GGUF convention](gguf_sparse_convention.html)) | Use a GGUF that follows the convention |
| `tensor '<name>' has type <t>; this runtime supports f32, f16 and q8_0` | Untested tensor type, refused by name | Reconvert to F32, F16 or Q8_0 |
| `tensor '<name>' is a vector of type <t>; vectors must be f32` | Quantized biases / LayerNorm | Keep 1-D tensors in F32 |
| `tensor '<name>' has shape [...], expected [...]` | Shape incompatible with the graph | Wrong file for this model |
| `token_type_embd.weight is not F32` | Quantized token-type table | Keep it in F32 |

Details on types: [Quantization](quantization.html). An exception thrown during loading is reported as `LEMBED_ERROR_GGUF`.

Another case: `lembed_sparse_text_embedding_set_max_batch_tokens()` called on an ONNX context returns `LEMBED_ERROR_UNSUPPORTED` with `max_batch_tokens applies to the GGUF runtime only` (`detail/sparse_text_embedding_gguf_impl.hpp`).

In Python, a model name ending in `.gguf` that is not an existing file and contains no `/` raises `FileNotFoundError: GGUF model not found: '<name>'`; with a `/` it is read as `<repo>/<file>` to download (`text_embedding.py`).
