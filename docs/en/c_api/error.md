---
nav_exclude: true
---

# Error Codes C API

This module defines status codes and thread-local error handling.

## `lembed_status_t`

| Value | Name | Description |
|-------|------|-------------|
| `0` | `LEMBED_OK` | Success |
| `1` | `LEMBED_ERROR_INVALID_ARGUMENT` | Invalid argument |
| `2` | `LEMBED_ERROR_OUT_OF_MEMORY` | Out of memory |
| `3` | `LEMBED_ERROR_ONNX_RUNTIME` | ONNX Runtime error |
| `4` | `LEMBED_ERROR_TOKENIZER` | Tokenizer error |
| `5` | `LEMBED_ERROR_DOWNLOAD` | Download error |
| `6` | `LEMBED_ERROR_IO` | I/O error |
| `7` | `LEMBED_ERROR_MODEL_NOT_FOUND` | Model not found |
| `8` | `LEMBED_ERROR_UNSUPPORTED` | Unsupported feature |
| `9` | `LEMBED_ERROR_BATCH_SIZE` | Invalid batch size |
| `10` | `LEMBED_ERROR_LLAMA` | llama.cpp error |
| `11` | `LEMBED_ERROR_CACHE_MISS` | Cache miss |

> `LEMBED_ERROR_BATCH_SIZE` is kept in the enum but **no code path returns it
> any more**: the guard that produced it was removed, and dynamic quantization
> accepts batching again. It stays in the API so the values keep their order
> (the ABI and the Python `_STATUS_MAP` mapping depend on it).

## Error handling functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_last_error()` | `const char*` | Last error message (thread-local) |
| `lembed_status_message(status)` | `const char*` | Message for the status code |
| `lembed_version()` | `const char*` | Library version |

> The function is named `lembed_status_message()`, **not**
> `lembed_status_to_string()`. Note also that the `LEMBED_ERROR_` prefix of the
> constants does not carry over to the function names.

## Example

```c
#include <libembedding/error.h>

lembed_status_t status = lembed_text_embedding_create(...);
if (status != LEMBED_OK) {
    printf("Error: %s\n", lembed_last_error());
    printf("Code: %d (%s)\n", status, lembed_status_message(status));
}
```

## See also

- [Configuration C API](config.html) — Version and macros
