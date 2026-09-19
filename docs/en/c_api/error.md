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
| `3` | `LEMBED_ERROR_RUNTIME` | Runtime error (ONNX Runtime) |
| `4` | `LEMBED_ERROR_TOKENIZER` | Tokenizer error |
| `5` | `LEMBED_ERROR_DOWNLOAD` | Download error |
| `6` | `LEMBED_ERROR_IO` | I/O error |
| `7` | `LEMBED_ERROR_MODEL_NOT_FOUND` | Model not found |
| `8` | `LEMBED_ERROR_UNSUPPORTED` | Unsupported feature |
| `9` | `LEMBED_ERROR_BATCH_SIZE` | Invalid batch size |
| `10` | `LEMBED_ERROR_LLAMA` | llama.cpp error |

## Error handling functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_last_error()` | `const char*` | Last error message (thread-local) |
| `lembed_status_to_string(status)` | `const char*` | Status code name |

## Example

```c
#include <libembedding/error.h>

lembed_status_t status = lembed_text_embedding_create(...);
if (status != LEMBED_OK) {
    printf("Error: %s\n", lembed_last_error());
    printf("Code: %s\n", lembed_status_to_string(status));
}
```

## See also

- [Configuration C API](config.html) — Version and macros
