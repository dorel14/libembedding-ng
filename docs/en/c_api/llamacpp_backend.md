---
nav_exclude: true
---

# llama.cpp Backend C API

This module provides interaction with the llama.cpp backend for GGUF models.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_llama_backend_available()` | `int` (1=yes, 0=no) | Check availability |
| `lembed_llama_version()` | `const char*` | Backend identifier |

> `lembed_llama_version()` does not return a version number: the current
> implementation returns the literal string `"llama.cpp enabled"`. Do not use it
> to detect a version.
>
> There is **no** `lembed_llama_set_logging()` and no
> `lembed_llama_get_n_gpu_layers()` in this API. GPU layers are set through the
> model options (`LEMBED_BACKEND_LLAMACPP` and `n_gpu_layers`), and logging
> through llama.cpp's own configuration.

## Example

```c
#include <libembedding/llamacpp_backend.h>

if (lembed_llama_backend_available()) {
    printf("llama.cpp: %s\n", lembed_llama_version());
} else {
    printf("llama.cpp backend not available\n");
}
```

## See also

- [Autotuner C API](autotuner.html) — Autotuning with llama.cpp
- [Worker Auto-Tune C API](worker_autotune.html) — Worker configuration
