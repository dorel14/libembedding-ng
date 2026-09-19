---
nav_exclude: true
---

# llama.cpp Backend C API

This module provides interaction with the llama.cpp backend for GGUF models.

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_llama_backend_available()` | `int` (1=yes, 0=no) | Check availability |
| `lembed_llama_version()` | `const char*` | llama.cpp version |
| `lembed_llama_set_logging(enable)` | `void` | Enable/disable logging |
| `lembed_llama_get_n_gpu_layers(model_name)` | `int` | Number of GPU layers |

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
