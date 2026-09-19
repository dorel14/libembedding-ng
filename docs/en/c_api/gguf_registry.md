---
nav_exclude: true
---

# GGUF Registry C API

This module manages the list of recommended GGUF models and their configuration.

## Types

| Type | Description |
|------|-------------|
| `lembed_gguf_model_info_t` | GGUF model information |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_list_gguf_models(out, count)` | `lembed_status_t` | List GGUF models |
| `lembed_find_gguf_model_by_name(name)` | `int` | Find GGUF model by name |
| `lembed_get_gguf_model_info(model_idx, out)` | `lembed_status_t` | Get GGUF model info |
| `lembed_validate_gguf_path(path)` | `lembed_status_t` | Validate GGUF path |
| `lembed_resolve_gguf_path(model_name, out_path)` | `lembed_status_t` | Resolve GGUF path |

## Example

```c
#include <libembedding/gguf_registry.h>

const lembed_model_info_t **models;
int count;
lembed_status_t status = lembed_list_gguf_models(&models, &count);
if (status == LEMBED_OK) {
    printf("GGUF models available: %d\n", count);
    for (int i = 0; i < count; i++) {
        printf("  %s (%s)\n",
            models[i]->model_name,
            models[i]->model_code);
    }
}
```

## See also

- [Models](models.html) — Models list (Python)
- [Model Downloading C API](downloader.html) — Model downloading
