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
| `lembed_find_gguf_model(name)` | `const lembed_gguf_model_info_t*` | Look up a GGUF model (NULL if unknown) |
| `lembed_default_gguf_model()` | `const lembed_gguf_model_info_t*` | Recommended default GGUF model |

> Both lookup functions return a **pointer into static data** — do not free it.
> There is no `lembed_get_gguf_model_info()`: iterate directly over the array
> returned by `lembed_list_gguf_models()`. `lembed_resolve_gguf_path()` is
> declared in `downloader.h`, not here.

## `lembed_gguf_model_info_t` fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `const char*` | Human-readable name (e.g. `"Snowflake-XS-Q4"`) |
| `gguf_url` | `const char*` | Download URL for the `.gguf` |
| `model_code` | `const char*` | Original HuggingFace code |
| `description` | `const char*` | Short description |
| `dim` | `int` | Embedding dimension |
| `params_m` | `int` | Parameter count in millions |
| `file_size_mb` | `int` | Approximate file size in MB |
| `quality_mteb` | `float` | MTEB Retrieval score (NDCG@10), 0 if unknown |
| `recommended_sessions` | `int` | Recommended session count |

> The field is named `name`, **not** `model_name`.

## Example

```c
#include <libembedding/gguf_registry.h>

const lembed_gguf_model_info_t **models;
int count;
lembed_status_t status = lembed_list_gguf_models(&models, &count);
if (status == LEMBED_OK) {
    printf("GGUF models available: %d\n", count);
    for (int i = 0; i < count; i++) {
        printf("  %s (%s, %d dim)\n",
            models[i]->name,
            models[i]->model_code,
            models[i]->dim);
    }
}
```

## See also

- [Models](models.html) — Models list (Python)
- [Model Downloading C API](downloader.html) — Model downloading
