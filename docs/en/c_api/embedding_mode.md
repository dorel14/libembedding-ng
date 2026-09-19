---
nav_exclude: true
---

# Embedding Modes C API

This module defines quality/speed modes and their recommended models.

## `lembed_embedding_mode_t` enum

| Value | Name | Description |
|-------|------|-------------|
| `0` | `LEMBED_MODE_FAST` | Fast mode (quantized, small model) |
| `1` | `LEMBED_MODE_BALANCED` | Quality/speed tradeoff |
| `2` | `LEMBED_MODE_QUALITY` | Maximum quality (FP32, large model) |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_mode_to_string(mode)` | `const char*` | Mode name as string |
| `lembed_recommended_model_for_mode(mode)` | `lembed_text_model_t` | Recommended model for mode |

## Mode/model mapping

| Mode | Recommended model |
|------|-------------------|
| FAST | Paraphrase-MiniLM-L12-v2 |
| BALANCED | BAAI/bge-small-en-v1.5 |
| QUALITY | BAAI/bge-base-en-v1.5 |

## Example

```c
#include <libembedding/embedding_mode.h>

const char *name = lembed_mode_to_string(LEMBED_MODE_BALANCED);
printf("Mode: %s\n", name);  // "balanced"

lembed_text_model_t model = lembed_recommended_model_for_mode(LEMBED_MODE_QUALITY);
// LEMBED_TEXT_BGE_BASE_EN_V15
```

## See also

- [Python API Reference](api_reference.html) — Python API overview
- [Getting Started](getting_started.html) — Python mode usage
