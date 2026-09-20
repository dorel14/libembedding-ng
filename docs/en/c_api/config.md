---
nav_exclude: true
---

# Configuration and Version C API

This module manages library version and configuration macros.

## Version

| Macro | Description |
|-------|-------------|
| `LIBEMBEDDING_VERSION_MAJOR` | Major version |
| `LIBEMBEDDING_VERSION_MINOR` | Minor version |
| `LIBEMBEDDING_VERSION_PATCH` | Patch version |
| `LIBEMBEDDING_VERSION_STRING` | Full version (e.g., "1.6.0") |

## Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_version()` | `const char*` | Library version |

## Configuration macros

| Macro | Description |
|-------|-------------|
| `LIBEMBEDDING_IMPLEMENTATION` | Enable header-only implementation (Linux/macOS) |
| `LIBEMBEDDING_NO_DOWNLOAD` | Disable model downloading |
| `LIBEMBEDDING_NO_IMAGE` | Disable images (stb_image) |
| `LIBEMBEDDING_INTEGRATION_TESTS` | Enable integration tests |
| `LIBEMBEDDING_BUILD_SHARED` | Build shared library for Python bindings (Windows: always SHARED) |

## Version

| Macro | Description |
|-------|-------------|
| `LIBEMBEDDING_VERSION_MAJOR` | Major version |
| `LIBEMBEDDING_VERSION_MINOR` | Minor version |
| `LIBEMBEDDING_VERSION_PATCH` | Patch version |
| `LIBEMBEDDING_VERSION_STRING` | Full version (e.g., "1.6.0") |

## Example

```c
#include <libembedding/config.h>
#include <stdio.h>

printf("libembedding v%s\n", LIBEMBEDDING_VERSION_STRING);
printf("Version: %d.%d.%d\n",
    LIBEMBEDDING_VERSION_MAJOR,
    LIBEMBEDDING_VERSION_MINOR,
    LIBEMBEDDING_VERSION_PATCH);
```

## See also

- [Error Codes C API](error.html) — Error handling
