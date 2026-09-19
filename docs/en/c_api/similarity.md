---
nav_exclude: true
---

# Similarity C API

This module provides native similarity functions operating on raw float arrays.

## Reference

| Function | Returns | Description |
|----------|---------|-------------|
| `lembed_cosine_similarity(a, b, dim)` | `float` | Cosine similarity |
| `lembed_dot_product(a, b, dim)` | `float` | Dot product |
| `lembed_euclidean_distance(a, b, dim)` | `float` | Euclidean (L2) distance |

## Example

```c
#include <libembedding/similarity.h>
#include <stdio.h>

float a[] = {0.1f, 0.2f, 0.3f};
float b[] = {0.4f, 0.5f, 0.6f};
int dim = 3;

float sim = lembed_cosine_similarity(a, b, dim);
float dp  = lembed_dot_product(a, b, dim);
float dist = lembed_euclidean_distance(a, b, dim);

printf("Cosine: %f\n", sim);
printf("Dot product: %f\n", dp);
printf("Euclidean: %f\n", dist);
```

## See also

- [Python API Reference](api_reference.html) — Python API overview
- [Similarity Functions](similarity.html) — Python version
