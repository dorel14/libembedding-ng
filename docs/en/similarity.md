---
nav_exclude: true
---

# Similarity Functions

Native functions for comparing two embedding vectors without instantiation overhead. Operates directly on 1-D `numpy` arrays of `float32`.

## Reference

| Function | Description |
|----------|-------------|
| `cosine_similarity(a, b)` | Cosine similarity between two vectors |
| `dot_product(a, b)` | Dot product between two vectors |
| `euclidean_distance(a, b)` | L2 distance between two vectors |

## Usage

```python
import numpy as np
from libembedding import cosine_similarity, dot_product, euclidean_distance

a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
b = np.array([0.4, 0.5, 0.6], dtype=np.float32)

# Cosine similarity (0.0 = orthogonal, 1.0 = identical)
sim = cosine_similarity(a, b)

# Dot product
dp = dot_product(a, b)

# Euclidean distance
dist = euclidean_distance(a, b)
```

## See also

- [Python API Reference](api_reference.html) — Python API overview
- [Runtime Statistics](python/stats.html) — `Stats` type used by `.stats()`
