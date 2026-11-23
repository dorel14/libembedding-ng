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
| `lembed_maxsim(q, nq, d, nd, dim, out_score)` | `lembed_status_t` | MaxSim (ColBERT) score of a query against one document |
| `lembed_maxsim_batch(q, nq, docs, offsets, num_docs, dim, out_scores)` | `lembed_status_t` | One MaxSim score per document, in input order |

## MaxSim (late interaction)

`lembed_maxsim` returns the sum, over the `nq` query vectors, of the maximum dot product with the `nd` document vectors. Matrices are row-major (`dim` floats per row); L2-normalised inputs give the cosine MaxSim. `nq == 0` or `nd == 0` gives a score of 0. No state, re-entrant. Portable C kernel (no intrinsics) that compilers auto-vectorise: one temporary copy of the query is allocated and freed within the call (a scalar loop with the same results is used if that allocation fails). Deterministic accumulation order.

`lembed_maxsim_batch` scores one query against several concatenated documents: document `i` is rows `offsets[i]` to `offsets[i+1]-1` of `docs` (`offsets` has `num_docs + 1` entries, `offsets[0] == 0`, non-decreasing). No sorting, no top-K, no index. Returns `LEMBED_ERROR_INVALID_ARGUMENT` for a NULL output pointer, NULL `q`/`d` with a non-zero count, a negative count, `dim <= 0` or invalid offsets.

```c
float score;
lembed_status_t st = lembed_maxsim(q, nq, d, nd, dim, &score);
```

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
