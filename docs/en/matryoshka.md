---
nav_exclude: true
---

# Matryoshka truncation

A model trained with *Matryoshka Representation Learning* (MRL) packs the most
useful information into the first components of its vectors. Keeping only the
first `k` components and renormalising gives a smaller vector that still works
for retrieval: less memory, less disk, faster dot products in your index.

libembedding truncates **after** inference: the model runs at its native
dimension, then each vector is cut to `k` components and L2-renormalised. The
encoder cost does not change.

> ⚠️ On a model that was **not** trained with Matryoshka, the prefix of a vector
> carries no quality guarantee. The library does not forbid it (it cannot know
> how a local model was trained), but results may collapse.

## Matryoshka-trained models in the registry

| Model | Native dimension | Dimensions documented by the authors |
|-------|------------------|---------------------------------------|
| `nomic-ai/nomic-embed-text-v1.5` (FP32 and INT8) | 768 | 768, 512, 256, 128, 64 |
| `mixedbread-ai/mxbai-embed-large-v1` (FP32 and INT8) | 1024 | 1024 down to 64 (any value; 512 recommended) |
| `google/embeddinggemma-300m` | 768 | 768, 512, 256, 128 |

The other registry models (including `nomic-ai/nomic-embed-text-v1`, the v1
without MRL) are not Matryoshka-trained.

## Python

```python
from libembedding import TextEmbedding

model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5", truncate_dim=256)
model.dim          # 256  (size of the returned vectors)
model.native_dim   # 768  (size produced by the model)
model.truncate_dim # 256

vecs = model.embed(["search_query: what is Matryoshka?"])
vecs.shape         # (1, 256), every row has unit norm
```

- `truncate_dim` must lie between `1` and the native dimension. `0`, a negative
  value or a value above the native dimension raise `ValueError` at
  construction; a non-`int` raises `TypeError`.
- `None` (default) keeps the whole vector.
- Vectors are **always renormalised**: cosine and dot product stay equivalent,
  as for a full vector.
- `embed()`, `embed_iter()`, `embed_bytes()`, `embed_stream()` and
  `embed_batched()` all apply the truncation. `TextEmbeddingPool` takes the same
  parameter and forwards it to every worker.

The output matches the numpy reference
`v[:, :k] / np.linalg.norm(v[:, :k], axis=1, keepdims=True)` to within 1e-6
(`python/tests/test_matryoshka.py`).

## C API

```c
#include <libembedding/libembedding.h>

lembed_status_t lembed_embeddings_truncate(lembed_embeddings_t* emb,
                                           int new_dim,
                                           int renormalize);
```

| Parameter | Description |
|-----------|-------------|
| `emb` | Dense result (`lembed_text_embedding_embed`, `lembed_image_embedding_embed`, or a caller-owned buffer). Modified in place. |
| `new_dim` | New dimension, `1 <= new_dim <= emb->dim`. |
| `renormalize` | `1` = L2-renormalise every truncated row (an all-zero row stays zero); `0` = plain prefix. |

Behaviour:

- Rows are compacted in place (`emb->data` becomes a
  `[num_embeddings * new_dim]` array) and `emb->dim` is updated.
- The buffer is **not reallocated**: `emb->data` keeps its address and
  `lembed_embeddings_free()` still releases it. The function neither frees nor
  allocates, so it also accepts a caller-owned buffer.
- `new_dim == emb->dim` is valid: nothing moves, renormalisation is applied if
  requested.
- Errors: NULL `emb`, `emb->dim <= 0`, `emb->num_embeddings < 0`, NULL `data`
  with rows, or `new_dim` outside `[1, emb->dim]` →
  `LEMBED_ERROR_INVALID_ARGUMENT`, details in `lembed_last_error()`. On error,
  `emb` is left untouched.

```c
lembed_embeddings_t emb = {0};
if (lembed_text_embedding_embed(ctx, texts, n, 0, &emb) == LEMBED_OK) {
    if (lembed_embeddings_truncate(&emb, 256, 1) != LEMBED_OK)
        fprintf(stderr, "%s\n", lembed_last_error());
    /* emb.dim == 256 */
    lembed_embeddings_free(&emb);
}
```

## Quality per dimension (nDCG@10, BEIR SciFact)

nDCG@10 on BEIR SciFact (test split, 5,183 documents, 300 queries, `max_length` 512, ONNX fp32). In parentheses, the share of the native-dimension quality that is kept.

| Model | 1024 | 768 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|---|
| `nomic-ai/nomic-embed-text-v1.5` | — | 0.7032 (native) | 0.7007 (99.6 %) | 0.6825 (97.1 %) | 0.6425 (91.4 %) | 0.5274 (75.0 %) |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.7389 (native) | 0.7316 (99.0 %) | 0.7277 (98.5 %) | 0.6932 (93.8 %) | 0.6714 (90.9 %) | 0.5985 (81.0 %) |

In practice, 512 dimensions keep more than 98 % of the quality with 1.5 to 2 times less memory, and 256 dimensions keep more than 93 %. Below 128, the loss becomes significant.

Measured on 2026-11-24 on an Intel Core i7-1065G7 (4 cores, Windows 11), ONNX Runtime 1.31.0, `dev` commit `071b1ce`. To reproduce:

```bash
python benchmarks/matryoshka/bench_matryoshka_ndcg.py \
  --beir ~/.cache/libembedding/datasets/beir-scifact/scifact \
  --model nomic-ai/nomic-embed-text-v1.5 \
  --model mixedbread-ai/mxbai-embed-large-v1
```
