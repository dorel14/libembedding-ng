---
nav_exclude: true
---

# Multi-vector (ColBERT) C API

`include/libembedding/multivector_embedding.h`: one vector **per token** instead of one per text (late interaction). ONNX Runtime backend only. Score with `lembed_maxsim` / `lembed_maxsim_batch` ([Similarity](similarity.html)). This API computes embeddings: no index, no sorting, no top-K.

## Reference

| Function | Description |
|---|---|
| `lembed_multivector_options_default()` | Default options |
| `lembed_multivector_create(options, &mv)` | Loads the model from `options->model_dir` (a local directory) |
| `lembed_multivector_embed(mv, texts, n, mode, &out)` | Embeds `n` texts as `LEMBED_MV_QUERY` or `LEMBED_MV_DOCUMENT` |
| `lembed_multivectors_free(&out)` | Frees `data` and `offsets`, zeroes the struct (idempotent) |
| `lembed_multivector_dim(mv)` | Vector dimension (96 for answerai-colbert-small-v1) |
| `lembed_multivector_desc(mv)` | Effective configuration (`lembed_model_desc_t`, owned by the handle) |
| `lembed_multivector_backend_name(mv)` | `"onnxruntime"` |
| `lembed_multivector_stats(mv, &st)` | Usage counters |
| `lembed_multivector_free(mv)` | Frees the handle |

## Result

`lembed_multivectors_t` is ragged: the vectors of text `i` are `data[offsets[i] * dim .. offsets[i+1] * dim)`, `offsets` has `num_docs + 1` entries, `offsets[0] == 0` and `offsets[num_docs] == total_vectors`.

## Pipeline

For each text: tokenise (with special tokens), truncate to `maxlen - 1`, insert the marker (`[unused0]` query, `[unused1]` document) at index 1, then:

- **query**: by default, **expansion**: padded with `[MASK]` up to `query_length`, attention 0 on the pads (unless `attend_to_expansion = 1`), **all** vectors are kept;
- **document**: the vectors of punctuation tokens (skiplist) are dropped; `[CLS]` and `[SEP]` are kept;
- only the inputs present in the graph are fed (`input_ids`, `attention_mask`, `token_type_ids` if the graph takes it);
- `normalize = 1`: per-vector L2 (idempotent: the official graph is already normalised).

## Model conventions (`onnx_config.json`, or `colbert_config.json`)

Fields read, all optional: `query_prefix_id`, `document_prefix_id`, `mask_token_id`, `query_prefix` / `document_prefix` (marker as a string, resolved in the vocabulary: `"[Q] "` is trimmed to `"[Q]"`), `query_length`, `document_length`, `attend_to_expansion_tokens`, `skiplist_words`, `embedding_dim`.

Defaults when absent: markers `[unused0]` / `[unused1]`, `[MASK]`, `query_length` 32, `document_length` 300, pad attention 0, skiplist `string.punctuation`. The `query_maxlen` / `doc_maxlen` options (> 0) override the model.

`do_query_expansion` is **not** consulted: the official answerai-colbert-small-v1 file declares `false`, yet the reference (pylate) expands its queries. Measured: minimum per-token cosine 0.999995 with expansion (pads masked out), 0.94 with attended pads, a different vector count without expansion. Turn it off with `query_expansion = 0`.

The ONNX file is looked up in this order: `model.onnx`, `vespa_colbert.onnx`, `onnx/model.onnx`. A string marker missing from the model vocabulary is an error (TODO: tokens added outside `model.vocab`, e.g. ModernBERT).

## Errors

| Case | Status |
|---|---|
| NULL / empty / negative argument | `LEMBED_ERROR_INVALID_ARGUMENT` |
| `.gguf` path | `LEMBED_ERROR_UNSUPPORTED` |
| Provider unavailable | `LEMBED_ERROR_UNSUPPORTED` |
| Graph output differs from `embedding_dim` (projection missing) | `LEMBED_ERROR_UNSUPPORTED` |
| Missing directory or file, invalid configuration, unresolved marker | `LEMBED_ERROR_IO` |

Details are in `lembed_last_error()`. Calls on one handle are serialised.

## Example

```c
lembed_multivector_options_t o = lembed_multivector_options_default();
o.model_dir = "models--answerdotai-answerai-colbert-small-v1";
lembed_multivector_t* mv = NULL;
if (lembed_multivector_create(&o, &mv) != LEMBED_OK) { /* lembed_last_error() */ }

const char* docs[] = { "Paris is the capital of France.", "Bananas grow in the tropics." };
const char* q[] = { "capital of France" };
lembed_multivectors_t dv, qv;
lembed_multivector_embed(mv, docs, 2, LEMBED_MV_DOCUMENT, &dv);
lembed_multivector_embed(mv, q, 1, LEMBED_MV_QUERY, &qv);

float scores[2];
lembed_maxsim_batch(qv.data, (int)qv.total_vectors, dv.data, dv.offsets, dv.num_docs, dv.dim, scores);

lembed_multivectors_free(&dv);
lembed_multivectors_free(&qv);
lembed_multivector_free(mv);
```

## Fidelity

Reference pylate 1.6.0, `answerai-colbert-small-v1`, 3 queries and 3 documents (`tests/data/colbert/`): same vector counts, minimum per-token cosine 0.999995 (queries and documents). Test: `tests/test_multivector_fidelity.cpp` (skipped when the model is absent).
