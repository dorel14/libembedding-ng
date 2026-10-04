# Sparse GGUF runtime

Version: 1.11.0
Status: implemented, covered by `tests/test_sparse_gguf.cpp` (67 assertions)
and `tests/test_sparse_gguf_vs_onnx.cpp` (integration, network required)

Describes **how a sparse GGUF file is executed**. The
[sparse GGUF convention](gguf_sparse_convention.html) describes what a file
*contains*. The two are separate and both normative: the convention decides
whether a file is accepted, this page describes the computation.

---

## 1. Entry points

```c
/* A local file */
lembed_sparse_embedding_ctx_t* ctx = NULL;
lembed_sparse_options_t opts = lembed_sparse_options_default();
opts.top_k = 256;

lembed_status_t s = lembed_sparse_text_embedding_create_from_gguf_path(
    "splade-pp-en-v1-q8_0.gguf", &opts, &ctx);

/* Or a HuggingFace repository, like the dense backend */
s = lembed_sparse_text_embedding_create_from_gguf_model(
    "cstr/splade-pp-en-v1-GGUF", "splade-pp-en-v1-q8_0.gguf", &opts, &ctx);

lembed_sparse_embeddings_t out;
lembed_sparse_text_embedding_embed(ctx, texts, n, 8, NULL, &out);
lembed_sparse_embeddings_free(&out);
lembed_sparse_text_embedding_free(ctx);
```

`lembed_sparse_text_embedding_create_from_path()` **routes on the extension**: a
path ending in `.gguf` goes straight to the GGUF runtime. A caller that only
knows a path therefore does not have to know the backend.

In Python:

```python
from libembedding import SparseTextEmbedding

# Local path
model = SparseTextEmbedding.from_gguf("splade-pp-en-v1-q8_0.gguf")

# HuggingFace repository
model = SparseTextEmbedding.from_gguf_model(
    "cstr/splade-pp-en-v1-GGUF", "splade-pp-en-v1-q8_0.gguf"
)

vectors = model.embed(["a fast brown fox"])
```

`SparseTextEmbedding("path/to/model.gguf")` works too: the routing is done in C.

The naming is **runtime-agnostic**: there is no `_llama_` in the identifiers.
llama.cpp is not involved (see section 2).

---

## 2. Why llama.cpp is not used

This is the non-negotiable point of the implementation, and it is verifiable in
the vendored snapshot.

llama.cpp v0.3.0 **cannot load** this family of files. It looks for
`blk.N.attn_q.weight`, `blk.N.ffn_up.weight`, `blk.N.layer_output_norm.weight`,
`blk.N.attn_output_norm.weight`, `token_embd_norm.weight` and
`token_types.weight` (`third_party/llama.cpp/src/llama-arch.cpp:407-453`), and
reads `%s.attention.head_count`, `%s.feed_forward_length`,
`%s.embedding_length`, `%s.attention.layer_norm_epsilon`. The SPLADE exports
carry:

| llama.cpp expects | the file contains |
|---|---|
| `blk.N.attn_q.weight` | `enc.N.attn.q.weight` |
| `blk.N.ffn_up.weight` | `enc.N.ffn.fc1.weight` |
| `blk.N.layer_output_norm.weight` | `enc.N.ln2.weight` |
| `blk.N.attn_output_norm.weight` | `enc.N.ln1.weight` |
| `token_embd_norm.weight` | `embd_ln.weight` |
| `token_types.weight` | `token_type_embd.weight` |
| `%s.embedding_length` | `bert.hidden_size` |
| `%s.block_count` | `bert.num_hidden_layers` |
| `%s.attention.head_count` | `bert.num_attention_heads` |
| `%s.feed_forward_length` | `bert.intermediate_size` |
| `%s.attention.layer_norm_epsilon` | `bert.layer_norm_eps` |
| `%s.context_length` | `bert.max_position_embeddings` |

A missing required tensor is fatal: `create_tensor` throws
`missing tensor 'blk.0.attn_q.weight'`
(`src/llama-model-loader.cpp:1103-1106`). Only `token_embd.weight` and
`position_embd.weight` match.

Consequently `llama.cpp = n/a` in
[`benchmarks/sparse/LE-9.6-results.md`](https://github.com/dorel14/libembedding-ng/blob/dev/benchmarks/sparse/LE-9.6-results.md)
is the *correct* result, not a gap to close.

The runtime is therefore ours: a ggml graph on the already-vendored CPU
backend, reading tensors through `gguf_init_from_file` and computing in
`ggml_mul_mat`. That is the conclusion of
[`LE-9.4-validation.md` section 5.3](https://github.com/dorel14/libembedding-ng/blob/dev/benchmarks/sparse/LE-9.4-validation.md).

---

## 3. The pipeline

```
text
  -> WordPiece ids         TokenizerWrapper (shared with the ONNX backend)
  -> BERT encoder          ggml graph, post-LayerNorm
  -> MLM head              ggml graph, ggml_mul_mat against token_embd
  -> log(1 + relu) max-pool  on the host, real tokens only
  -> special ids removed   ids from the metadata
  -> top_k / min_weight / storage_format   sparse_prune_and_sort()
  -> {indices, values}
```

### 3.1 Encoder - the post-LN order is not negotiable

BERT is **post-LayerNorm**, unlike almost every transformer llama.cpp runs.
Getting the order wrong produces a model that loads, runs and returns plausible
vectors:

- the QKV projections read the **normalised** input;
- `ln1` applies **after** the attention residual;
- `ln2` applies **after** the feed-forward residual.

The graph is written in that order, copied from llama.cpp's own BERT builder
(`src/models/bert.cpp:114-225`), which does exactly this. The implementation is
`include/libembedding/detail/gguf/gguf_bert_graph.hpp`.

Two details verified in the vendored ggml:

- `ggml_gelu` **is** the tanh approximation
  (`ggml-cpu/vec.h:968-970`: `0.5x(1+tanh(sqrt(2/pi) * x(1+0.044715x^2)))`),
  which is what a BERT export is trained against. `ggml_gelu_erf` is a different
  function and must not be substituted. `llama-graph.cpp:1806-1813` makes the
  same choice for the BERT FFN.
- `ggml_norm` **subtracts the mean** (`ggml-cpu/ops.cpp:3721-3737`): it is a true
  LayerNorm, not an RMSNorm. It works on F32 only, so the LayerNorm scales and
  biases are applied separately (`ggml_mul` then `ggml_add`), which tolerates
  quantised LayerNorm weights.

### 3.2 MLM head - in the graph, never in a loop

```c
x    = ggml_add(ggml_mul_mat(mlm_transform_w, h), mlm_transform_b);
x    = ggml_gelu(x);
x    = layernorm(x, mlm_ln_w, mlm_ln_b, eps);
logits = ggml_add(ggml_mul_mat(token_embd_w, x), mlm_bias);   /* [vocab, tokens] */
```

The decoder projection **is** the token embedding matrix: BERT ties the MLM head
to the token embeddings, so there is no separate `vocab x hidden` matrix in the
file, and none is invented here.

The head is a `ggml_mul_mat`, not a scalar loop. A loop would be
`O(tokens x hidden x vocab)`, unacceptable at V = 30522, and the operator also
dequantises Q8_0 for us, which disposes of the dequantised-read trap: **no
quantised weight is ever read element by element** (a `ggml_backend_tensor_get`
past `ggml_nbytes` aborts the process).

### 3.3 Batches and mask

Every token of a batch forms **a single row**. The [T, T] attention mask is
block-diagonal by document and -infinity on padding; that mask is what separates
the documents, not a second graph. A batch of two and a batch of five therefore
produce the same vector - which `test_sparse_gguf.cpp` checks.

### 3.4 Tokenizer: shared, not duplicated

A GGUF carries only a vocabulary (`tokenizer.ggml.tokens`), no
`tokenizer.json`. The runtime feeds the **same** `TokenizerWrapper` as the ONNX
backend from those strings. The intended consequence: both backends tokenise
identically by construction, so the similarity measured between them measures
quantisation and the runtime, not tokenizer drift.

The price is that the ONNX backend's known approximations are shared rather than
duplicated: ASCII-only lowercasing, punctuation split per byte, no accent
stripping. That is a deliberate choice, not an oversight; a faithful HuggingFace
tokenizer would need generated Unicode tables (~470 KB) in order to diverge from
the reference backend this runtime is measured against.

---

## 4. Refusals

An unreadable file, one with no sparse head, no vocabulary, or an unsuitable
head is **refused with a reason**, never accepted silently.

| Situation | Code | Message |
|---|---|---|
| null / empty path | `LEMBED_ERROR_INVALID_ARGUMENT` | - |
| unreadable file or not a GGUF | `LEMBED_ERROR_UNSUPPORTED` | `cannot read GGUF file '...'` |
| no `general.architecture` | `LEMBED_ERROR_UNSUPPORTED` | `GGUF header has no general.architecture` |
| no `<arch>.vocab_size` | `LEMBED_ERROR_UNSUPPORTED` | `GGUF has no <arch>.vocab_size to pool over` |
| no LayerNorm epsilon | `LEMBED_ERROR_UNSUPPORTED` | `GGUF header has no <arch>.attention.layer_norm_epsilon` |
| missing encoder tensor | `LEMBED_ERROR_UNSUPPORTED` | `missing encoder tensor '...'` (from `classify()`) |
| no `mlm_transform.weight` | `LEMBED_ERROR_UNSUPPORTED` | `GGUF carries no SPLADE head (mlm_transform.weight is absent)` |
| formula is not SPLADE | `LEMBED_ERROR_UNSUPPORTED` | `GGUF sparse formula is not SPLADE` |
| **both** SPLADE and `sparse_linear` | `LEMBED_ERROR_UNSUPPORTED` | `refusing to guess which one this file is meant for` |
| vocabulary absent or wrong size | `LEMBED_ERROR_UNSUPPORTED` | `GGUF has no tokenizer.ggml.tokens vocabulary array` |
| non-BERT tokenizer | `LEMBED_ERROR_UNSUPPORTED` | `GGUF declares tokenizer model '...', which this runtime does not implement` |
| ggml error | `LEMBED_ERROR_GGUF` | exception message |

The refusal happens **before** any weight allocation: telling someone their dense
file has no sparse head must not cost a hundred megabytes of copying.
`LEMBED_ERROR_GGUF` is a code *appended* to the enumeration, not
`LEMBED_ERROR_LLAMA`, because llama.cpp is not involved.

`max_length = 0` means "as long as the position table allows". The request is
**capped** at the model's context length rather than refused: a longer sequence
would index past the position table.

---

## 5. Measurements

LE-9.6 criterion: cosine >= 0.90 between sparse GGUF and sparse ONNX, on the same
texts.

```
sparse GGUF vs ONNX: min cos 0.9998, mean cos 0.9999 over 12 texts (bar 0.90)
```

Measured on `cstr/splade-pp-en-v1-q8_0.gguf` (Q8_0, ~111 MB) against
`prithivida/Splade_PP_en_v1` (ONNX), 12 texts, `top_k = 0`. The margin is an
order of magnitude above the bar: what is measured is Q8_0 quantisation noise,
not an implementation divergence.

Both models derive from the same HuggingFace export, which is what makes the
comparison legitimate; the GGUF repository is Apache-2.0.

`tests/test_sparse_gguf.cpp` is hermetic: fixtures written to a temp directory,
weights from a deterministic generator, F32 / F16 / Q8_0. No network, a few
milliseconds.

### Throughput, and where it stands

`benchmarks/sparse/bench_sparse_backend_comparison.py`, 95 timed documents,
batch 32, `top_k = 50`:

| Backend | docs/s | ms/doc | Load (ms) | Peak RAM (MB) |
|---|---|---|---|---|
| Sparse ONNX | 9.7 | 102.80 | 1176 | 1368 |
| Sparse GGUF (this runtime) | 2.8 | 357.85 | 109 | 905 |

The runtime is therefore **3.5x slower** than ONNX Runtime, for **a third less
RAM** and a **ten times faster** load. Throughput is the weak point, and it is
expected: every call rebuilds the graph and reserves a fresh activation buffer,
where ONNX Runtime keeps a fused graph and its memory pool. The fix is known and
bounded -- see "No graph cache" below -- but it is not done in P1.5c, whose exit
criterion was fidelity.

---

## 6. Known limits

- **Memory at load**: the GGUF blob is read into RAM and then copied into the
  backend buffer, so the peak is twice the file size (~222 MB for 111 MB). The
  steady state is the file itself: the blob is freed as soon as the vocabulary
  has been extracted from it.
- **Context length**: capped by `bert.max_position_embeddings` (512 on SPLADE
  PP v1). Beyond that, positions do not exist.
- **Tokenisation**: see 3.4. On accented text both backends diverge from
  HuggingFace in the same way.
- **No GPU**: ggml CPU backend only. `provider` is reported as
  `LEMBED_PROVIDER_CPU` and ignored at construction.
- **No graph cache**: the graph is rebuilt on every call. On a single-text
  streaming workload that would show; it is not the intended usage profile.