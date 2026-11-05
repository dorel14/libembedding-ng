# Sparse GGUF runtime

Version: 1.11.0
Status: implemented, covered by `tests/test_sparse_gguf.cpp` (96 assertions)
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
[`LE-9.6-results.md`](../archive/LE-9.6-results.md) (archived)
is the *correct* result, not a gap to close.

The runtime is therefore ours: a ggml graph on the already-vendored CPU
backend, reading tensors through `gguf_init_from_file` and computing in
`ggml_mul_mat`. That is the conclusion of
[`LE-9.4-validation.md` section 5.3](../archive/LE-9.4-validation.md) (archived).

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

Texts in each call are tokenized once, sorted by descending length, then split
into batches bounded two ways: at most `batch_size` texts, and at most 512
padded token columns (`batch × seq_len`, see
`detail/gguf/sparse_batching.hpp`). Results are restored to input order. The
second bound is the one that caps the cost: the graph and the
`[vocab, batch × seq_len]` F32 logits grow with the padded width, not with the
number of texts.
On `benchmarks/corpus.txt` (392 texts), batch 16 and 4 threads, one pass
processed 8,256 tokens before bucketing (6,830 real + 1,426 padded, 17.3%
padding) and 6,928 after (6,830 real + 98 padded, 1.4%). This reduces token
work by 16.1%. Graph builds per pass fell from 20 to 9. Observed median latency
went from 27.76 s to 21.11 s, but the before/after repetitions were not paired;
this throughput result is indicative and needs confirmation with interleaved
runs.
Changing graph shapes can also change floating-point rounding slightly. On
these 392 texts, before/after bucketing at batch 16, full vectors had a maximum
weight delta of 0.00168 and a minimum cosine of 0.99999984; 12 texts differed
by one near-zero-score term. With `top_k = 50`, two texts selected a different
term at the ranking boundary (minimum cosine 0.99759). Bucketing preserves
output order and typical results remain very close, but exact term identity is
not guaranteed for these near-ties.

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
the documents, not a second graph. The synthetic fixed-weight fixture in
`test_sparse_gguf.cpp` checks exact equality between batch 1 and batches 2 to 5;
this does not guarantee bitwise equality for a real quantized model, because GEMM
shapes change with batch size and padding.

On `splade-pp-en-v1-q8_0.gguf`, compared with embedding each of the 392 texts in
`benchmarks/corpus.txt` individually (4 threads, batch 16), the full vectors
have a minimum cosine of 0.99976 without sorting and 0.99986 with sorting. The
minimum cosine over the top 50 terms alone is 0.99807 and 0.99877, respectively.
The set of 50 terms differs from the individual reference for 52 texts without
sorting, versus 9 after length sorting. Sorting therefore reduces the measured
drift, but does not guarantee identical selected terms.

For example, for the corpus's second text, term 2156 changes from 0.16757 at
batch 1 to 0.14872 at sorted batch 16, while term 2235 changes from 0.16218 to
0.16151; the top-50 selection changes. Ranking scores on a query set with
relevance labels have not been measured. Applications that depend on exact
top-term selection should validate their own corpus and batch configuration.

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
- **Single-shape graph cache**: the graph and its allocation are reused while
  sequence length and batch size stay unchanged. A different shape replaces
  the cache and triggers a rebuild, so variable-length batches can still pay
  this cost.
- **Full projection and logits**: `top_k` limits the sparse output, not the MLM
  projection over the vocabulary. The graph still produces logits for every
  token and transfers them to the host before pooling.
- **Batch size**: the GGUF default is 16 texts per batch and remains
  configurable. On a 392-text corpus measured on an i7-1065G7, batch 2 with
  4 threads reached 18.0 documents/s, versus 16.7 at batch 16 and 14.2 at
  batch 64 in an initial sweep. An alternating 16/2/2/16 repeat measured
  averages of 16.2 and 16.3 documents/s, respectively: comparable throughput
  within observed variation. Batch 2 used about 252 MiB peak RSS and 5.3 %
  padding, versus 334 MiB and 17.3 % for batch 16; batch 16 reduces the number
  of batches and showed slightly lower variation. It remains the default as a
  compromise, not as the minimum-memory setting or a universal optimum. Results
  depend on CPU, corpus, and sequence lengths.
- **Per-batch token budget**: at most 512 padded columns per graph, whatever
  `batch_size` is. Measured on an i7-1065G7, 4 threads, with the
  `integration_sparse_gguf` profiler:
  - 256 short texts (`customers_sample.txt`, ~13 tokens), batch 256: no budget
    12.6 documents/s and 1,642 MiB peak RSS; budget 128 / 256 / 512 / 1024:
    25.5 / 26.1 / 26.4 / 24.0 documents/s, 276 / 325 / 415 / 596 MiB.
  - 96 Reuters stories (`reuters_clean.txt`, long), batch 16: no budget
    1.51 documents/s and 2,208 MiB; budget 512 / 1024 / 2048: 1.83 / 1.61 /
    1.63 documents/s, 358 / 481 / 728 MiB.

  512 gives the best throughput on both profiles and bounds memory.

  The same budget (`kSparseMaxBatchTokens`, `detail/sparse_batching.hpp`)
  applies to the sparse ONNX path, which had neither length sorting nor a cap:
  its default batch of 256 asked ONNX Runtime for 16 GB on 512-token documents
  (`bad allocation`). ONNX sweep, batch 256: 128 SciFact abstracts 1.51 / 1.52 /
  1.50 / 1.49 documents/s and 842 / 839 / 1,084 / 1,478 MiB for 256 / 512 /
  1024 / 2048 tokens; 256 short texts 38.6 / 41.3 / 37.9 / 33.2 documents/s,
  versus 22.6 before the fix.
- **Workers**: a GGUF session runs several independent graphs (`workers`) that
  share one read-only copy of the weights and split the batches of a call. One
  graph stops gaining from extra threads (ggml synchronises every thread after
  each of its ~460 operations); independent graphs do not wait for each other.
  `num_threads` is the total budget (0 = every logical CPU) and `workers` the
  number of graphs (0 = `num_threads / 2`, one per physical core on a
  hyper-threaded CPU). A call that is a single batch, typically a query, runs
  on one worker with the whole budget. Measured on an i7-1065G7 (profiler,
  interleaved): 256 short texts 26.1 / 25.1 documents/s at 1 × 4, 23.5 at
  1 × 8, 34.6 / 31.4 auto (4 × 2), RSS 279 → 320 MiB; 64 SciFact abstracts
  0.92 / 0.93 at 1 × 4, 0.98 at 1 × 8, 1.24 / 1.22 auto, RSS 298 → 491 MiB.
  Logits are read in place from the graph buffer, with no per-worker copy.

## 7. GGUF profiler

The manual `integration_sparse_gguf` profiler measures the runtime with a local
GGUF model. Its opt-in instrumentation is compiled into this executable only;
the normal DLL and public API are unchanged. Build it with tests enabled:

```powershell
cmake --build build --config Release --target integration_sparse_gguf
build\tests\Release\integration_sparse_gguf.exe `
  --model "$HOME\.cache\libembedding\models--cstr-splade-pp-en-v1-GGUF\splade-pp-en-v1-q8_0.gguf" `
  --corpus benchmarks\corpus.txt `
  --batch 4 --threads 4 --warmups 2 --repetitions 30 --top-k 50
```

`--corpus` expects one non-empty document per line. Without it, the profiler
uses twelve built-in texts. `--batch`, `--threads`, `--warmups`,
`--repetitions`, `--top-k` and `--max-length` configure batch size, ggml
threads, untimed warmups, timed passes, pruning and maximum sequence length.
`--max-batch-tokens` overrides the per-batch token budget (0 disables it);
`--workers` sets the number of graphs (0 = auto) and `--threads` the total
thread budget (0 = every logical CPU).

For each timed pass, the stage table reports median, mean, minimum, empirical
nearest-rank p95, maximum and standard deviation for tokenisation, input
packing, graph construction/allocation, attention mask, input upload, ggml
execution, logits download, pooling, pruning/sorting and unattributed wrapper
time. The summary also reports document throughput, average nnz, current/peak
RSS, process CPU/wall time, graph and padding counters, graph shape/node count,
and an inventory of ggml operations.

Stage durations are aggregated across every batch in a pass; they are not
per-document latencies or per-operator timings. `GraphExec` times
`ggml_backend_graph_compute`: it can include CPU planning and workspace
preparation/allocation, and does not isolate individual kernel times. The final
operation inventory counts nodes but does not assign time to them. Use at least
30 repetitions for a more informative tail estimate; with few repetitions, p95
can equal the maximum. The first warmup builds the graph and reports that cost
separately from timed passes.

### Q8_0 GEMM microbenchmark

The `gguf_gemm_microbench` target isolates ggml matrix multiplications using
synthetic Q8_0-quantized weights and F32 activations. It measures the BERT
shapes `768x768`, `3072x768`, `768x3072` and the vocabulary projection
`30522x768`, for `N=32, 64, 80, 128, 256, 512`, at 1, 2 and 4 threads. Its
persistent thread pools are paused outside their measurements so an inactive
pool does not consume CPU time.

```powershell
cmake -S . -B build -DLIBEMBEDDING_BUILD_BENCHMARKS=ON
cmake --build build --config Release --target gguf_gemm_microbench
build\benchmarks\Release\gguf_gemm_microbench.exe `
  --warmups 2 --repetitions 5
```

`--shape-index 0..3` and `--n 32|64|80|128|256|512` restrict a run to one
shape or column count. `--threads 1|2|4` selects the worker count;
`--cpumask <mask>` applies a strict affinity mask to the ggml pool, for example
`--threads 4 --cpumask 0x55`. The latter is for comparing CPU placements in
this microbenchmark; it does not change libembedding runtime affinity. GFLOPS
uses `2*M*K*N` operations per product. This benchmark compares shapes and
thread scaling; its synthetic weights, lack of attention and reduced graph do
not reproduce full inference or its real weights.