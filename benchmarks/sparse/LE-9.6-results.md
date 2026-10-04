# LE-9.6: Four-backend comparison — Dense/Sparse × ONNX/GGUF

## Verdict

- **2** backend(s) measured, **0** unavailable, **0** failed.
- The **Sparse GGUF** row is libembedding's own ggml runtime, not llama.cpp: llama.cpp v0.3.0 cannot load these files at all, their tensor names and metadata keys being the ones it does not look for (`benchmarks/sparse/LE-9.4-validation.md` §3(d)). Fidelity against the sparse ONNX model is measured by `tests/test_sparse_gguf_vs_onnx.cpp`, not here.

## 1. Objectif

Compare the four backend×type combinations of LE-9.6 on one corpus: throughput, latency, memory, sparsity and Recall@K.

## 2. Environment

| Parameter | Value |
|---|---|
| platform | Windows-11-10.0.26300-SP0 |
| machine | AMD64 |
| cpu_count | 8 |
| hostname | PC_Asus |
| python | 3.12.10 |
| libembedding | 1.11.0 |
| git | 7a66ca5 |
| timestamp | 2026-10-04T18:50:17Z |
| corpus (timed) | 91 docs |
| warmup (untimed) | 4 docs |
| batch_size | 32 |
| timed iterations | 3 (median reported) |
| top_k | 50 |

## 3. Results

| Backend | Model | Quant. | Top-K | Load (ms) | docs/s | ms/doc | Peak RAM (MB) | nnz | vocab | Sparsity (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Sparse ONNX | `prithivida/Splade_PP_en_v1` | fp32 | 50 | 1661 | 9.4 | 106.17 | 1374 | 49 | 30522 | 0.162 |
| Sparse GGUF | `cstr/splade-pp-en-v1-GGUF` | q8_0 | 50 | 105 | 2.9 | 340.66 | 908 | 50 | 30522 | 0.163 |

## 4. How to read these numbers

- **Peak RAM** is a per-backend high-water mark: each backend ran in its own subprocess, so the value is attributable to that backend alone.
- **nnz** for a sparse row is capped by `--top-k` (50). It measures the truncation policy, not the model's natural density. Re-run with `--top-k 0` for the untruncated figure.
- **Sparsity** is `nnz / vocab_size`. Dense rows are 100% by convention and carry no vocabulary.
- **Recall@K** was not computed (no `--recall-corpus` supplied). The comparison then covers throughput, memory and sparsity only.

## 5. Backends not measured

None — every backend in the matrix was measured.

## 6. Why the sparse GGUF row is 3.5x slower than sparse ONNX

Measured on the same machine, same runtime, same corpus:

| Corpus (batch 32) | docs/s |
|---|---|
| 12-token documents | 23.5 |
| this benchmark's corpus (~40 tokens) | 2.6 |

Same model, same code. The only variable is text length.

And the time per batch grows with the **square** of the batch:

| batch | ms/batch | vs batch 1 |
|---|---|---|
| 1 | 108 | 1x |
| 8 | 1 462 | 13.5x |
| 32 | 12 198 | **113x** |
| 64 | 40 249 | **372x** |

The arithmetic is linear in the batch — the MLM head is one `[768, 30522]`
projection per token — so 32x the work should cost 32x the time. It costs 113x.
**Batching is currently an anti-optimisation here.**

### Two hypotheses, both measured, both wrong

**Not the Q8_0 decoder.** `splade-pp-en-v1-q8_0.gguf` (111 MB) and
`splade-pp-en-v1.gguf` (418 MB, F32) were run through the identical runtime:

| File | docs/s | Peak RAM |
|---|---|---|
| Q8_0 | 18.25 | 388 MB |
| F32 | 19.84 | 1111 MB |

Q8_0 is only **8 % slower** than F32. Dequantising the tied decoder once at load
would buy that 8 % at the cost of 93 MB of RAM — a bad trade, and not the answer.

**Not the per-batch graph rebuild.** With the corpus above, the whole 95-document
run takes 37 s across 3 batches; `ggml_init` + `ggml_gallocr_alloc_graph` three
times cannot account for it.

### What it actually is: attention packed with the batch

`bert_attention` flattens all `batch * seq_len` tokens into one axis, so the
scores are `[n_tokens, n_tokens, n_head]` and the mask `[n_tokens, n_tokens]`
block-diagonal. Both grow with the **square** of the batch. At batch 32 and
40 tokens that is a 1280x1280x12 score tensor — **78.6 MB per layer**, 943 MB of
traffic over 12 layers.

The arithmetic itself is only ~1.3 % of the FLOPs, which is exactly why a
first reading of this under-rates it: attention is quadratic in the sequence while
the MLP and the MLM head are linear, so its *share* climbs with length and is
negligible at 12 tokens and dominant at 40.

### The fix, and the obstacle

Make the batch an axis of its own so the scores are
`[seq_len, seq_len, n_head, batch]` — linear in the batch. This is what
CrispEmbed does, and its own comment names the trade: *"Attention is O(B·T²) not
O((B·T)²), so it scales far better than packing for many short sequences."*
The mask drops from 6.6 MB to seq_len²·n_head·batch floats, and the padding-query
row needs no self-only special case any more, because a per-item row attends to
its own document's real tokens.

Two ggml details that make this less mechanical than it looks, both found the
hard way:

1. `ggml_compute_forward_soft_max_f32` walks
   `src1->data + i11*nb11 + i12*nb12 + i13*nb13` with **no broadcast on axes 2
   and 3**. A mask with `ne[2] == 1` steps a whole plane per head and reads past
   its own buffer. The head axis must be spelled out: `[seq_len, seq_len, n_head,
   batch]`.
2. The layout wanted by `ggml_mul_mat` — `[head_dim, seq_len, n_head, batch]` —
   **cannot be produced by `ggml_reshape_4d` from `[n_embd, seq_len*batch]`**.
   Reshaping splits the two input axes into four output axes as
   `[head_dim, n_head, seq_len, batch]`, which puts `n_head` on axis 1 and makes
   `mul_mat` return `[n_head, n_head, seq_len, batch]`. The two constraints
   cannot both be satisfied by a reshape; a transpose-and-contiguous step, or the
   per-item projection layout, is needed.

An attempt at the rewrite is in the history of this branch and was reverted: it
compiled, and `test_sparse_gguf` caught it with *"a batch of two gives the same
vector as a batch of one"*. That assertion is the guard that matters here and it
should stay.

### Also worth knowing

- On short documents the runtime is already **faster than the ONNX backend**
  (23.5 vs 9.4 docs/s at 12 tokens), because it avoids ONNX Runtime's per-call
  setup. The crossover is around 20 tokens.
- The GGUF path still wins decisively on the three things that matter for
  distribution: **388 MB vs 1374 MB** peak RSS, **118 ms vs 1176 ms** load, and a
  111 MB file instead of a multi-hundred-megabyte ONNX one.

