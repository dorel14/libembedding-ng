# LE-9.6: Four-backend comparison — Dense/Sparse × ONNX/GGUF

## Verdict

- **1** backend(s) measured, **0** unavailable, **0** failed.
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
| git | b052dcd |
| timestamp | 2026-10-06T18:14:21Z |
| corpus (timed) | 91 docs |
| warmup (untimed) | 4 docs |
| batch_size | 1 |
| timed iterations | 3 (median reported) |
| top_k | 50 |

## 3. Results

| Backend | Model | Quant. | Top-K | Load (ms) | docs/s | ms/doc | Peak RAM (MB) | nnz | vocab | Sparsity (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Sparse GGUF | `cstr/splade-pp-en-v1-GGUF` | q8_0 | 50 | 148 | 8.5 | 117.10 | 294 | 50 | 30522 | 0.163 |

## 4. How to read these numbers

- **Peak RAM** is a per-backend high-water mark: each backend ran in its own subprocess, so the value is attributable to that backend alone.
- **nnz** for a sparse row is capped by `--top-k` (50). It measures the truncation policy, not the model's natural density. Re-run with `--top-k 0` for the untruncated figure.
- **Sparsity** is `nnz / vocab_size`. Dense rows are 100% by convention and carry no vocabulary.
- **Recall@K** was not computed (no `--recall-corpus` supplied). The comparison then covers throughput, memory and sparsity only.

## 5. Backends not measured

None — every backend in the matrix was measured.

