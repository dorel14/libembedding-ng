# LE-9.6: Four-backend comparison — Dense/Sparse × ONNX/GGUF

## Verdict

- **3** backend(s) measured, **1** unavailable, **0** failed.
- **Sparse GGUF** is unavailable — blocked, not measured: llama.cpp v0.3.0 loads a BERT GGUF but neither loads nor builds the mlm_* projection head SPLADE requires (third_party/llama.cpp/src/models/bert.cpp:23-74 and 227-232), so it never produces the [seq_len, vocab_size] logits. The GGUF files themselves do carry the head, so a different runtime may work -- see the P1.5a/P1.5b spike in benchmarks/sparse/LE-9.4-validation.md before concluding anything about sparse GGUF
- Consequently there is no sparse GGUF column: libembedding's sparse path is ONNX-only today. This is a *blocked* state, not a closed door — the P1.5a/P1.5b spike decides whether a different runtime changes it (`benchmarks/sparse/LE-9.4-validation.md`).

## 1. Objectif

Compare the four backend×type combinations of LE-9.6 on one corpus: throughput, latency, memory, sparsity and Recall@K.

## 2. Environment

| Parameter | Value |
|---|---|
| platform | Windows-11-10.0.26200-SP0 |
| machine | AMD64 |
| cpu_count | 8 |
| hostname | PC_Asus |
| python | 3.12.10 |
| libembedding | 1.10.1 |
| git | 6d22255 |
| timestamp | 2026-10-03T16:22:03Z |
| corpus (timed) | 95 docs |
| warmup (untimed) | 5 docs |
| batch_size | 32 |
| timed iterations | 3 (median reported) |
| top_k | 50 |

## 3. Results

| Backend | Model | Quant. | Top-K | Load (ms) | docs/s | ms/doc | Peak RAM (MB) | nnz | vocab | Sparsity (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense ONNX | `sentence-transformers/all-MiniLM-L6-v2` | fp32 | — | 352 | 134.7 | 7.42 | 203 | 384 | — | 100.000 |
| Dense GGUF | `MiniLM-L6-Q4` | q4_k_m | — | 81 | 53.0 | 18.87 | 75 | 384 | — | 100.000 |
| Sparse ONNX | `prithivida/Splade_PP_en_v1` | fp32 | 50 | 1347 | 8.5 | 117.97 | 1371 | 50 | 30522 | 0.162 |
| Sparse GGUF | `SPLADE-PP-En-v1` | q4_k_m | — | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## 4. How to read these numbers

- **Peak RAM** is a per-backend high-water mark: each backend ran in its own subprocess, so the value is attributable to that backend alone.
- **nnz** for a sparse row is capped by `--top-k` (50). It measures the truncation policy, not the model's natural density. Re-run with `--top-k 0` for the untruncated figure.
- **Sparsity** is `nnz / vocab_size`. Dense rows are 100% by convention and carry no vocabulary.
- **Recall@K** was not computed (no `--recall-corpus` supplied). The comparison then covers throughput, memory and sparsity only.

## 5. Backends not measured

### Sparse GGUF — `SPLADE-PP-En-v1`

Reason: blocked, not measured: llama.cpp v0.3.0 loads a BERT GGUF but neither loads nor builds the mlm_* projection head SPLADE requires (third_party/llama.cpp/src/models/bert.cpp:23-74 and 227-232), so it never produces the [seq_len, vocab_size] logits. The GGUF files themselves do carry the head, so a different runtime may work -- see the P1.5a/P1.5b spike in benchmarks/sparse/LE-9.4-validation.md before concluding anything about sparse GGUF

