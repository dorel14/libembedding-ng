# Sparse ONNX vs sparse GGUF

- Date: 2026-11-05T18:18:59Z — git `107ec2b` — libembedding 1.11.0
- Machine: Windows-11-10.0.26300-SP0, 8 logical CPUs
- Dataset: scifact (1000 documents, 300 queries, split `test`)
- Settings: threads=0, batch=library default, max_length=512, top_k=0, rounds=2

## Retrieval quality

| Backend | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| onnx | 0.8110 | 0.9660 | 0.7908 |
| gguf | 0.8104 | 0.9660 | 0.7900 |

Top-10 ranking overlap ONNX vs GGUF: **0.990**

## Vector fidelity (GGUF against ONNX)

| Texts | n | cosine mean | cosine p5 | cosine min | max Δweight | top-10 term overlap | nnz GGUF/ONNX |
|---|---|---|---|---|---|---|---|
| documents | 1000 | 0.99989 | 0.99984 | 0.99967 | 0.1782 | 0.991 | 1.016 |
| queries | 300 | 0.99985 | 0.99975 | 0.99950 | 0.1406 | 0.996 | 1.022 |

### Where the documents' weight differences sit

Band = the larger of the two weights of a term. Relative = |Δ| / that weight.

| Weight band | terms | mean \|Δ\| | p99 \|Δ\| | max \|Δ\| | max relative |
|---|---|---|---|---|---|
| [0, 0.1) | 24156 | 0.0142 | 0.0491 | 0.0928 | 1.000 |
| [0.1, 0.5) | 66447 | 0.0139 | 0.0486 | 0.1782 | 1.000 |
| [0.5, 1) | 55798 | 0.0099 | 0.0339 | 0.0984 | 0.165 |
| [1, 2) | 58925 | 0.0070 | 0.0230 | 0.0830 | 0.075 |
| [2, inf) | 4755 | 0.0061 | 0.0180 | 0.0307 | 0.014 |

Terms present on one side only: 4384 (2.09 % of terms), weight p99 0.0512, max 0.1038.

## Cost

| Backend | corpus docs/s (median) | query p50 ms | load s | peak RSS MB | threads | batch_size | docs/batch |
|---|---|---|---|---|---|---|---|
| onnx | 1.51 | 43.9 | 2.49 | 903 | runtime default | 256 | 1.19 |
| gguf | 1.16 | 72.0 | 0.10 | 545 | 8 | 16 | 1.19 |

Throughput is the median over rounds; backends alternate within each round. Peak RSS is per subprocess. threads, batch_size and docs/batch are what each backend resolved and ran: docs/batch is the corpus size over the batches actually run, below batch_size when the token budget cuts. GGUF splits its threads over threads / 2 workers.
