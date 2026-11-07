# Sparse ONNX vs sparse GGUF

- Date: 2026-11-04T19:56:42Z — git `1d2ef88` — libembedding 1.11.2
- Machine: Windows-11-10.0.26300-SP0, 8 logical CPUs
- Dataset: scifact (1000 documents, 300 queries, split `test`)
- Settings: threads=4, batch=16, max_length=512, top_k=0, rounds=2

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

## Cost

| Backend | corpus docs/s (median) | query p50 ms | load s | peak RSS MB |
|---|---|---|---|---|
| onnx | 0.97 | 41.9 | 9.78 | 4119 |
| gguf | 0.88 | 59.5 | 1.04 | 410 |

Throughput is the median over rounds; backends alternate within each round. Peak RSS is per subprocess.
