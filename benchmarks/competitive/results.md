# Competitive dense benchmark (T15, reduced protocol) — 2026-11-22

libembedding against SentenceTransformers and FastEmbed on the same machine, the
same corpus and the same batch size. Script: `bench_competitive.py`.

## Conditions

| Field | Value |
|---|---|
| Machine | Intel i7-1065G7 (4 cores / 8 threads, AVX-512), Windows 11, on mains, quiet machine |
| Versions | libembedding-ng 1.16.0 (PyPI wheel, ONNX Runtime 1.31.0), sentence-transformers 6.3.0 (torch 2.15.0+cpu), fastembed 0.9.1 |
| Corpus | first 1000 non-empty lines of `benchmarks/datas/customers_sample.txt` (short texts) |
| Batch / threads | 32 / library default |
| Protocol | warmup on 50 texts, 3 timed passes (median), one process per measurement, 3 rounds with the library order rotated, median of the rounds |
| Models | `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, FP32, each library's own default files |

## Results

| Model | Library | docs/s (min-max) | Load (ms) | Peak RSS (MB) |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | libembedding | 271.7 (269.1-275.5) | **692** | **203** |
| all-MiniLM-L6-v2 | SentenceTransformers | 279.4 (278.8-286.7) | 9554 | 539 |
| all-MiniLM-L6-v2 | FastEmbed | **283.1** (276.2-290.9) | 1042 | 230 |
| bge-small-en-v1.5 | libembedding | 138.8 (135.8-139.3) | **902** | **246** |
| bge-small-en-v1.5 | SentenceTransformers | **148.2** (145.7-148.9) | 9777 | 580 |
| bge-small-en-v1.5 | FastEmbed | 132.0 (129.5-132.1) | 1103 | 272 |

## Reading

- Throughput is on par: libembedding is 3 to 4 % behind on MiniLM, 6 % behind
  SentenceTransformers and 5 % ahead of FastEmbed on bge-small. These gaps are of the
  order of the run-to-run spread: no library wins clearly at batch 32 on short texts.
- Load time is where libembedding stands out: 0.7 to 0.9 s, against about 1.0 to 1.1 s
  for FastEmbed and 9.5 to 9.8 s for SentenceTransformers (torch import included).
- Peak memory is the lowest: 203-246 MB, against 230-272 MB for FastEmbed and
  539-580 MB for SentenceTransformers.

## Limits

- Reduced protocol: one batch size, default threads, two models, FP32 ONNX only. The full
  LE-11 protocol (batch 1 to 128, three thread settings, GGUF, sparse, quality on BEIR
  pairs) remains to be run.
- Each library uses its own model files and its own defaults (pooling, max length),
  so the comparison measures the out-of-the-box experience, not identical graphs.
- A single machine: no conclusion for ARM or AMD.
