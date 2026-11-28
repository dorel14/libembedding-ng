# Retrieval quality by quantization variant

Dataset: BEIR SciFact test (5183 docs, 300 queries). Brute-force cosine, no timing claim.
Cosine vs FP32: mean cosine between a document embedding and the FP32 entry's embedding of the same document.

| Model | Registry entry | Quantization | nDCG@10 | MRR@10 | Recall@100 | Cosine vs FP32 |
|---|---|---|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | Xenova/bge-small-en-v1.5 | none | 0.7127 | 0.6822 | 0.9417 | 1.00000 |
| BAAI/bge-small-en-v1.5 | onnx-community/bge-small-en-v1.5-ONNX | dynamic | 0.7077 | 0.6766 | 0.9467 | 0.99831 |
| BAAI/bge-small-en-v1.5 | Qdrant/bge-small-en-v1.5-onnx-Q | fp16 | 0.7127 | 0.6822 | 0.9417 | 1.00000 |
| sentence-transformers/all-MiniLM-L6-v2 | Qdrant/all-MiniLM-L6-v2-onnx | none | 0.6541 | 0.6071 | 0.9317 | 1.00000 |
| sentence-transformers/all-MiniLM-L6-v2 | Xenova/all-MiniLM-L6-v2 | dynamic | 0.6519 | 0.6077 | 0.9367 | 0.99336 |

## Run metadata

| Field | Value |
|---|---|
| git commit | 81492b1 (dirty) |
| timestamp | 2026-11-22T12:35:17Z |
| libembedding_version | 1.11.2 |
| python | 3.12.10 |
| machine | AMD64 |
| cpu_model | Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz |
| physical_cores | 4 |
| logical_cores | 8 |
| ram_mb | 16124 |
| cpu_features | AVX,FMA,AVX2,AVX512F |
| os | Windows-11-10.0.26300-SP0 |
| onnxruntime_version | 1.31.0 |
| llama_cpp_build | llama.cpp enabled |
| ggml_cpu_features | SSE3 SSSE3 AVX AVX2 F16C FMA BMI2 AVX512 AVX512_VNNI LLAMAFILE OPENMP REPACK |
| threads | 0 |
| batch | 32 |
| warmup | 0 |
| iterations | 1 |
| format | onnx |
| quantization | all registry variants |
| dataset | BEIR SciFact test (5183 docs, 300 queries) |
