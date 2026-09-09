"""
Test autotune from Python
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import autotune, TextEmbeddingPool
import time

print("=== Python Autotune Test ===\n")

print("Running autotune (QUICK mode)...")
result = autotune("Qdrant/all-MiniLM-L6-v2-onnx")

print(f"\nOptimal configuration:")
print(f"  workers:    {result.workers}")
print(f"  threads:    {result.threads}")
print(f"  batch_size: {result.batch_size}")
print(f"  throughput: {result.throughput_docs_sec:.0f} docs/s")
print(f"  latency:    {result.latency_ms:.1f} ms/text")

# Test with the optimal config
print(f"\nTesting with optimal config...")
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
    offline=True,
    show_download_progress=False,
)

# Generate test texts
texts = [f"This is test sentence number {i} for benchmarking." for i in range(100)]

# Warmup
pool.embed(texts[:10])

# Benchmark
t0 = time.perf_counter()
embeddings = pool.embed(texts)
t1 = time.perf_counter()

elapsed = t1 - t0
docs_per_sec = len(texts) / elapsed

print(f"  {len(texts)} texts in {elapsed*1000:.0f} ms = {docs_per_sec:.0f} docs/s")
print(f"  Embeddings shape: {embeddings.shape}")

pool.close()
print("\nDone.")
