"""
Test autotune with cache - uses cache automatically
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import TextEmbedding, TextEmbeddingPool
import time

print("=== Autotune with Cache Test ===\n")

# Generate test texts
texts = [f"This is test sentence number {i} for benchmarking with various lengths." for i in range(100)]

# Test 1: TextEmbedding with autotune=True
print("--- TextEmbedding(autotune=True) ---")
print("First call (cache miss + autotune):")
t0 = time.perf_counter()
model1 = TextEmbedding("Qdrant/all-MiniLM-L6-v2-onnx", autotune=True, offline=True, show_download_progress=False)
t1 = time.perf_counter()
print(f"  Init time: {(t1-t0)*1000:.0f} ms")
print(f"  Threads: {model1._threads}, batch_size: {model1.batch_size}")

print("Second call (cache hit, instant):")
t0 = time.perf_counter()
model2 = TextEmbedding("Qdrant/all-MiniLM-L6-v2-onnx", autotune=True, offline=True, show_download_progress=False)
t1 = time.perf_counter()
print(f"  Init time: {(t1-t0)*1000:.0f} ms")
print(f"  Threads: {model2._threads}, batch_size: {model2.batch_size}")

# Benchmark
result1 = model1.embed(texts)
result2 = model2.embed(texts)
print(f"  Embeddings shape: {result1.shape}, {result2.shape}")

model1.close()
model2.close()

print()

# Test 2: TextEmbeddingPool with autotune=True
print("--- TextEmbeddingPool(autotune=True) ---")
print("First call (cache miss + autotune):")
t0 = time.perf_counter()
pool1 = TextEmbeddingPool("Qdrant/all-MiniLM-L6-v2-onnx", autotune=True, offline=True, show_download_progress=False)
t1 = time.perf_counter()
print(f"  Init time: {(t1-t0)*1000:.0f} ms")
print(f"  Workers: {pool1.num_workers}")

print("Second call (cache hit, instant):")
t0 = time.perf_counter()
pool2 = TextEmbeddingPool("Qdrant/all-MiniLM-L6-v2-onnx", autotune=True, offline=True, show_download_progress=False)
t1 = time.perf_counter()
print(f"  Init time: {(t1-t0)*1000:.0f} ms")
print(f"  Workers: {pool2.num_workers}")

# Benchmark
t0 = time.perf_counter()
embeddings = pool2.embed(texts)
t1 = time.perf_counter()
docs_per_sec = len(texts) / (t1 - t0)
print(f"  {len(texts)} texts in {(t1-t0)*1000:.0f} ms = {docs_per_sec:.0f} docs/s")

pool1.close()
pool2.close()

print("\nDone.")
