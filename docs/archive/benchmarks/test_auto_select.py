"""
Test auto model selection from Python
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import auto_select_model, TextEmbeddingPool
import time

print("=== Auto Model Selection Test ===\n")

for use_case in ["speed", "quality", "balanced"]:
    print(f"\n--- Use case: {use_case} ---")
    result = auto_select_model(use_case)

    print(f"\n  Best model: {result.model_name}")
    print(f"  Model code: {result.model_code}")
    print(f"  Dimension:  {result.dim}")
    print(f"  Config:     {result.workers} workers × {result.threads} threads, batch={result.batch_size}")
    print(f"  Throughput: {result.throughput_docs_sec:.0f} docs/s")
    print(f"  Latency:    {result.latency_ms:.1f} ms/text")
    print(f"  Score:      {result.score:.1f}")

print("\n\n=== Testing selected model ===\n")

# Use balanced mode for testing
result = auto_select_model("balanced")
print(f"Selected: {result.model_name}")

# Create pool with optimal config
pool = TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
    offline=True,
    show_download_progress=False,
)

# Generate test texts (various lengths)
texts = []
for i in range(100):
    length = 16 + (i % 5) * 32  # 16, 48, 80, 112, 144 tokens
    words = [f"word{j}" for j in range(length)]
    texts.append(" ".join(words))

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
