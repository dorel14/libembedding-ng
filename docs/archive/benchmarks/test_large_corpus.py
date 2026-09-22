"""
Test autotune with large corpus (2M lines simulation)
Demonstrates stratified sampling for efficient benchmarking.
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import TextEmbedding, TextEmbeddingPool, clear_autotune_cache

print("=== Autotune with Large Corpus (2M lines) ===\n")

# Simulate a large corpus with varied lengths (like real RAG data)
def generate_large_corpus(n_lines: int) -> list[str]:
    """Generate a corpus with realistic length distribution."""
    short_queries = [
        "What is AI?",
        "How does BERT work?",
        "Explain transformers",
        "What is NLP?",
        "Define embedding",
    ]

    medium_texts = [
        "Machine learning algorithms can identify patterns in large datasets automatically without explicit programming instructions or human intervention.",
        "Natural language processing enables computers to understand human language through computational techniques and statistical models.",
        "Deep neural networks learn hierarchical representations of data through multiple layers of abstraction.",
    ]

    long_docs = [
        "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy. Large language models trained on vast amounts of text data have demonstrated capabilities that were previously thought to be decades away including reasoning summarization and creative writing.",
        "The transformer architecture introduced in the attention is all you need paper has become the foundation for most modern natural language processing systems. Unlike recurrent neural networks which process sequences sequentially transformers use self-attention mechanisms to process all positions simultaneously enabling much more parallelization and reducing training time significantly.",
    ]

    corpus = []
    # Distribution: 30% short, 50% medium, 20% long (typical RAG)
    for i in range(n_lines):
        r = random.random()
        if r < 0.3:
            corpus.append(random.choice(short_queries))
        elif r < 0.8:
            corpus.append(random.choice(medium_texts))
        else:
            corpus.append(random.choice(long_docs))

    return corpus

# Generate large corpus
n_lines = 2_000_000
print(f"Generating corpus of {n_lines:,} lines...")
t0 = time.perf_counter()
large_corpus = generate_large_corpus(n_lines)
t1 = time.perf_counter()
print(f"  Generated in {(t1-t0):.1f}s")
print(f"  Corpus size: {len(large_corpus):,} texts")
print(f"  Avg length: {sum(len(t.split()) for t in large_corpus) / len(large_corpus):.0f} tokens")
print()

# Clear cache
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")

# Test 1: Autotune with large corpus (sampling)
print("--- Autotune with 2M corpus (stratified sampling) ---")
print("  Sampling 100 representative texts...")
t0 = time.perf_counter()
model = TextEmbedding(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=large_corpus,
    autotune_max_samples=100,
    offline=True,
    show_download_progress=False,
)
t1 = time.perf_counter()
print(f"  Autotune time: {(t1-t0):.1f}s")
print(f"  Config: threads={model._threads}, batch_size={model.batch_size}")
model.close()

print()

# Test 2: TextEmbeddingPool with large corpus
print("--- TextEmbeddingPool with 2M corpus ---")
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")
t0 = time.perf_counter()
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=large_corpus,
    autotune_max_samples=100,
    offline=True,
    show_download_progress=False,
)
t1 = time.perf_counter()
print(f"  Autotune time: {(t1-t0):.1f}s")
print(f"  Workers: {pool.num_workers}")

# Benchmark on a sample of the large corpus
print()
print("--- Benchmark on 1000 texts from large corpus ---")
sample = large_corpus[:1000]
t0 = time.perf_counter()
embeddings = pool.embed(sample)
t1 = time.perf_counter()
docs_per_sec = len(sample) / (t1 - t0)
print(f"  {len(sample)} texts in {(t1-t0)*1000:.0f} ms = {docs_per_sec:.0f} docs/s")
print(f"  Embeddings shape: {embeddings.shape}")

pool.close()

print()
print("=== Summary ===")
print(f"  Corpus: {n_lines:,} texts")
print(f"  Sample for autotune: 100 texts (0.005% of corpus)")
print(f"  Autotune time: ~10-15s")
print(f"  Throughput: {docs_per_sec:.0f} docs/s")
print(f"  Time to embed full corpus: {n_lines / docs_per_sec / 60:.1f} minutes")

print("\nDone.")
