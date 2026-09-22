"""
Test autotune with custom corpus (user's actual texts)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import TextEmbedding, TextEmbeddingPool, clear_autotune_cache
import time

print("=== Autotune with Custom Corpus ===\n")

# Simulate user's actual corpus (e.g., RAG chunks)
user_corpus = [
    # Short queries
    "What is machine learning?",
    "How does BERT work?",
    "Explain transformers",
    # Medium documents
    "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with the interactions between computers and human language in particular how to program computers to process and analyze large amounts of natural language data.",
    "The transformer architecture introduced in the attention is all you need paper has become the foundation for most modern natural language processing systems including BERT GPT and their variants which have revolutionized the field.",
    # Long documents (RAG chunks)
    "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy. Large language models trained on vast amounts of text data have demonstrated capabilities that were previously thought to be decades away including reasoning summarization and creative writing that rivals human output in many domains.",
    "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think. Since then the field has gone through periods of optimism and disappointment known as AI winters. Today we are in a period of rapid advancement driven by increases in computational power the availability of large datasets and improvements in algorithms particularly deep learning.",
] * 10  # Repeat to get more samples

print(f"User corpus: {len(user_corpus)} texts")
print(f"Average length: {sum(len(t.split()) for t in user_corpus) / len(user_corpus):.0f} tokens")
print()

# Clear cache to test fresh autotune
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")

# Test 1: Autotune with SYNTHETIC corpus (default)
print("--- Autotune with synthetic corpus (default) ---")
t0 = time.perf_counter()
model1 = TextEmbedding(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    offline=True,
    show_download_progress=False,
)
t1 = time.perf_counter()
print(f"  Init time: {(t1-t0)*1000:.0f} ms")
print(f"  Config: threads={model1._threads}, batch_size={model1.batch_size}")
model1.close()

print()

# Clear cache again
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")

# Test 2: Autotune with USER'S corpus
print("--- Autotune with user's corpus ---")
t0 = time.perf_counter()
model2 = TextEmbedding(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=user_corpus,
    offline=True,
    show_download_progress=False,
)
t1 = time.perf_counter()
print(f"  Init time: {(t1-t0)*1000:.0f} ms")
print(f"  Config: threads={model2._threads}, batch_size={model2.batch_size}")
model2.close()

print()

# Test 3: TextEmbeddingPool with user's corpus
print("--- TextEmbeddingPool with user's corpus ---")
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")
t0 = time.perf_counter()
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=user_corpus,
    offline=True,
    show_download_progress=False,
)
t1 = time.perf_counter()
print(f"  Init time: {(t1-t0)*1000:.0f} ms")
print(f"  Workers: {pool.num_workers}")

# Benchmark with user's corpus
t0 = time.perf_counter()
embeddings = pool.embed(user_corpus)
t1 = time.perf_counter()
docs_per_sec = len(user_corpus) / (t1 - t0)
print(f"  {len(user_corpus)} texts in {(t1-t0)*1000:.0f} ms = {docs_per_sec:.0f} docs/s")

pool.close()

print("\nDone.")
