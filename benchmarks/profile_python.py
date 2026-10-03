"""
Profile Python overhead vs C++ overhead
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from libembedding import TextEmbedding

# Small corpus for profiling
CORPUS = [
    "Hello world",
    "Machine learning is great",
    "Natural language processing",
    "The quick brown fox jumps",
    "Artificial intelligence transforms",
    "Data science and analytics",
    "Cloud computing infrastructure",
    "Software engineering practices",
    "Computer vision applications",
    "Deep neural networks",
    "Big data processing",
    "Natural language understanding",
    "Reinforcement learning algorithms",
    "Generative adversarial networks",
    "Transformer architecture models",
    "Attention mechanism details",
    "Convolutional neural networks",
    "Recurrent neural networks",
    "Long short-term memory",
    "BERT pre-training",
    "GPT language model",
    "Text classification tasks",
    "Named entity recognition",
    "Sentiment analysis datasets",
    "Question answering systems",
    "Machine translation models",
    "Speech recognition systems",
    "Image classification tasks",
    "Object detection frameworks",
    "Semantic segmentation models",
    "Text summarization techniques",
    "Information retrieval systems",
    "Document embedding methods",
    "Similarity search algorithms",
    "Vector database indexing",
    "Approximate nearest neighbors",
    "Hierarchical navigable small world",
]

def profile_python_overhead(model, n_iter=100):
    """Profile the Python overhead separately from inference."""
    print(f"Profiling Python overhead ({len(CORPUS)} texts, {n_iter} iterations)...")

    # Warmup
    model.embed(CORPUS)

    # Full embed
    times_full = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.embed(CORPUS)
        t1 = time.perf_counter()
        times_full.append(t1 - t0)

    mean_full = np.mean(times_full) * 1000
    std_full = np.std(times_full) * 1000
    print(f"  Full embed: {mean_full:.2f} ± {std_full:.2f} ms")
    print(f"  Per text: {mean_full/len(CORPUS):.2f} ms")

    return mean_full


def main():
    print("=" * 60)
    print("Python Overhead Profiling")
    print("=" * 60)
    print()

    model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        threads=4,
        batch_size=64,
        offline=True,
        show_download_progress=False,
    )

    print(f"Dimension: {model.dim}")
    print(f"Corpus: {len(CORPUS)} texts")
    print()

    # Profile
    full_ms = profile_python_overhead(model, n_iter=50)

    print()
    print("Analysis:")
    print(f"  If C++ inference alone: ~{len(CORPUS) * 10:.0f} ms ({len(CORPUS)} texts × 10 ms)")
    print(f"  Python total: {full_ms:.0f} ms")
    print(f"  Python overhead: {full_ms - len(CORPUS) * 10:.0f} ms")
    print(f"  Overhead per text: {(full_ms - len(CORPUS) * 10) / len(CORPUS):.1f} ms")

    model.close()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
