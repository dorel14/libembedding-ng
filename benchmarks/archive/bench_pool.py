"""
Benchmark TextEmbeddingPool - Python multi-worker embedding
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import TextEmbedding, TextEmbeddingPool
import numpy as np

# Multilingual corpus with various lengths
CORPUS = [
    "Hello world.",
    "Bonjour le monde.",
    "Hallo Welt.",
    "Hola mundo.",
    "Ciao mondo.",
    "Olá mundo.",
    "Привет мир.",
    "こんにちは世界。",
    "안녕하세요 세계.",
    "你好世界。",
    "Machine learning transforms data.",
    "L'intelligence artificielle transforme les données.",
    "Künstliche Intelligenz verändert die Welt.",
    "Climate change affects global weather.",
    "Quantum computing promises revolution.",
    "The history of ancient Rome spans centuries.",
    "Natural language processing is a subfield of linguistics and artificial intelligence.",
    "The transformer architecture has become the foundation for modern NLP systems.",
    "Deep learning is part of machine learning methods based on artificial networks.",
    "Le traitement automatique du langage naturel est un domaine de l'IA.",
    "Die künstliche Intelligenz verändert die Art und Weise wie wir arbeiten.",
    "El procesamiento del lenguaje natural es un campo de la informática.",
    "Machine learning algorithms identify patterns in large datasets automatically.",
    "Climate change affects global weather patterns significantly.",
    "The history of ancient Rome spans over a thousand years.",
    "Quantum computing promises to revolutionize cryptography.",
    "Les algorithmes d'apprentissage automatique identifient des motifs.",
    "Die künstliche Intelligenz ist ein Gebiet der Informatik.",
    "El aprendizaje automático permite a las computadoras aprender.",
    "Natural language processing enables computers to understand language.",
    "The transformer model uses self-attention efficiently.",
    "Deep neural networks learn hierarchical representations through layers.",
    "Artificial intelligence has made significant progress in recent years.",
    "The development of modern AI began in the nineteen fifties.",
    "L'intelligence artificielle a fait des progrès significatifs.",
    "Die künstliche Intelligenz hat bedeutende Fortschritte gemacht.",
    "El inteligencia artificial ha hecho progresos significativos.",
]


def benchmark_single(threads, n_iter=5):
    """Benchmark single TextEmbedding session."""
    print(f"  Single session (threads={threads})...")

    model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        threads=threads,
        batch_size=64,
        offline=True,
        show_download_progress=False,
    )

    # Warmup
    model.embed(CORPUS[:4])

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = model.embed(CORPUS)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = np.mean(times)
    docs_per_sec = len(CORPUS) / mean_time

    print(f"    {len(CORPUS)} texts: {mean_time*1000:.1f} ms = {docs_per_sec:.0f} docs/s")

    model.close()
    return docs_per_sec


def benchmark_pool(workers, threads_per_worker, n_iter=5):
    """Benchmark TextEmbeddingPool."""
    print(f"  Pool ({workers} workers × {threads_per_worker} threads)...")

    pool = TextEmbeddingPool(
        "sentence-transformers/all-MiniLM-L6-v2",
        workers=workers,
        threads_per_worker=threads_per_worker,
        batch_size=64,
        offline=True,
        show_download_progress=False,
    )

    # Warmup
    pool.embed(CORPUS[:4])

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = pool.embed(CORPUS)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = np.mean(times)
    docs_per_sec = len(CORPUS) / mean_time

    print(f"    {len(CORPUS)} texts: {mean_time*1000:.1f} ms = {docs_per_sec:.0f} docs/s")

    pool.close()
    return docs_per_sec


def main():
    print("=" * 60)
    print("TextEmbeddingPool Benchmark")
    print("=" * 60)
    print(f"Corpus: {len(CORPUS)} texts, multilingual")
    print()

    results = []

    # Single session benchmarks
    print("--- Single Session ---")
    for threads in [1, 4]:
        tps = benchmark_single(threads)
        results.append((f"single_t{threads}", tps))
    print()

    # Pool benchmarks
    print("--- TextEmbeddingPool ---")
    for workers, threads in [(2, 1), (4, 1), (8, 1)]:
        tps = benchmark_pool(workers, threads)
        results.append((f"pool_{workers}x{threads}", tps))
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'Docs/s':<10} {'Speedup':<10}")
    print("-" * 40)
    baseline = results[0][1]
    for name, tps in results:
        speedup = tps / baseline
        print(f"{name:<20} {tps:<10.0f} {speedup:<10.1f}x")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
