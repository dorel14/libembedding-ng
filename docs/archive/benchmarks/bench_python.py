"""
Python bindings benchmark for libembedding.
Tests throughput with different models and configurations.
"""

import time
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from libembedding import TextEmbedding
import numpy as np

# Short corpus for fair comparison with C++
CORPUS = [
    "Hello world.",
    "Bonjour le monde.",
    "Hallo Welt.",
    "Hola mundo.",
]


def benchmark_model(model_name, threads, batch_size, n_iterations=5):
    """Benchmark a model configuration."""
    print(f"  Loading {model_name} (threads={threads}, batch={batch_size})...")

    try:
        model = TextEmbedding(
            model_name,
            threads=threads,
            batch_size=batch_size,
            offline=True,
            show_download_progress=False,
        )
    except Exception as e:
        print(f"    SKIP: {e}")
        return None

    dim = model.dim
    print(f"    Dimension: {dim}")

    # Warmup
    model.embed(CORPUS[:4])

    # Benchmark
    times = []
    for i in range(n_iterations):
        t0 = time.perf_counter()
        result = model.embed(CORPUS)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)
    n_texts = len(CORPUS)
    docs_per_sec = n_texts / mean_time

    print(f"    {n_texts} texts: {mean_time*1000:.1f}±{std_time*1000:.1f} ms = {docs_per_sec:.0f} docs/s")

    model.close()
    return {
        'model': model_name,
        'threads': threads,
        'batch_size': batch_size,
        'dim': dim,
        'mean_ms': mean_time * 1000,
        'std_ms': std_time * 1000,
        'docs_per_sec': docs_per_sec,
    }


def main():
    print("=" * 70)
    print("Python Bindings Benchmark - libembedding")
    print("=" * 70)
    print(f"Corpus: {len(CORPUS)} texts, multilingual, various lengths")
    print()

    configs = [
        # (model_name, threads, batch_size)
        ("sentence-transformers/all-MiniLM-L6-v2", 1, 64),
        ("sentence-transformers/all-MiniLM-L6-v2", 4, 64),
        ("BAAI/bge-small-en-v1.5", 1, 64),
        ("BAAI/bge-small-en-v1.5", 4, 64),
    ]

    results = []
    for model_name, threads, batch_size in configs:
        result = benchmark_model(model_name, threads, batch_size, n_iterations=5)
        if result:
            results.append(result)
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Threads':<8} {'Batch':<8} {'Docs/s':<10} {'ms/text':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<35} {r['threads']:<8} {r['batch_size']:<8} "
              f"{r['docs_per_sec']:<10.0f} {r['mean_ms']/len(CORPUS):<10.1f}")
    print()

    # Test edge cases
    print("=" * 70)
    print("Edge Cases")
    print("=" * 70)

    model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        threads=1,
        offline=True,
        show_download_progress=False,
    )

    edge_cases = [
        ("Empty string", ""),
        ("Single char", "a"),
        ("Whitespace", "   "),
        ("Numbers", "12345 67890"),
        ("Special chars", "!@#$%^&*()"),
        ("Unicode emoji", "Hello 😀 world"),
        ("Cyrillic", "Привет мир"),
        ("Chinese", "你好世界"),
        ("Japanese", "こんにちは"),
        ("Korean", "안녕하세요"),
        ("Very long", "word " * 100),
    ]

    for name, text in edge_cases:
        try:
            result = model.embed([text])
            # Check for NaN
            has_nan = np.any(np.isnan(result)) or np.any(np.isinf(result))
            status = "FAIL" if has_nan else "OK"
            print(f"  {name:<20} {status}")
        except Exception as e:
            print(f"  {name:<20} ERROR: {e}")

    model.close()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
