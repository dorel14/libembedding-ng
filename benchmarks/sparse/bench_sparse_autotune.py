"""
Sparse auto-tuning benchmark.
Finds optimal: pruning_threshold, top_k, quantization, storage_format.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python/src"))

from libembedding import SparseTextEmbedding


def benchmark_config(model_name: str, texts: list[str],
                     pruning_threshold: float, top_k: int,
                     n_runs: int = 3) -> dict:
    """Benchmark a specific sparse configuration."""
    model = SparseTextEmbedding(model_name)

    times = []
    sizes = []
    for _ in range(n_runs):
        start = time.perf_counter()
        results = []
        for text in texts:
            # Apply pruning and top_k
            vec = model.embed([text])[0]
            if hasattr(vec, 'indices'):
                # Filter by threshold
                filtered = [(i, v) for i, v in zip(vec.indices, vec.values) if abs(v) >= pruning_threshold]
                # Top-k by absolute value
                filtered.sort(key=lambda x: abs(x[1]), reverse=True)
                filtered = filtered[:top_k]
                size = len(filtered) * 8  # approximate bytes
            else:
                filtered = [(idx, val) for idx, val in vec.items() if abs(val) >= pruning_threshold]
                filtered.sort(key=lambda x: abs(x[1]), reverse=True)
                filtered = filtered[:top_k]
                size = len(filtered) * 8
            results.append(filtered)
            sizes.append(size)

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000 / len(texts))

    import statistics
    return {
        "pruning_threshold": pruning_threshold,
        "top_k": top_k,
        "latency_ms_per_doc": statistics.mean(times),
        "avg_output_size_bytes": statistics.mean(sizes),
    }


def main():
    parser = argparse.ArgumentParser(description="Sparse auto-tuning")
    parser.add_argument("--model", type=str, default="Qdrant/splade-vit-b16-mpnet")
    parser.add_argument("--texts", type=int, default=100)
    args = parser.parse_args()

    texts = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning.",
        "What is NLP?",
        "What is computer vision?",
        "How do transformers work?",
        "What is BERT?",
        "Explain GPT models.",
        "What are embeddings?",
        "Describe sparse vectors.",
    ] * (args.texts // 10)

    print(f"Auto-tuning sparse config for {args.model}")
    print(f"Corpus: {len(texts)} texts")
    print()

    # Parameter grid
    thresholds = [0.0, 0.01, 0.05, 0.1]
    top_ks = [32, 64, 128, 256]

    best_score = float('inf')
    best_config = None
    all_results = []

    for threshold in thresholds:
        for top_k in top_ks:
            result = benchmark_config(args.model, texts, threshold, top_k)
            result["score"] = result["latency_ms_per_doc"] + result["avg_output_size_bytes"] * 0.001
            all_results.append(result)

            if result["score"] < best_score:
                best_score = result["score"]
                best_config = result

            print(f"  threshold={threshold:.2f}, top_k={top_k:3d}: "
                  f"{result['latency_ms_per_doc']:.2f} ms/doc, "
                  f"{result['avg_output_size_bytes']:.0f} bytes → score={result['score']:.2f}")

    # Best config
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"pruning_threshold: {best_config['pruning_threshold']}")
    print(f"top_k: {best_config['top_k']}")
    print(f"latency: {best_config['latency_ms_per_doc']:.2f} ms/doc")
    print(f"output_size: {best_config['avg_output_size_bytes']:.0f} bytes")

    # Save to cache
    cache_dir = Path.home() / ".cache" / "libembedding" / "sparse_autotune"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{args.model.replace('/', '_')}.json"
    with open(cache_file, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nCached to: {cache_file}")


if __name__ == "__main__":
    import argparse
    main()
