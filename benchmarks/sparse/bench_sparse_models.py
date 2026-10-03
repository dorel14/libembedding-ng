"""
Sparse embedding model comparison benchmark.
Measures: loading time, inference time, memory, avg non-zero terms, output size.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python/src"))

from libembedding import SparseTextEmbedding


def benchmark_model(model_name: str, texts: list[str], n_runs: int = 5) -> dict:
    """Benchmark a sparse embedding model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Loading time
    start = time.perf_counter()
    model = SparseTextEmbedding(model_name)
    load_time = time.perf_counter() - start
    print(f"Loading time: {load_time*1000:.1f} ms")

    # Warmup
    model.embed(texts[:10])

    results = {
        "model": model_name,
        "loading_time_ms": load_time * 1000,
        "inference_times": [],
        "memories": [],
        "avg_sparse_terms": [],
        "output_sizes": [],
    }

    for run in range(n_runs):
        # Inference time + memory
        tracemalloc.start()
        start = time.perf_counter()
        result = model.embed(texts)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results["inference_times"].append(elapsed * 1000 / len(texts))
        results["memories"].append(peak / 1024 / 1024)

        # Sparse stats
        total_terms = sum(len(v.indices) if hasattr(v, 'indices') else len(v) for v in result)
        avg_terms = total_terms / len(texts)
        results["avg_sparse_terms"].append(avg_terms)

        # Output size (approximate)
        output_size = sum(len(v.indices) * 8 if hasattr(v, 'indices') else len(v) * 8 for v in result)
        results["output_sizes"].append(output_size / 1024)

    # Aggregate results
    import statistics
    results["inference_ms_per_doc"] = statistics.mean(results["inference_times"])
    results["inference_std"] = statistics.stdev(results["inference_times"]) if n_runs > 1 else 0
    results["memory_mb"] = statistics.mean(results["memories"])
    results["avg_terms"] = statistics.mean(results["avg_sparse_terms"])
    results["output_kb"] = statistics.mean(results["output_sizes"])

    print(f"Inference: {results['inference_ms_per_doc']:.2f} ± {results['inference_std']:.2f} ms/doc")
    print(f"Memory: {results['memory_mb']:.1f} MB")
    print(f"Avg sparse terms: {results['avg_terms']:.1f}")
    print(f"Output size: {results['output_kb']:.1f} KB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sparse embedding model comparison")
    parser.add_argument("--models", nargs="+", default=["Qdrant/splade-vit-b16-mpnet", "BAAI/bge-m3"])
    parser.add_argument("--texts", type=str, default="")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    # Generate sample texts
    if args.texts:
        with open(args.texts) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = [
            "What is the capital of France?",
            "How does a neural network work?",
            "Explain the theory of relativity.",
            "What is machine learning?",
            "Describe the water cycle.",
            "What causes earthquakes?",
            "How do vaccines work?",
            "What is quantum computing?",
            "Explain photosynthesis.",
            "What is the speed of light?",
        ] * 10  # 100 texts

    print(f"Benchmarking {len(texts)} texts, {args.runs} runs each")

    all_results = []
    for model_name in args.models:
        try:
            result = benchmark_model(model_name, texts, args.runs)
            all_results.append(result)
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"ERROR benchmarking {model_name}: {e}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Load (ms)':<12} {'Inf (ms/doc)':<15} {'Mem (MB)':<12} {'Terms':<10} {'Out (KB)':<12}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<30} {r['loading_time_ms']:<12.1f} {r['inference_ms_per_doc']:<15.2f} {r['memory_mb']:<12.1f} {r['avg_terms']:<10.1f} {r['output_kb']:<12.1f}")


if __name__ == "__main__":
    main()
