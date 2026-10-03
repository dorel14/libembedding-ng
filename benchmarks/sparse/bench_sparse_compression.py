"""
Sparse embedding compression benchmark.
Compares dict[int, float] vs CSR vs numpy sparse formats.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python/src"))

import numpy as np


def benchmark_format(name: str, format_func, vectors: list, n_runs: int = 5) -> dict:
    """Benchmark a specific sparse format."""
    results = {
        "format": name,
        "encode_times": [],
        "access_times": [],
        "sizes": [],
    }

    for _ in range(n_runs):
        # Encode time
        start = time.perf_counter()
        encoded = [format_func(v) for v in vectors]
        encode_time = time.perf_counter() - start
        results["encode_times"].append(encode_time * 1000 / len(vectors))

        # Access time (random access)
        start = time.perf_counter()
        for i in range(0, len(encoded), max(1, len(encoded) // 100)):
            _ = encoded[i]
        access_time = time.perf_counter() - start
        results["access_times"].append(access_time * 1000)

        # Size
        size = sys.getsizeof(encoded)
        results["sizes"].append(size / 1024)

    import statistics
    return {
        "format": name,
        "encode_ms_per_vec": statistics.mean(results["encode_times"]),
        "access_ms": statistics.mean(results["access_times"]),
        "size_kb": statistics.mean(results["sizes"]),
    }


def format_dict(vector: dict[int, float]) -> dict[int, float]:
    """Python dict format."""
    return dict(vector)


def format_csr(vector: dict[int, float]) -> tuple:
    """CSR format: (indices, values)."""
    indices = list(vector.keys())
    values = list(vector.values())
    return (indices, values)


def format_numpy(vector: dict[int, float]) -> any:
    """NumPy sparse COO format."""
    if not vector:
        return None
    indices = list(vector.keys())
    values = list(vector.values())
    return indices, values  # Simplified COO representation


def main():
    # Generate sample sparse vectors
    np.random.seed(42)
    vectors = []
    for _ in range(1000):
        # Random sparse vector: 50-150 non-zero terms out of 30000
        nnz = np.random.randint(50, 150)
        indices = np.random.choice(30000, nnz, replace=False).tolist()
        values = np.random.randn(nnz).tolist()
        vectors.append(dict(zip(indices, values)))

    print("Benchmarking sparse formats...")
    print(f"Vectors: {len(vectors)}, avg nnz: {sum(len(v) for v in vectors) / len(vectors):.0f}")
    print()

    formats = [
        ("dict[int, float]", format_dict),
        ("CSR (indices, values)", format_csr),
        ("numpy (indices, values)", format_numpy),
    ]

    results = []
    for name, func in formats:
        try:
            result = benchmark_format(name, func, vectors)
            results.append(result)
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"ERROR benchmarking {name}: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("COMPRESSION BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Format':<30} {'Encode (ms/vec)':<18} {'Access (ms)':<15} {'Size (KB)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['format']:<30} {r['encode_ms_per_vec']:<18.4f} {r['access_ms']:<15.4f} {r['size_kb']:<12.1f}")


if __name__ == "__main__":
    main()
