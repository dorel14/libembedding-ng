#!/usr/bin/env python3
"""
Benchmark: libembedding (Python bindings) vs fastembed (Python)
Same model, same corpus, same methodology.
"""

import json
import math
import os
import resource
import sys
import time


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def p95(v):
    if not v:
        return 0.0
    s = sorted(v)
    idx = int(math.ceil(0.95 * len(s))) - 1
    return s[min(idx, len(s) - 1)]


def peak_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def load_corpus(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def bench_libembedding(corpus):
    """Benchmark libembedding Python bindings."""
    # Add the Python package to path
    pkg_dir = os.path.join(os.path.dirname(__file__), "..", "python", "src")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    from libembedding import TextEmbedding

    WARMUP = 1
    ITERS = 10
    BATCH_SIZES = [1, 8, 32, 128, 512]

    # Model load
    load_times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        model = TextEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2",
            threads=1,
            show_download_progress=False,
        )
        t1 = time.perf_counter()
        if i >= WARMUP:
            load_times.append((t1 - t0) * 1000)
        model.close()

    model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        threads=1,
        show_download_progress=False,
    )
    rss_after_load = peak_rss_mb()

    # Single latency
    single_times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        model.embed(["The quick brown fox jumps over the lazy dog."])
        t1 = time.perf_counter()
        if i >= WARMUP:
            single_times.append((t1 - t0) * 1000)

    # Batch throughput (auto threads)
    model.close()
    model = TextEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        show_download_progress=False,
    )

    batch_throughput = {}
    for bsz in BATCH_SIZES:
        n = min(bsz, len(corpus))
        batch_texts = corpus[:n]
        samples = []
        for i in range(WARMUP + ITERS):
            t0 = time.perf_counter()
            model.embed(batch_texts, batch_size=bsz)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            if i >= WARMUP and elapsed > 0:
                samples.append(n / elapsed)
        batch_throughput[str(bsz)] = {"median_texts_per_sec": round(median(samples), 1)}

    rss_peak = peak_rss_mb()
    model.close()

    return {
        "library": "libembedding-py",
        "model": "all-MiniLM-L6-v2",
        "platform": "macOS arm64",
        "benchmarks": {
            "model_load_ms": {"median": round(median(load_times), 2), "p95": round(p95(load_times), 2)},
            "single_latency_ms": {"median": round(median(single_times), 2), "p95": round(p95(single_times), 2)},
            "batch_throughput": batch_throughput,
            "memory": {"after_load_rss_mb": round(rss_after_load, 1), "peak_rss_mb": round(rss_peak, 1)},
        },
    }


def bench_fastembed(corpus):
    """Benchmark fastembed (Python)."""
    from fastembed import TextEmbedding

    WARMUP = 1
    ITERS = 10
    BATCH_SIZES = [1, 8, 32, 128, 512]

    # Model load
    load_times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            threads=1,
        )
        t1 = time.perf_counter()
        if i >= WARMUP:
            load_times.append((t1 - t0) * 1000)
        del model

    model = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threads=1,
    )
    rss_after_load = peak_rss_mb()

    # Single latency
    single_times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        list(model.embed(["The quick brown fox jumps over the lazy dog."], batch_size=1))
        t1 = time.perf_counter()
        if i >= WARMUP:
            single_times.append((t1 - t0) * 1000)

    # Batch throughput (auto threads)
    del model
    model = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    batch_throughput = {}
    for bsz in BATCH_SIZES:
        n = min(bsz, len(corpus))
        batch_texts = corpus[:n]
        samples = []
        for i in range(WARMUP + ITERS):
            t0 = time.perf_counter()
            list(model.embed(batch_texts, batch_size=bsz))
            t1 = time.perf_counter()
            elapsed = t1 - t0
            if i >= WARMUP and elapsed > 0:
                samples.append(n / elapsed)
        batch_throughput[str(bsz)] = {"median_texts_per_sec": round(median(samples), 1)}

    rss_peak = peak_rss_mb()
    del model

    return {
        "library": "fastembed-py",
        "model": "all-MiniLM-L6-v2",
        "platform": "macOS arm64",
        "benchmarks": {
            "model_load_ms": {"median": round(median(load_times), 2), "p95": round(p95(load_times), 2)},
            "single_latency_ms": {"median": round(median(single_times), 2), "p95": round(p95(single_times), 2)},
            "batch_throughput": batch_throughput,
            "memory": {"after_load_rss_mb": round(rss_after_load, 1), "peak_rss_mb": round(rss_peak, 1)},
        },
    }


def print_comparison(r1, r2):
    """Print side-by-side comparison."""
    b1 = r1["benchmarks"]
    b2 = r2["benchmarks"]

    print(f"\n{'Metric':<28} {'libembedding-py':>16} {'fastembed-py':>16} {'Speedup':>10}")
    print("-" * 72)

    v1, v2 = b1["model_load_ms"]["median"], b2["model_load_ms"]["median"]
    print(f"{'Model load (ms)':<28} {v1:>14.1f}ms {v2:>14.1f}ms {v2/v1:>9.1f}x")

    v1, v2 = b1["single_latency_ms"]["median"], b2["single_latency_ms"]["median"]
    print(f"{'Single latency (ms)':<28} {v1:>14.1f}ms {v2:>14.1f}ms {v2/v1:>9.1f}x")

    for bsz in ["1", "8", "32", "128", "512"]:
        v1 = b1["batch_throughput"][bsz]["median_texts_per_sec"]
        v2 = b2["batch_throughput"][bsz]["median_texts_per_sec"]
        ratio = v1 / v2 if v2 > 0 else 0
        print(f"{'Batch ' + bsz + ' (texts/s)':<28} {v1:>14.1f}   {v2:>14.1f}   {ratio:>9.1f}x")

    v1 = b1["memory"].get("peak_rss_mb", 0)
    v2 = b2["memory"].get("peak_rss_mb", 0)
    ratio = v2 / v1 if v1 > 0 else 0
    print(f"{'Peak RSS (MB)':<28} {v1:>14.1f}   {v2:>14.1f}   {ratio:>9.1f}x less")
    print()


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "corpus.txt"
    )
    corpus = load_corpus(corpus_path)
    if not corpus:
        print(f"Failed to load corpus from {corpus_path}", file=sys.stderr)
        sys.exit(1)

    print("=== libembedding (Python bindings) ===")
    r1 = bench_libembedding(corpus)
    print(json.dumps(r1, indent=2))

    print("\n=== fastembed (Python) ===")
    r2 = bench_fastembed(corpus)
    print(json.dumps(r2, indent=2))

    print_comparison(r1, r2)


if __name__ == "__main__":
    main()
