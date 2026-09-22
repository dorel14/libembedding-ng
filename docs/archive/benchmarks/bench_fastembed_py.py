#!/usr/bin/env python3
"""
fastembed (Python) benchmark harness.
Measures model load time, single-text latency, and batch throughput.
Outputs JSON to stdout for comparison with libembedding and fastembed-rs.
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
    # macOS: ru_maxrss is in bytes; Linux: in KB
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def load_corpus(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
    corpus = load_corpus(corpus_path)
    if not corpus:
        print(f"Failed to load corpus from {corpus_path}", file=sys.stderr)
        sys.exit(1)

    # Import after arg parsing to avoid measuring import time
    from fastembed import TextEmbedding

    WARMUP = 1
    ITERS = 10
    BATCH_SIZES = [1, 8, 32, 128, 512]

    # ── Benchmark A: Model load time ──────────────────────────────
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

    # Create persistent model for remaining benchmarks
    model = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threads=1,
    )
    rss_after_load = peak_rss_mb()

    # ── Benchmark B: Single text latency (1 thread) ──────────────
    single_times = []
    single_text = ["The quick brown fox jumps over the lazy dog."]

    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        list(model.embed(single_text, batch_size=1))
        t1 = time.perf_counter()
        if i >= WARMUP:
            single_times.append((t1 - t0) * 1000)

    # ── Benchmark C: Batch throughput (auto threads) ──────────────
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

    # ── Output JSON ───────────────────────────────────────────────
    result = {
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
