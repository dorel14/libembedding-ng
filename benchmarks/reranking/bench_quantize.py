"""
Quantization benchmark: Jina-v1-turbo FP32 vs INT8.
Measures latency, throughput, RAM, and score correlation.
"""
import argparse
import os
import sys
import time

# Patch cffi
import cffi

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import Reranker


def generate_doc_with_length(target_tokens, seed=42):
    import random
    random.seed(seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "data", "model", "system", "algorithm",
        "search", "query", "document", "text", "information", "result",
        "process", "analysis", "method", "approach", "technique",
        "application", "performance", "evaluation", "research", "study",
    ]
    target_words = max(1, int(target_tokens * 0.75))
    doc_words = []
    while len(doc_words) < target_words:
        doc_words.append(random.choice(words))
    return " ".join(doc_words[:target_words])


def percentile(values, p):
    s = sorted(values)
    idx = int(p / 100.0 * len(s))
    return s[min(idx, len(s) - 1)]


def benchmark_model(model_name, docs, query, threads, batch_size, warmup, iterations):
    """Benchmark a reranker model and return timing stats."""
    import subprocess

    # Measure RSS before
    pid = os.getpid()
    result = subprocess.run(
        ['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'WorkingSetSize'],
        capture_output=True, text=True, timeout=5, check=False)
    rss_before = 0
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line.isdigit():
            rss_before = int(line) / (1024 * 1024)
            break

    # Load model
    t0 = time.perf_counter()
    reranker = Reranker(
        model_name,
        threads=threads,
        batch_size=batch_size,
        offline=True,
        show_download_progress=False,
    )
    load_time = (time.perf_counter() - t0) * 1000

    # Measure RSS after load
    result = subprocess.run(
        ['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'WorkingSetSize'],
        capture_output=True, text=True, timeout=5, check=False)
    rss_after_load = 0
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line.isdigit():
            rss_after_load = int(line) / (1024 * 1024)
            break

    # Warmup
    for _ in range(warmup):
        reranker.rerank(query, docs, batch_size=batch_size)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        reranker.rerank(query, docs, batch_size=batch_size)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    # Get sample scores for quality comparison
    sample_scores = reranker.rerank(query, docs[:5], batch_size=batch_size)

    reranker.close()

    p50 = percentile(times, 50)
    p95 = percentile(times, 95)
    p99 = percentile(times, 99)
    mean = sum(times) / len(times)

    return {
        'p50': p50,
        'p95': p95,
        'p99': p99,
        'mean': mean,
        'ms_doc': p50 / len(docs),
        'load_ms': load_time,
        'rss_before': rss_before,
        'rss_after': rss_after_load,
        'rss_delta': rss_after_load - rss_before,
        'scores': [s.score for s in sample_scores],
    }


def main():
    parser = argparse.ArgumentParser(description='Jina FP32 vs INT8 quantization benchmark')
    parser.add_argument('--docs', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()

    query = "What is deep learning?"
    docs = [generate_doc_with_length(128, seed=i) for i in range(args.docs)]

    print("=" * 80)
    print("Quantization Benchmark: Jina-v1-turbo FP32 vs INT8")
    print("=" * 80)
    print(f"Docs: {args.docs} (~128 tokens each)")
    print(f"Threads: {args.threads}, Batch: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # FP32
    print("--- FP32 ---")
    fp32 = benchmark_model(
        "jinaai/jina-reranker-v1-turbo-en",
        docs, query, args.threads, args.batch_size, args.warmup, args.iterations
    )
    print(f"  P50: {fp32['p50']:.1f} ms ({fp32['ms_doc']:.1f} ms/doc)")
    print(f"  P95: {fp32['p95']:.1f} ms")
    print(f"  P99: {fp32['p99']:.1f} ms")
    print(f"  Load time: {fp32['load_ms']:.0f} ms")
    print(f"  RAM delta: {fp32['rss_delta']:.0f} MB")
    print()

    # INT8
    print("--- INT8 (quantized) ---")
    int8 = benchmark_model(
        "jinaai/jina-reranker-v1-turbo-en-quantized",
        docs, query, args.threads, args.batch_size, args.warmup, args.iterations
    )
    print(f"  P50: {int8['p50']:.1f} ms ({int8['ms_doc']:.1f} ms/doc)")
    print(f"  P95: {int8['p95']:.1f} ms")
    print(f"  P99: {int8['p99']:.1f} ms")
    print(f"  Load time: {int8['load_ms']:.0f} ms")
    print(f"  RAM delta: {int8['rss_delta']:.0f} MB")
    print()

    # Score correlation
    print("--- Score Correlation ---")
    if fp32['scores'] and int8['scores']:
        import math
        # Compute Pearson correlation
        n = min(len(fp32['scores']), len(int8['scores']))
        fp32_s = fp32['scores'][:n]
        int8_s = int8['scores'][:n]
        mean_fp32 = sum(fp32_s) / n
        mean_int8 = sum(int8_s) / n
        cov = sum((fp32_s[i] - mean_fp32) * (int8_s[i] - mean_int8) for i in range(n)) / n
        std_fp32 = math.sqrt(sum((s - mean_fp32)**2 for s in fp32_s) / n)
        std_int8 = math.sqrt(sum((s - mean_int8)**2 for s in int8_s) / n)
        corr = cov / (std_fp32 * std_int8) if std_fp32 > 0 and std_int8 > 0 else 0
        print(f"  Pearson correlation: {corr:.4f}")
        print(f"  FP32 scores: {[f'{s:.3f}' for s in fp32_s]}")
        print(f"  INT8 scores: {[f'{s:.3f}' for s in int8_s]}")
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Metric':<20} | {'FP32':>10} | {'INT8':>10} | {'Speedup':>10}")
    print("-" * 55)
    print(f"{'P50 (ms)':<20} | {fp32['p50']:>10.1f} | {int8['p50']:>10.1f} | {fp32['p50']/int8['p50']:>9.2f}x")
    print(f"{'P95 (ms)':<20} | {fp32['p95']:>10.1f} | {int8['p95']:>10.1f} | {fp32['p95']/int8['p95']:>9.2f}x")
    print(f"{'P99 (ms)':<20} | {fp32['p99']:>10.1f} | {int8['p99']:>10.1f} | {fp32['p99']/int8['p99']:>9.2f}x")
    print(f"{'ms/doc':<20} | {fp32['ms_doc']:>10.1f} | {int8['ms_doc']:>10.1f} | {fp32['ms_doc']/int8['ms_doc']:>9.2f}x")
    print(f"{'RAM delta (MB)':<20} | {fp32['rss_delta']:>10.0f} | {int8['rss_delta']:>10.0f} | {fp32['rss_delta']/int8['rss_delta'] if int8['rss_delta'] > 0 else 0:>9.2f}x")
    print()

    if int8['p50'] < fp32['p50'] * 0.9:
        print("=> INT8 is FASTER than FP32. Quantization is beneficial.")
    elif int8['p50'] > fp32['p50'] * 1.1:
        print("=> INT8 is SLOWER than FP32. Quantization is NOT beneficial.")
    else:
        print("=> INT8 and FP32 are comparable. Quantization has minimal impact on speed.")
    print()


if __name__ == '__main__':
    main()
