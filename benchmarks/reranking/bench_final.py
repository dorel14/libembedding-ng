"""
Comprehensive reranking benchmark: FP32 vs INT8, all configurations.
Produces the final numbers for the investigation report.
"""
import argparse
import os
import subprocess
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


def generate_doc(target_tokens, seed=42):
    import random
    random.seed(seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "data", "model", "system", "algorithm",
        "search", "query", "document", "text", "information", "result",
        "process", "analysis", "method", "approach", "technique",
        "application", "performance", "evaluation", "research", "study",
        "experiment", "implementation", "development", "design", "architecture",
        "framework", "library", "tool", "service", "platform", "interface",
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


def get_rss_mb():
    try:
        pid = os.getpid()
        result = subprocess.run(
            ['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'WorkingSetSize'],
            capture_output=True, text=True, timeout=5, check=False)
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line.isdigit():
                return int(line) / (1024 * 1024)
    except (OSError, subprocess.SubprocessError) as exc:
        # wmic is absent from recent Windows builds. Report the failure rather
        # than returning a silent 0 MB, which reads as a real measurement.
        print(f"warning: RSS probe failed ({exc}); reporting 0 MB", file=sys.stderr)
    return 0


def benchmark_config(model_name, docs, query, threads, batch_size, warmup=2, iterations=20):
    """Benchmark a specific configuration."""
    rss_before = get_rss_mb()

    t0 = time.perf_counter()
    reranker = Reranker(
        model_name,
        threads=threads,
        batch_size=batch_size,
        offline=True,
        show_download_progress=False,
    )
    load_time = (time.perf_counter() - t0) * 1000

    rss_after = get_rss_mb()

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

    info = reranker.info()
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
        'rss_delta': rss_after - rss_before,
        'max_length': info.max_length,
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive reranking benchmark')
    parser.add_argument('--docs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()

    query = "What is deep learning?"

    print("=" * 90)
    print("COMPREHENSIVE RERANKING BENCHMARK")
    print("=" * 90)
    print(f"Docs: {args.docs}, Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # Configurations to test
    configs = [
        # (model_name, threads, batch_size, doc_tokens)
        ("jinaai/jina-reranker-v1-turbo-en", 4, 8, 128),           # FP32 baseline
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 4, 8, 128), # INT8 baseline
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 1, 8, 128), # INT8 1 thread
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 2, 8, 128), # INT8 2 threads
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 4, 4, 128), # INT8 batch=4
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 4, 16, 128),# INT8 batch=16
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 4, 8, 32),  # INT8 short docs
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 4, 8, 64),  # INT8 short docs
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 4, 8, 256), # INT8 medium docs
        ("jinaai/jina-reranker-v1-turbo-en-quantized", 4, 8, 512), # INT8 long docs
    ]

    results = []
    for model, threads, batch_size, doc_tokens in configs:
        docs = [generate_doc(doc_tokens, seed=i) for i in range(args.docs)]
        label = f"{'INT8' if 'quantized' in model else 'FP32'} t={threads} b={batch_size} tok={doc_tokens}"

        print(f"--- {label} ---")
        r = benchmark_config(model, docs, query, threads, batch_size, args.warmup, args.iterations)
        r['label'] = label
        r['model'] = 'INT8' if 'quantized' in model else 'FP32'
        r['threads'] = threads
        r['batch_size'] = batch_size
        r['doc_tokens'] = doc_tokens
        results.append(r)

        print(f"  P50: {r['p50']:.1f} ms ({r['ms_doc']:.1f} ms/doc)")
        print(f"  P95: {r['p95']:.1f} ms")
        print(f"  P99: {r['p99']:.1f} ms")
        print(f"  RAM: {r['rss_delta']:.0f} MB")
        print()

    # Summary table
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print()
    print(f"{'Config':<45} | {'P50':>8} | {'P95':>8} | {'P99':>8} | {'ms/doc':>8} | {'RAM':>6}")
    print("-" * 95)
    for r in results:
        print(f"{r['label']:<45} | {r['p50']:>7.1f} | {r['p95']:>7.1f} | {r['p99']:>7.1f} | {r['ms_doc']:>7.1f} | {r['rss_delta']:>5.0f}")
    print()

    # Best configurations
    print("=" * 90)
    print("BEST CONFIGURATIONS")
    print("=" * 90)
    print()

    # Fastest overall
    fastest = min(results, key=lambda x: x['ms_doc'])
    print(f"Fastest ms/doc: {fastest['label']} ({fastest['ms_doc']:.1f} ms/doc)")

    # Lowest RAM
    lowest_ram = min(results, key=lambda x: x['rss_delta'])
    print(f"Lowest RAM: {lowest_ram['label']} ({lowest_ram['rss_delta']:.0f} MB)")

    # Best P95 for 128 tokens
    r128 = [r for r in results if r['doc_tokens'] == 128]
    if r128:
        best_p95 = min(r128, key=lambda x: x['p95'])
        print(f"Best P95 (128 tok): {best_p95['label']} ({best_p95['p95']:.1f} ms)")

    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| Config | P50 (ms) | P95 (ms) | P99 (ms) | ms/doc | RAM (MB) |")
    print("|--------|----------|----------|----------|--------|----------|")
    for r in results:
        print(f"| {r['label']} | {r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | {r['ms_doc']:.1f} | {r['rss_delta']:.0f} |")


if __name__ == '__main__':
    main()
