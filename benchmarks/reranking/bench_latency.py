"""
Reranking latency benchmark - measures absolute cost for different corpus sizes.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_common import benchmark_rerank, generate_documents, get_rss_mb
from libembedding import Reranker


def main():
    parser = argparse.ArgumentParser(description='Reranking latency benchmark')
    parser.add_argument('--model', default='BAAI/bge-reranker-base')
    parser.add_argument('--docs', default='10,20,50,100,200',
                        help='Comma-separated list of document counts')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    doc_counts = [int(x) for x in args.docs.split(',')]
    query = "What is deep learning?"

    print("=" * 70)
    print("Reranking Latency Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Query: '{query}'")
    print(f"Threads: {args.threads}")
    print(f"Batch size: {args.batch_size or 'default (256)'}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    print("Loading model...")
    t0 = __import__('time').perf_counter()
    reranker = Reranker(
        args.model,
        threads=args.threads,
        batch_size=args.batch_size,
        offline=args.offline,
        show_download_progress=False,
    )
    load_time = (__import__('time').perf_counter() - t0) * 1000
    rss_after_load = get_rss_mb()
    print(f"  Load time: {load_time:.0f} ms")
    print(f"  RSS after load: {rss_after_load:.1f} MB")
    print()

    print(f"{'Docs':>6} | {'Total (ms)':>12} | {'ms/doc':>10} | {'docs/sec':>10} | {'p95 (ms)':>10}")
    print("-" * 65)

    results = []
    for n in doc_counts:
        docs = generate_documents(n)
        stats = benchmark_rerank(reranker, query, docs, args.batch_size,
                                 warmup=args.warmup, iterations=args.iterations)
        ms_per_doc = stats['median_ms'] / n
        docs_per_sec = 1000.0 / ms_per_doc if ms_per_doc > 0 else 0
        print(f"{n:>6} | {stats['median_ms']:>12.2f} | {ms_per_doc:>10.2f} | {docs_per_sec:>10.1f} | {stats['p95_ms']:>10.2f}")
        results.append({
            'docs': n,
            'total_ms': stats['median_ms'],
            'ms_per_doc': ms_per_doc,
            'docs_per_sec': docs_per_sec,
            'p95_ms': stats['p95_ms'],
        })

    rss_peak = get_rss_mb()
    reranker.close()

    print()
    print(f"Peak RSS: {rss_peak:.1f} MB")
    print()

    # Summary for markdown
    print("## Results (for markdown)")
    print()
    print("| Docs | Temps (ms) | ms/doc |")
    print("|------|-----------|--------|")
    for r in results:
        print(f"| {r['docs']}   | {r['total_ms']:.1f}     | {r['ms_per_doc']:.1f}  |")


if __name__ == '__main__':
    main()
