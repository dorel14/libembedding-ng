"""
Reranking batching benchmark - measures batch_size sensitivity.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_common import benchmark_rerank, generate_documents
from libembedding import Reranker


def main():
    parser = argparse.ArgumentParser(description='Reranking batching benchmark')
    parser.add_argument('--model', default='BAAI/bge-reranker-base')
    parser.add_argument('--docs', type=int, default=50)
    parser.add_argument('--batch-sizes', default='1,8,16,32')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    query = "What is deep learning?"
    docs = generate_documents(args.docs)

    print("=" * 70)
    print("Reranking Batching Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Docs: {args.docs}")
    print(f"Threads: {args.threads or 'auto'}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    print("Loading model...")
    reranker = Reranker(
        args.model,
        threads=args.threads,
        offline=args.offline,
        show_download_progress=False,
    )
    print()

    print(f"{'batch_size':>12} | {'Total (ms)':>12} | {'ms/doc':>10} | {'docs/sec':>10} | {'p95 (ms)':>10}")
    print("-" * 70)

    results = []
    for bs in batch_sizes:
        stats = benchmark_rerank(reranker, query, docs, bs,
                                 warmup=args.warmup, iterations=args.iterations)
        ms_per_doc = stats['median_ms'] / args.docs
        docs_per_sec = 1000.0 / ms_per_doc if ms_per_doc > 0 else 0
        print(f"{bs:>12} | {stats['median_ms']:>12.2f} | {ms_per_doc:>10.2f} | {docs_per_sec:>10.1f} | {stats['p95_ms']:>10.2f}")
        results.append({
            'batch_size': bs,
            'total_ms': stats['median_ms'],
            'ms_per_doc': ms_per_doc,
            'docs_per_sec': docs_per_sec,
            'p95_ms': stats['p95_ms'],
        })

    reranker.close()
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| batch_size | Temps total (ms) | ms/doc | docs/sec |")
    print("|------------|-----------------|--------|----------|")
    for r in results:
        print(f"| {r['batch_size']}          | {r['total_ms']:.1f}               | {r['ms_per_doc']:.1f}      | {r['docs_per_sec']:.1f}        |")


if __name__ == '__main__':
    main()
