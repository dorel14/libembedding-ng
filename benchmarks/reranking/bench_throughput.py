"""
Reranking throughput benchmark - compares different reranker models.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_common import benchmark_rerank, generate_documents, get_rss_mb
from libembedding import Reranker


def main():
    parser = argparse.ArgumentParser(description='Reranking throughput benchmark')
    parser.add_argument('--models', default='BAAI/bge-reranker-base')
    parser.add_argument('--docs', type=int, default=50)
    parser.add_argument('--threads', default='1,2,4,8')
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]
    thread_counts = [int(x) for x in args.threads.split(',')]
    query = "What is deep learning?"
    docs = generate_documents(args.docs)

    print("=" * 70)
    print("Reranking Throughput Benchmark")
    print("=" * 70)
    print(f"Docs: {args.docs}")
    print(f"Batch size: {args.batch_size or 'default (256)'}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    all_results = []

    for model_name in model_names:
        print(f"--- {model_name} ---")
        for threads in thread_counts:
            try:
                t0 = __import__('time').perf_counter()
                reranker = Reranker(
                    model_name,
                    threads=threads,
                    batch_size=args.batch_size,
                    offline=args.offline,
                    show_download_progress=False,
                )
                load_time = (__import__('time').perf_counter() - t0) * 1000

                stats = benchmark_rerank(reranker, query, docs, args.batch_size,
                                         warmup=args.warmup, iterations=args.iterations)
                rss_after = get_rss_mb()

                ms_per_doc = stats['median_ms'] / args.docs
                docs_per_sec = 1000.0 / ms_per_doc if ms_per_doc > 0 else 0

                print(f"  threads={threads}: {stats['median_ms']:.1f} ms total, "
                      f"{ms_per_doc:.2f} ms/doc, {docs_per_sec:.1f} docs/s, "
                      f"RSS: {rss_after:.0f} MB")

                all_results.append({
                    'model': model_name,
                    'threads': threads,
                    'total_ms': stats['median_ms'],
                    'ms_per_doc': ms_per_doc,
                    'docs_per_sec': docs_per_sec,
                    'rss_mb': rss_after,
                    'load_ms': load_time,
                })
                reranker.close()
            except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
                print(f"  threads={threads}: SKIP ({e})")
                # Closing a context that failed to build must not mask the
                # error already reported just above.
                try:
                    reranker.close()
                except Exception:  # noqa: BLE001, S110 - deliberate cleanup
                    pass
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<40} {'Threads':>8} {'ms/doc':>10} {'docs/s':>10} {'RSS (MB)':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<40} {r['threads']:>8} {r['ms_per_doc']:>10.2f} "
              f"{r['docs_per_sec']:>10.1f} {r['rss_mb']:>10.0f}")
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| Modèle | ms/doc | RAM (MB) |")
    print("|--------|--------|----------|")
    # Group by model, show best threads
    best_by_model = {}
    for r in all_results:
        mn = r['model']
        if mn not in best_by_model or r['docs_per_sec'] > best_by_model[mn]['docs_per_sec']:
            best_by_model[mn] = r
    for mn, r in best_by_model.items():
        print(f"| {mn} | {r['ms_per_doc']:.1f} | {r['rss_mb']:.0f} |")


if __name__ == '__main__':
    main()
