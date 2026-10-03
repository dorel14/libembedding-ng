"""
Reranking workers benchmark - tests parallelism: N sessions x 1 thread vs 1 session x N threads.
Note: libembedding doesn't have a session pool for rerankers yet.
This benchmark tests thread scaling with a single session.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from bench_common import benchmark_rerank, generate_documents, get_rss_mb
from libembedding import Reranker


def main():
    parser = argparse.ArgumentParser(description='Reranking workers benchmark')
    parser.add_argument('--model', default='BAAI/bge-reranker-base')
    parser.add_argument('--docs', type=int, default=50)
    parser.add_argument('--total-threads', default='1,2,4,8')
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    thread_counts = [int(x) for x in args.total_threads.split(',')]
    query = "What is deep learning?"
    docs = generate_documents(args.docs)

    print("=" * 70)
    print("Reranking Workers Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Docs: {args.docs}")
    print(f"Batch size: {args.batch_size or 'default (256)'}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()
    print("NOTE: libembedding has no session pool for rerankers yet.")
    print("Testing single-session thread scaling only.")
    print()

    print(f"{'Config':>10} | {'Threads':>8} | {'Total (ms)':>12} | {'ms/doc':>10} | {'docs/sec':>10} | {'RSS (MB)':>10}")
    print("-" * 75)

    results = []
    for threads in thread_counts:
        config = f"1x{threads}"
        try:
            reranker = Reranker(
                args.model,
                threads=threads,
                batch_size=args.batch_size,
                offline=args.offline,
                show_download_progress=False,
            )

            stats = benchmark_rerank(reranker, query, docs, args.batch_size,
                                     warmup=args.warmup, iterations=args.iterations)
            rss = get_rss_mb()

            ms_per_doc = stats['median_ms'] / args.docs
            docs_per_sec = 1000.0 / ms_per_doc if ms_per_doc > 0 else 0

            print(f"{config:>10} | {threads:>8} | {stats['median_ms']:>12.2f} | "
                  f"{ms_per_doc:>10.2f} | {docs_per_sec:>10.1f} | {rss:>10.1f}")

            results.append({
                'config': config,
                'threads': threads,
                'total_ms': stats['median_ms'],
                'ms_per_doc': ms_per_doc,
                'docs_per_sec': docs_per_sec,
                'rss_mb': rss,
            })
            reranker.close()
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"{config:>10} | {threads:>8} | SKIP: {e}")
            # Closing a context that failed to build must not mask the
            # error already reported just above.
            try:
                reranker.close()
            except Exception:  # noqa: BLE001, S110 - deliberate cleanup
                pass
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| Config | Docs/s | ms/doc | CPU% | RAM (MB) |")
    print("|--------|--------|--------|------|----------|")
    for r in results:
        print(f"| {r['config']}    | {r['docs_per_sec']:.1f} | {r['ms_per_doc']:.1f} | ?    | {r['rss_mb']:.0f} |")


if __name__ == '__main__':
    main()
