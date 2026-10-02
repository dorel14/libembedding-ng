"""
Reranking top_k benchmark - measures real cost vs number of candidates to rerank.
This is the most useful benchmark for Whoosh-NG users.

Question answered: "How many documents can I rerank before UX becomes too slow?"
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))

from libembedding import Reranker


def generate_documents(n, seed=42):
    """Generate n synthetic documents of varying lengths."""
    import random
    random.seed(seed)
    base_docs = [
        "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Pizza is a traditional Italian dish made with dough, tomato sauce, and cheese.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "The Python programming language was created by Guido van Rossum in 1991.",
        "Quantum computing leverages quantum mechanical phenomena to perform computation.",
        "The Great Wall of China is a series of fortifications built over centuries.",
        "Natural language processing enables computers to understand human language.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
        "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
        "Shakespeare wrote 37 plays during his lifetime in Elizabethan England.",
        "CRISPR gene editing allows scientists to modify DNA sequences with precision.",
        "Electric vehicles are becoming more popular as battery technology improves.",
        "The theory of relativity changed our understanding of space, time, and gravity.",
        "Blockchain technology enables decentralized and transparent digital transactions.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The Internet of Things connects billions of devices to exchange data.",
        "Antibiotics revolutionized medicine by treating bacterial infections effectively.",
        "Virtual reality creates immersive digital environments for users to explore.",
    ]
    docs = []
    for i in range(n):
        doc = base_docs[i % len(base_docs)]
        repeat = (i % 3) + 1
        docs.append((doc + " ") * repeat)
    return docs


def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def main():
    parser = argparse.ArgumentParser(description='Reranking top_k benchmark')
    parser.add_argument('--model', default='BAAI/bge-reranker-base')
    parser.add_argument('--top-k', default='5,10,20,50,100',
                        help='Comma-separated list of top_k values')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    top_k_values = [int(x) for x in args.top_k.split(',')]
    query = "What is deep learning?"

    print("=" * 70)
    print("Reranking top_k Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Query: '{query}'")
    print(f"Threads: {args.threads}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()
    print("Objectif: combien de docs peut-on reranker avant que l'UX soit trop lente?")
    print()

    print("Loading model...")
    t0 = time.perf_counter()
    reranker = Reranker(
        args.model,
        threads=args.threads,
        batch_size=args.batch_size,
        offline=args.offline,
        show_download_progress=False,
    )
    load_time = (time.perf_counter() - t0) * 1000
    print(f"  Load time: {load_time:.0f} ms")
    print()

    # Pre-generate max docs
    max_k = max(top_k_values)
    all_docs = generate_documents(max_k)

    print(f"{'top_k':>8} | {'Total (ms)':>12} | {'ms/doc':>10} | {'docs/sec':>10} | {'p95 (ms)':>10} | {'UX':>8}")
    print("-" * 75)

    results = []
    for k in top_k_values:
        docs = all_docs[:k]

        # Warmup
        for _ in range(args.warmup):
            reranker.rerank(query, docs, batch_size=args.batch_size)

        # Benchmark
        times = []
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            reranker.rerank(query, docs, batch_size=args.batch_size)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        med_ms = median(times)
        ms_per_doc = med_ms / k
        docs_per_sec = 1000.0 / ms_per_doc if ms_per_doc > 0 else 0

        # UX rating
        if med_ms < 200:
            ux = "Excellent"
        elif med_ms < 500:
            ux = "Bon"
        elif med_ms < 1000:
            ux = "OK"
        elif med_ms < 3000:
            ux = "Lent"
        else:
            ux = "Trop"

        print(f"{k:>8} | {med_ms:>12.1f} | {ms_per_doc:>10.1f} | {docs_per_sec:>10.1f} | {med_ms:>10.1f} | {ux:>8}")

        results.append({
            'top_k': k,
            'total_ms': med_ms,
            'ms_per_doc': ms_per_doc,
            'docs_per_sec': docs_per_sec,
            'ux': ux,
        })

    reranker.close()
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| top_k | temps (ms) | ms/doc | docs/sec | UX |")
    print("|-------|-----------|--------|----------|-----|")
    for r in results:
        print(f"| {r['top_k']} | {r['total_ms']:.0f} | {r['ms_per_doc']:.1f} | {r['docs_per_sec']:.1f} | {r['ux']} |")
    print()

    # Recommendation
    print("## Recommandation Whoosh-NG")
    print()
    for r in results:
        if r['total_ms'] < 1000:
            print(f"**top_k={r['top_k']}: {r['total_ms']:.0f}ms — {r['ux']}**")
        elif r['total_ms'] < 3000:
            print(f"top_k={r['top_k']}: {r['total_ms']:.0f}ms — {r['ux']} (acceptable)")
        else:
            print(f"top_k={r['top_k']}: {r['total_ms']:.0f}ms — {r['ux']} (trop lent)")
    print()


if __name__ == '__main__':
    main()
