"""
Reranking document length benchmark - measures sensitivity to token length.
The cross-encoder cost is very sensitive to sequence length.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))

from libembedding import Reranker


def generate_doc_with_length(target_tokens, seed=42):
    """Generate a document with approximately target_tokens words."""
    import random
    random.seed(seed)
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "artificial", "intelligence", "neural", "network",
        "deep", "learning", "data", "science", "algorithm", "model", "training",
        "classification", "regression", "clustering", "embedding", "vector",
        "search", "database", "index", "query", "document", "text", "natural",
        "language", "processing", "computer", "vision", "image", "recognition",
        "transformer", "attention", "encoder", "decoder", "bert", "gpt",
        "optimization", "gradient", "descent", "backpropagation", "layer",
        "activation", "function", "relu", "sigmoid", "softmax", "loss",
        "accuracy", "precision", "recall", "f1", "score", "evaluation",
    ]
    # Approx 1 token = 0.75 words, so target_words = target_tokens * 0.75
    target_words = max(1, int(target_tokens * 0.75))
    doc_words = []
    while len(doc_words) < target_words:
        doc_words.append(random.choice(words))
    return " ".join(doc_words[:target_words])


def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def main():
    parser = argparse.ArgumentParser(description='Reranking document length benchmark')
    parser.add_argument('--model', default='BAAI/bge-reranker-base')
    parser.add_argument('--lengths', default='32,64,128,256,512',
                        help='Comma-separated target token lengths')
    parser.add_argument('--docs', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(',')]
    query = "What is deep learning?"

    print("=" * 70)
    print("Reranking Document Length Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Docs: {args.docs}, Threads: {args.threads}, Batch: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()
    print("Objectif: mesurer l'impact de la longueur des documents sur le coût.")
    print("Le cross-encoder est très sensible à la longueur de la séquence.")
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

    print(f"{'tokens/doc':>12} | {'words/doc':>12} | {'Total (ms)':>12} | {'ms/doc':>10} | {'ms/100tok':>10}")
    print("-" * 70)

    results = []
    for target_tokens in lengths:
        docs = [generate_doc_with_length(target_tokens, seed=i) for i in range(args.docs)]
        actual_words = len(docs[0].split())

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
        ms_per_doc = med_ms / args.docs
        ms_per_100tok = med_ms / (args.docs * target_tokens / 100)

        print(f"{target_tokens:>12} | {actual_words:>12} | {med_ms:>12.1f} | {ms_per_doc:>10.1f} | {ms_per_100tok:>10.1f}")

        results.append({
            'target_tokens': target_tokens,
            'actual_words': actual_words,
            'total_ms': med_ms,
            'ms_per_doc': ms_per_doc,
            'ms_per_100tok': ms_per_100tok,
        })

    reranker.close()
    print()

    # Analysis
    print("======================================================================")
    print("ANALYSIS")
    print("======================================================================")
    print()
    if len(results) >= 2:
        base = results[0]
        for r in results[1:]:
            ratio_tokens = r['target_tokens'] / base['target_tokens']
            ratio_time = r['total_ms'] / base['total_ms']
            print(f"  {base['target_tokens']} → {r['target_tokens']} tokens "
                  f"({ratio_tokens:.1f}x) → {ratio_time:.2f}x time")
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| tokens/doc | Total (ms) | ms/doc | ms/100tok |")
    print("|------------|-----------|--------|----------|")
    for r in results:
        print(f"| {r['target_tokens']} | {r['total_ms']:.0f} | {r['ms_per_doc']:.1f} | {r['ms_per_100tok']:.1f} |")


if __name__ == '__main__':
    main()
