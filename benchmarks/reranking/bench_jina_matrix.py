"""
Jina-v1-turbo benchmark matrix: document length x top_k.
"""
import argparse
import os
import sys
import time

# Patch cffi to handle duplicate declarations in _cdefs.h
import cffi

_original_cdef = cffi.FFI.cdef
def _patched_cdef(self, csource, override=False, packed=False, pack=None):
    return _original_cdef(self, csource, override=True, packed=packed, pack=pack)
cffi.FFI.cdef = _patched_cdef

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
    parser = argparse.ArgumentParser(description='Jina-turbo benchmark matrix: length x top_k')
    parser.add_argument('--model', default='jinaai/jina-reranker-v1-turbo-en')
    parser.add_argument('--lengths', default='32,128,512,2048',
                        help='Comma-separated target token lengths')
    parser.add_argument('--top-k', default='5,10,20',
                        help='Comma-separated top_k values')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(',')]
    top_k_values = [int(x) for x in args.top_k.split(',')]
    query = "What is deep learning?"

    print("=" * 80)
    print("Jina-v1-turbo Benchmark Matrix: Document Length x top_k")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}, Batch: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
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

    # Generate docs for max length
    max_k = max(top_k_values)

    # Results matrix
    results = {}

    for target_tokens in lengths:
        docs = [generate_doc_with_length(target_tokens, seed=i) for i in range(max_k)]

        for k in top_k_values:
            k_docs = docs[:k]

            # Warmup
            for _ in range(args.warmup):
                reranker.rerank(query, k_docs, batch_size=args.batch_size)

            # Benchmark
            times = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                reranker.rerank(query, k_docs, batch_size=args.batch_size)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            med_ms = median(times)
            results[(target_tokens, k)] = med_ms

    reranker.close()

    # Print matrix
    print("=" * 80)
    print("RESULTS MATRIX (total time in ms)")
    print("=" * 80)
    print()
    print(f"{'tokens/doc':>12} |", end="")
    for k in top_k_values:
        print(f" top_k={k:<4} |", end="")
    print()
    print("-" * (14 + len(top_k_values) * 14))

    for target_tokens in lengths:
        print(f"{target_tokens:>12} |", end="")
        for k in top_k_values:
            val = results.get((target_tokens, k), 0)
            print(f" {val:>8.0f}   |", end="")
        print()
    print()

    # Print ms/doc matrix
    print("=" * 80)
    print("RESULTS MATRIX (ms/doc)")
    print("=" * 80)
    print()
    print(f"{'tokens/doc':>12} |", end="")
    for k in top_k_values:
        print(f" top_k={k:<4} |", end="")
    print()
    print("-" * (14 + len(top_k_values) * 14))

    for target_tokens in lengths:
        print(f"{target_tokens:>12} |", end="")
        for k in top_k_values:
            val = results.get((target_tokens, k), 0)
            ms_doc = val / k
            print(f" {ms_doc:>8.1f}   |", end="")
        print()
    print()

    # UX rating matrix
    print("=" * 80)
    print("UX RATING MATRIX")
    print("=" * 80)
    print()
    print("Legend: [E]xcellent <200ms | [G]ood <500ms | [O]K <1000ms | [S]low <3000ms | [T]oo slow >3000ms")
    print()
    print(f"{'tokens/doc':>12} |", end="")
    for k in top_k_values:
        print(f" top_k={k:<4} |", end="")
    print()
    print("-" * (14 + len(top_k_values) * 14))

    for target_tokens in lengths:
        print(f"{target_tokens:>12} |", end="")
        for k in top_k_values:
            val = results.get((target_tokens, k), 0)
            if val < 200:
                rating = "  [E]   "
            elif val < 500:
                rating = "  [G]   "
            elif val < 1000:
                rating = "  [O]   "
            elif val < 3000:
                rating = "  [S]   "
            else:
                rating = "  [T]   "
            print(f"{rating}|", end="")
        print()
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| tokens/doc |", end="")
    for k in top_k_values:
        print(f" top_k={k} |", end="")
    print()
    print("|------------|" + ("-----------|" * len(top_k_values)))
    for target_tokens in lengths:
        print(f"| {target_tokens} |", end="")
        for k in top_k_values:
            val = results.get((target_tokens, k), 0)
            print(f" {val:.0f}ms |", end="")
        print()
    print()

    # Budget calculator
    print("=" * 80)
    print("RERANKING BUDGET CALCULATOR")
    print("=" * 80)
    print()
    print("Given a latency budget, find the max (length, top_k) combination:")
    print()
    for budget in [200, 500, 1000, 2000, 5000]:
        print(f"  Budget {budget}ms:")
        best = None
        for target_tokens in lengths:
            for k in top_k_values:
                val = results.get((target_tokens, k), 0)
                if val <= budget and (
                    best is None
                    or k > best[1]
                    or (k == best[1] and target_tokens > best[0])
                ):
                    best = (target_tokens, k, val)
        if best:
            print(f"    -> max: top_k={best[1]}, {best[0]} tokens/doc, {best[2]:.0f}ms")
        else:
            print("    -> no combination fits within budget")
    print()


if __name__ == '__main__':
    main()
