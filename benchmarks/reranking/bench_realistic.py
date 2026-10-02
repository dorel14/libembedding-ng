"""
Realistic document benchmark with P50/P95/P99 percentiles.
Tests Jina-v1-turbo with realistic document lengths and measures tail latency.

This is the benchmark that matters for Whoosh-NG production use:
- Realistic document lengths (20-1000 tokens)
- P50/P95/P99 percentiles (not just median)
- Multiple top_k values
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


def generate_realistic_doc(target_tokens, seed=42):
    """Generate a realistic-looking document with approximately target_tokens words."""
    import random
    random.seed(seed)

    # Realistic English words (not just ML terms)
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "machine", "learning", "data", "model", "system", "algorithm", "search", "query",
        "document", "text", "information", "result", "process", "analysis", "method",
        "approach", "technique", "application", "performance", "evaluation", "research",
        "study", "experiment", "implementation", "development", "design", "architecture",
        "framework", "library", "tool", "service", "platform", "interface", "api",
    ]

    target_words = max(1, int(target_tokens * 0.75))
    doc_words = []
    while len(doc_words) < target_words:
        doc_words.append(random.choice(words))
    return " ".join(doc_words[:target_words])


def percentile(values, p):
    """Calculate the p-th percentile (0-100)."""
    s = sorted(values)
    idx = int(p / 100.0 * len(s))
    return s[min(idx, len(s) - 1)]


def main():
    parser = argparse.ArgumentParser(description='Realistic document benchmark with P50/P95/P99')
    parser.add_argument('--model', default='jinaai/jina-reranker-v1-turbo-en')
    parser.add_argument('--lengths', default='20,50,100,200,500,1000',
                        help='Comma-separated target token lengths')
    parser.add_argument('--top-k', default='5,10,20,50',
                        help='Comma-separated top_k values')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(',')]
    top_k_values = [int(x) for x in args.top_k.split(',')]
    query = "What is deep learning?"

    print("=" * 80)
    print("Realistic Document Benchmark with P50/P95/P99")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}, Batch: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()
    print("Objective: measure tail latency for production Whoosh-NG use.")
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

    # Results storage
    results = {}

    for target_tokens in lengths:
        docs = [generate_realistic_doc(target_tokens, seed=i) for i in range(max(top_k_values))]

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

            p50 = percentile(times, 50)
            p95 = percentile(times, 95)
            p99 = percentile(times, 99)
            mean = sum(times) / len(times)

            results[(target_tokens, k)] = {
                'p50': p50,
                'p95': p95,
                'p99': p99,
                'mean': mean,
                'ms_doc': p50 / k,
            }

    reranker.close()

    # Print results
    for k in top_k_values:
        print("=" * 80)
        print(f"top_k = {k}")
        print("=" * 80)
        print()
        print(f"{'tokens/doc':>12} | {'P50 (ms)':>10} | {'P95 (ms)':>10} | {'P99 (ms)':>10} | {'Mean (ms)':>10} | {'ms/doc':>8}")
        print("-" * 75)

        for target_tokens in lengths:
            r = results[(target_tokens, k)]
            print(f"{target_tokens:>12} | {r['p50']:>10.1f} | {r['p95']:>10.1f} | {r['p99']:>10.1f} | {r['mean']:>10.1f} | {r['ms_doc']:>8.1f}")
        print()

    # P95-focused matrix (most important for production)
    print("=" * 80)
    print("P95 LATENCY MATRIX (ms) - Production SLA target")
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
            r = results[(target_tokens, k)]
            print(f" {r['p95']:>8.0f}   |", end="")
        print()
    print()

    # P99 matrix
    print("=" * 80)
    print("P99 LATENCY MATRIX (ms) - Worst case")
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
            r = results[(target_tokens, k)]
            print(f" {r['p99']:>8.0f}   |", end="")
        print()
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("### P50 (median)")
    print()
    print("| tokens/doc |", end="")
    for k in top_k_values:
        print(f" top_k={k} |", end="")
    print()
    print("|------------|" + ("-----------|" * len(top_k_values)))
    for target_tokens in lengths:
        print(f"| {target_tokens} |", end="")
        for k in top_k_values:
            r = results[(target_tokens, k)]
            print(f" {r['p50']:.0f}ms |", end="")
        print()
    print()

    print("### P95 (production SLA)")
    print()
    print("| tokens/doc |", end="")
    for k in top_k_values:
        print(f" top_k={k} |", end="")
    print()
    print("|------------|" + ("-----------|" * len(top_k_values)))
    for target_tokens in lengths:
        print(f"| {target_tokens} |", end="")
        for k in top_k_values:
            r = results[(target_tokens, k)]
            print(f" {r['p95']:.0f}ms |", end="")
        print()
    print()

    print("### P99 (worst case)")
    print()
    print("| tokens/doc |", end="")
    for k in top_k_values:
        print(f" top_k={k} |", end="")
    print()
    print("|------------|" + ("-----------|" * len(top_k_values)))
    for target_tokens in lengths:
        print(f"| {target_tokens} |", end="")
        for k in top_k_values:
            r = results[(target_tokens, k)]
            print(f" {r['p99']:.0f}ms |", end="")
        print()
    print()


if __name__ == '__main__':
    main()
