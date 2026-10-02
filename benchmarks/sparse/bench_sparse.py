"""
Sparse embedding benchmark.
Compares SPLADE++ vs BGE-M3 on latency, memory, and term distribution.
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
from libembedding import SparseTextEmbedding


def percentile(values, p):
    s = sorted(values)
    idx = int(p / 100.0 * len(s))
    return s[min(idx, len(s) - 1)]


def get_rss_mb():
    import subprocess
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


def benchmark_sparse_model(model_name, texts, threads, batch_size, warmup=2, iterations=10):
    """Benchmark a sparse embedding model."""
    rss_before = get_rss_mb()

    t0 = time.perf_counter()
    model = SparseTextEmbedding(
        model_name,
        threads=threads,
        batch_size=batch_size,
        offline=True,
        show_download_progress=False,
    )
    load_time = (time.perf_counter() - t0) * 1000

    rss_after_load = get_rss_mb()

    # Warmup
    for _ in range(warmup):
        model.embed(texts, batch_size=batch_size)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.embed(texts, batch_size=batch_size)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    # Get term statistics
    sample = model.embed(texts[:1], batch_size=1)[0]
    avg_terms = len(sample.indices)

    model.close()

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(0.95 * len(times))]

    return {
        'load_ms': load_time,
        'rss_delta': rss_after_load - rss_before,
        'p50': p50,
        'p95': p95,
        'ms_doc': p50 / len(texts),
        'docs_per_sec': 1000.0 / (p50 / len(texts)) if p50 > 0 else 0,
        'avg_terms': avg_terms,
    }


def main():
    parser = argparse.ArgumentParser(description='Sparse embedding benchmark')
    parser.add_argument('--docs', type=int, default=20, help='Number of documents')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--batch-size', type=int, default=0, help='Batch size (0=default)')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=10, help='Benchmark iterations')
    args = parser.parse_args()

    print("=" * 80)
    print("Sparse Embedding Benchmark")
    print("=" * 80)
    print(f"Docs: {args.docs}, Threads: {args.threads}, Batch: {args.batch_size or 'default'}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    # Test documents
    docs = [
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning learns through trial and error.",
        "Transfer learning reuses pre-trained models.",
        "Attention mechanisms focus on relevant input parts.",
        "BERT is a transformer-based language model.",
        "GPT generates text autoregressively.",
        "Vector search finds similar embeddings.",
        "BM25 is a traditional retrieval algorithm.",
        "Hybrid search combines sparse and dense methods.",
        "SPLADE produces sparse lexical representations.",
        "Inverted index maps terms to documents.",
        "TF-IDF weights terms by frequency.",
        "Information retrieval finds relevant documents.",
        "Question answering retrieves answers from text.",
        "Named entity recognition identifies proper nouns.",
        "Sentiment analysis determines text polarity.",
        "Machine translation converts text between languages.",
    ][:args.docs]

    if len(docs) < args.docs:
        # Repeat if needed
        docs = docs * (args.docs // len(docs) + 1)
        docs = docs[:args.docs]

    print(f"Test documents: {len(docs)}")
    print()

    models = [
        "prithivida/Splade_PP_en_v1",
        "BAAI/bge-m3",
    ]

    results = []
    for model_name in models:
        print(f"--- {model_name} ---")
        try:
            r = benchmark_sparse_model(
                model_name, docs, args.threads, args.batch_size,
                args.warmup, args.iterations
            )
            r['model'] = model_name
            results.append(r)

            print(f"  Load time: {r['load_ms']:.0f} ms")
            print(f"  RAM delta: {r['rss_delta']:.0f} MB")
            print(f"  P50: {r['p50']:.1f} ms ({r['ms_doc']:.1f} ms/doc)")
            print(f"  P95: {r['p95']:.1f} ms")
            print(f"  Throughput: {r['docs_per_sec']:.1f} docs/s")
            print(f"  Avg terms/doc: {r['avg_terms']}")
        except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
            print(f"  FAILED: {e}")
            results.append({
                'model': model_name,
                'error': str(e),
            })
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Model':<35} | {'Load':>8} | {'RAM':>6} | {'P50':>10} | {'ms/doc':>8} | {'docs/s':>8} | {'terms':>8}")
    print("-" * 95)
    for r in results:
        if r.get('error'):
            print(f"{r['model']:<35} | {'ERROR':>8} | {'---':>6} | {'---':>10} | {'---':>8} | {'---':>8} | {'---':>8}")
        else:
            print(f"{r['model']:<35} | {r['load_ms']:>7.0f} | {r['rss_delta']:>5.0f} | {r['p50']:>9.1f} | {r['ms_doc']:>7.1f} | {r['docs_per_sec']:>7.1f} | {r['avg_terms']:>7.0f}")
    print()

    # Markdown
    print("## Results (for markdown)")
    print()
    print("| Model | Load (ms) | RAM (MB) | P50 (ms) | ms/doc | docs/s | avg terms |")
    print("|-------|-----------|----------|----------|--------|--------|-----------|")
    for r in results:
        if not r.get('error'):
            print(f"| {r['model']} | {r['load_ms']:.0f} | {r['rss_delta']:.0f} | {r['p50']:.1f} | {r['ms_doc']:.1f} | {r['docs_per_sec']:.1f} | {r['avg_terms']:.0f} |")


if __name__ == '__main__':
    main()
