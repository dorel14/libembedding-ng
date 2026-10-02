"""
Reranking ORT settings investigation - tests GraphOptimizationLevel and ExecutionMode.
Based on the embedding benchmarks where ORT settings made a big difference.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))

from libembedding import Reranker


def generate_documents(n, seed=42):
    """Generate n synthetic documents."""
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
    parser = argparse.ArgumentParser(description='Reranking ORT settings investigation')
    parser.add_argument('--model', default='BAAI/bge-reranker-base')
    parser.add_argument('--docs', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    query = "What is deep learning?"
    docs = generate_documents(args.docs)

    print("=" * 70)
    print("Reranking ORT Settings Investigation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Docs: {args.docs}, Threads: {args.threads}, Batch: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()
    print("Objectif: tester si les réglages ORT peuvent améliorer les perfs.")
    print()
    print("NOTE: Cette investigation nécessite de modifier le code C++ pour changer")
    print("les paramètres ORT (GraphOptimizationLevel, ExecutionMode).")
    print()
    print("Settings ORT actuels dans onnx_session_impl.hpp:")
    print("  - GraphOptimizationLevel: ORT_ENABLE_ALL (niveau 3)")
    print("  - MemPattern: EnableMemPattern = ON")
    print("  - CpuMemArena: EnableCpuMemArena = ON")
    print("  - IntraOpNumThreads: 4 (configuré)")
    print("  - InterOpNumThreads: 1 (fixé)")
    print()

    print("======================================================================")
    print("BASELINE (settings actuels)")
    print("======================================================================")
    print()

    print("Loading model with current settings...")
    t0 = time.perf_counter()
    reranker = Reranker(
        args.model,
        threads=args.threads,
        batch_size=args.batch_size,
        offline=args.offline,
        show_download_progress=False,
    )
    load_time = (time.perf_counter() - t0) * 1000

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
    docs_per_sec = 1000.0 / ms_per_doc if ms_per_doc > 0 else 0

    print(f"  Load time:      {load_time:.0f} ms")
    print(f"  Total time:     {med_ms:.2f} ms")
    print(f"  ms/doc:         {ms_per_doc:.2f}")
    print(f"  docs/sec:       {docs_per_sec:.1f}")
    print()

    info = reranker.info()
    print(f"  Model:          {info.name}")
    print(f"  Max length:     {info.max_length} tokens")
    print(f"  Provider:       {info.provider}")
    print()

    reranker.close()

    print("======================================================================")
    print("INVESTIGATION NÉCESSAIRE")
    print("======================================================================")
    print()
    print("Pour tester d'autres settings ORT, il faut modifier onnx_session_impl.hpp:")
    print()
    print("1. GraphOptimizationLevel:")
    print("   - ORT_DISABLE_ALL (0) — pas d'optimisation")
    print("   - ORT_ENABLE_BASIC (1) — optimisations de base")
    print("   - ORT_ENABLE_EXTENDED (2) — optimisations étendues")
    print("   - ORT_ENABLE_ALL (3) — actuel, toutes optimisations")
    print()
    print("2. ExecutionMode:")
    print("   - ORT_SEQUENTIAL (0) — exécution séquentielle")
    print("   - ORT_PARALLEL (1) — exécution parallèle")
    print()
    print("3. MemPattern / CpuMemArena:")
    print("   - Actuellement ON/OFF")
    print("   - Tester OFF/OFF, ON/OFF, OFF/ON")
    print()
    print("4. IntraOpNumThreads:")
    print("   - Actuellement 4")
    print("   - Tester 1, 2, 4, 8")
    print()

    print("## Results (for markdown)")
    print()
    print("| Setting | Value | ms/doc | docs/s |")
    print("|---------|-------|--------|--------|")
    print(f"| Baseline (ORT_ENABLE_ALL, 4 threads) | current | {ms_per_doc:.1f} | {docs_per_sec:.1f} |")
    print()
    print("## Prochaines étapes")
    print()
    print("1. Modifier onnx_session_impl.hpp pour exposer les settings ORT")
    print("2. Recompiler et relancer ce benchmark")
    print("3. Comparer les combinaisons")
    print()


if __name__ == '__main__':
    main()
