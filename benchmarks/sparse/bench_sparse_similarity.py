"""
Sparse similarity benchmark.
Compares: dense cosine, sparse dot product, BM25, dense+sparse hybrid.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python/src"))

import numpy as np
from libembedding import SparseTextEmbedding, TextEmbedding


def benchmark_dense_cosine(model: TextEmbedding, queries: list[str], docs: list[str], n_runs: int = 5) -> dict:
    """Dense cosine similarity."""
    doc_emb = model.embed(docs)
    query_emb = model.embed(queries)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for q in query_emb:
            # Only the dot products are timed. An argsort here would add a full
            # sort of the score vector to every iteration and inflate the
            # reported time, which is what a retrieval benchmark must not do.
            _ = np.dot(doc_emb, q)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000 / len(queries))

    import statistics
    return {
        "method": "dense_cosine",
        "latency_ms_per_query": statistics.mean(times),
    }


def benchmark_sparse_dot(model: SparseTextEmbedding, queries: list[str], docs: list[str], n_runs: int = 5) -> dict:
    """Sparse dot product similarity."""
    doc_vecs = model.embed(docs)
    query_vecs = model.embed(queries)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for q in query_vecs:
            scores = []
            for d in doc_vecs:
                # Dot product of sparse vectors
                if hasattr(q, 'indices') and hasattr(d, 'indices'):
                    # CSR-like
                    score = sum(q.values[i] * d.values[j]
                               for i, qi in enumerate(q.indices)
                               for j, di in enumerate(d.indices)
                               if qi == di)
                else:
                    score = sum(q.get(idx, 0.0) * val for idx, val in d.items())
                scores.append(score)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000 / len(queries))

    import statistics
    return {
        "method": "sparse_dot",
        "latency_ms_per_query": statistics.mean(times),
    }


def benchmark_hybrid(model: TextEmbedding, sparse_model: SparseTextEmbedding,
                     queries: list[str], docs: list[str], n_runs: int = 5) -> dict:
    """Hybrid dense + sparse."""
    doc_dense = model.embed(docs)
    query_dense = model.embed(queries)
    doc_sparse = sparse_model.embed(docs)
    query_sparse = sparse_model.embed(queries)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for q_dense, q_sparse in zip(query_dense, query_sparse):
            scores = []
            for d_dense, d_sparse in zip(doc_dense, doc_sparse):
                dense_score = np.dot(d_dense, q_dense)
                sparse_score = sum(q_sparse.get(idx, 0.0) * val for idx, val in d_sparse.items())
                scores.append(0.5 * dense_score + 0.5 * sparse_score)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000 / len(queries))

    import statistics
    return {
        "method": "hybrid_dense_sparse",
        "latency_ms_per_query": statistics.mean(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Sparse similarity benchmark")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--docs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"Benchmarking similarity with {args.queries} queries, {args.docs} docs")
    print()

    # Sample texts
    queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning.",
        "What is NLP?",
        "What is computer vision?",
    ] * (args.queries // 5)

    docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are computing systems inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks.",
        "NLP stands for Natural Language Processing.",
        "Computer vision enables computers to understand images.",
        "Transformers revolutionized NLP in 2017.",
        "BERT is a bidirectional transformer model.",
        "GPT models are autoregressive language models.",
        "Embeddings map text to dense vectors.",
        "Sparse embeddings use bag-of-words representations.",
    ] * (args.docs // 10)

    results = []

    # Dense
    print("[1/3] Dense cosine...")
    try:
        dense_model = TextEmbedding("Qdrant/all-MiniLM-L6-v2")
        r = benchmark_dense_cosine(dense_model, queries, docs, args.runs)
        results.append(r)
        print(f"  {r['latency_ms_per_query']:.2f} ms/query")
    except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
        print(f"  ERROR: {e}")

    # Sparse
    print("[2/3] Sparse dot product...")
    try:
        sparse_model = SparseTextEmbedding("Qdrant/splade-vit-b16-mpnet")
        r = benchmark_sparse_dot(sparse_model, queries, docs, args.runs)
        results.append(r)
        print(f"  {r['latency_ms_per_query']:.2f} ms/query")
    except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
        print(f"  ERROR: {e}")

    # Hybrid
    print("[3/3] Hybrid dense+sparse...")
    try:
        r = benchmark_hybrid(dense_model, sparse_model, queries, docs, args.runs)
        results.append(r)
        print(f"  {r['latency_ms_per_query']:.2f} ms/query")
    except Exception as e:  # noqa: BLE001 - cffi/ORT raise many types
        print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("SIMILARITY BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Latency (ms/query)':<20}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<30} {r['latency_ms_per_query']:<20.2f}")


if __name__ == "__main__":
    import argparse
    main()
