"""
Quality benchmark: FP32 vs INT8 for reranking.
Measures NDCG@10, MRR, and Recall@10 to determine if quantization
preserves ranking quality enough to be the default.

This is the deciding factor for DEFAULT model selection:
- If NDCG diff < 1%: INT8 can be default
- If NDCG diff > 1%: keep FP32 as default
"""
import argparse
import math
import os
import sys

# Patch cffi
import cffi

_orig = cffi.FFI.cdef
def _patch(self, cs, override=False, **kw):
    return _orig(self, cs, override=True, **kw)
cffi.FFI.cdef = _patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'src'))
from libembedding import Reranker


def compute_dcg(relevances, k=10):
    """Compute DCG@k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def compute_ndcg(scores_with_relevance, k=10):
    """Compute NDCG@k given list of (score, relevance) tuples."""
    # Sort by score descending (what the reranker produces)
    sorted_by_score = sorted(scores_with_relevance, key=lambda x: x[0], reverse=True)
    relevances = [r for _, r in sorted_by_score]

    # Ideal DCG (sorted by relevance descending)
    ideal_relevances = sorted([r for _, r in scores_with_relevance], reverse=True)

    dcg = compute_dcg(relevances, k)
    idcg = compute_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(scores_with_relevance, relevance_threshold=1):
    """Compute MRR (Mean Reciprocal Rank)."""
    sorted_by_score = sorted(scores_with_relevance, key=lambda x: x[0], reverse=True)
    for i, (_, rel) in enumerate(sorted_by_score):
        if rel >= relevance_threshold:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall(scores_with_relevance, k=10, relevance_threshold=1):
    """Compute Recall@k."""
    total_relevant = sum(1 for _, r in scores_with_relevance if r >= relevance_threshold)
    if total_relevant == 0:
        return 0.0

    sorted_by_score = sorted(scores_with_relevance, key=lambda x: x[0], reverse=True)
    relevant_in_top_k = sum(1 for _, r in sorted_by_score[:k] if r >= relevance_threshold)

    return relevant_in_top_k / total_relevant


def create_test_queries():
    """Create test queries with known relevance judgments.

    Returns list of (query, documents, relevance_scores) tuples.
    relevance_scores: 0=irrelevant, 1=somewhat relevant, 2=highly relevant
    """
    test_cases = [
        # Query 1: Deep learning
        (
            "What is deep learning?",
            [
                "Deep learning uses neural networks with multiple layers to model complex patterns.",
                "Machine learning is a branch of artificial intelligence.",
                "Pizza is a traditional Italian dish made with dough, tomato sauce, and cheese.",
                "Deep learning has revolutionized computer vision and natural language processing.",
                "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
                "Neural networks are the foundation of deep learning algorithms.",
                "Climate change refers to long-term shifts in global temperatures.",
                "Convolutional neural networks are widely used in image recognition.",
                "The Python programming language was created by Guido van Rossum.",
                "Recurrent neural networks are designed for sequential data processing.",
                "Quantum computing leverages quantum mechanical phenomena.",
                "Transfer learning allows models to leverage pre-trained knowledge.",
            ],
            [2, 1, 0, 2, 0, 2, 0, 1, 0, 1, 0, 1]  # relevance scores
        ),
        # Query 2: Machine learning applications
        (
            "What are applications of machine learning?",
            [
                "Machine learning is used in recommendation systems like Netflix.",
                "The Great Wall of China is a series of fortifications.",
                "Natural language processing enables computers to understand human language.",
                "Pizza is a traditional Italian dish.",
                "Computer vision allows machines to interpret visual information.",
                "The human brain contains approximately 86 billion neurons.",
                "Machine learning powers voice assistants like Siri and Alexa.",
                "Renewable energy sources include solar and wind power.",
                "Reinforcement learning is used in game playing and robotics.",
                "Shakespeare wrote 37 plays during his lifetime.",
            ],
            [2, 0, 2, 0, 1, 0, 2, 0, 1, 0]
        ),
        # Query 3: Neural networks
        (
            "How do neural networks work?",
            [
                "Neural networks consist of interconnected nodes organized in layers.",
                "The Python programming language was created by Guido van Rossum in 1991.",
                "Backpropagation is the algorithm used to train neural networks.",
                "Pizza is a traditional Italian dish.",
                "Activation functions introduce non-linearity to neural networks.",
                "Climate change refers to long-term shifts in temperatures.",
                "Deep learning uses neural networks with multiple layers.",
                "The Eiffel Tower is in Paris, France.",
                "Gradient descent optimizes neural network weights.",
                "Quantum computing leverages quantum phenomena.",
            ],
            [2, 0, 2, 0, 1, 0, 2, 0, 1, 0]
        ),
        # Query 4: Climate change
        (
            "What causes climate change?",
            [
                "Climate change is primarily caused by greenhouse gas emissions from human activities.",
                "Machine learning is a branch of artificial intelligence.",
                "Rising global temperatures are linked to increased CO2 concentrations.",
                "Deep learning uses neural networks with multiple layers.",
                "Deforestation contributes to climate change by reducing carbon absorption.",
                "Pizza is a traditional Italian dish.",
                "The Paris Agreement aims to limit global warming to 1.5 degrees.",
                "Neural networks are the foundation of deep learning.",
                "Renewable energy sources can help mitigate climate change.",
                "The Eiffel Tower is a wrought-iron lattice tower.",
            ],
            [2, 0, 2, 0, 1, 0, 2, 0, 1, 0]
        ),
        # Query 5: Italian cuisine
        (
            "What is traditional Italian food?",
            [
                "Pizza is a traditional Italian dish made with dough, tomato sauce, and cheese.",
                "Deep learning uses neural networks with multiple layers.",
                "Pasta is a staple of Italian cuisine with many regional varieties.",
                "Machine learning is a branch of artificial intelligence.",
                "Italian cuisine emphasizes fresh, high-quality ingredients.",
                "The Eiffel Tower is in Paris, France.",
                "Risotto is a traditional Italian rice dish.",
                "Neural networks are the foundation of deep learning.",
                "Tiramisu is a popular Italian dessert.",
                "Climate change refers to long-term shifts in temperatures.",
            ],
            [2, 0, 2, 0, 1, 0, 2, 0, 1, 0]
        ),
    ]
    return test_cases


def benchmark_quality(model_name, test_cases, k=10):
    """Run quality benchmark for a model."""
    reranker = Reranker(model_name, offline=True, show_download_progress=False)

    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for query, docs, relevances in test_cases:
        # Get reranker scores
        results = reranker.rerank(query, docs)

        # Build (score, relevance) pairs
        scores_with_relevance = []
        for i, res in enumerate(results):
            scores_with_relevance.append((res.score, relevances[res.index]))

        # Compute metrics
        ndcg = compute_ndcg(scores_with_relevance, k)
        mrr = compute_mrr(scores_with_relevance)
        recall = compute_recall(scores_with_relevance, k)

        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)
        recall_scores.append(recall)

    reranker.close()

    # Average across queries
    n = len(test_cases)
    return {
        'ndcg@10': sum(ndcg_scores) / n,
        'mrr': sum(mrr_scores) / n,
        'recall@10': sum(recall_scores) / n,
        'ndcg_scores': ndcg_scores,
        'mrr_scores': mrr_scores,
        'recall_scores': recall_scores,
    }


def main():
    parser = argparse.ArgumentParser(description='Quality benchmark: FP32 vs INT8')
    parser.add_argument('--k', type=int, default=10, help='Top-k for metrics')
    args = parser.parse_args()

    test_cases = create_test_queries()

    print("=" * 70)
    print("Quality Benchmark: FP32 vs INT8 for Reranking")
    print("=" * 70)
    print(f"Queries: {len(test_cases)}")
    print(f"Metrics: NDCG@{args.k}, MRR, Recall@{args.k}")
    print()

    # FP32
    print("--- FP32 ---")
    fp32 = benchmark_quality("jinaai/jina-reranker-v1-turbo-en", test_cases, args.k)
    print(f"  NDCG@{args.k}: {fp32['ndcg@10']:.4f}")
    print(f"  MRR: {fp32['mrr']:.4f}")
    print(f"  Recall@{args.k}: {fp32['recall@10']:.4f}")
    print()

    # INT8
    print("--- INT8 (quantized) ---")
    int8 = benchmark_quality("jinaai/jina-reranker-v1-turbo-en-quantized", test_cases, args.k)
    print(f"  NDCG@{args.k}: {int8['ndcg@10']:.4f}")
    print(f"  MRR: {int8['mrr']:.4f}")
    print(f"  Recall@{args.k}: {int8['recall@10']:.4f}")
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Metric':<15} | {'FP32':>10} | {'INT8':>10} | {'Diff':>10} | {'Diff %':>10}")
    print("-" * 60)

    for metric in ['ndcg@10', 'mrr', 'recall@10']:
        fp32_val = fp32[metric]
        int8_val = int8[metric]
        diff = int8_val - fp32_val
        diff_pct = (diff / fp32_val * 100) if fp32_val > 0 else 0
        print(f"{metric:<15} | {fp32_val:>10.4f} | {int8_val:>10.4f} | {diff:>+10.4f} | {diff_pct:>+9.2f}%")
    print()

    # Decision
    ndcg_diff_pct = abs((int8['ndcg@10'] - fp32['ndcg@10']) / fp32['ndcg@10'] * 100) if fp32['ndcg@10'] > 0 else 0

    print("=" * 70)
    print("DECISION")
    print("=" * 70)
    print()
    print(f"NDCG@{args.k} difference: {ndcg_diff_pct:.2f}%")
    print()

    if ndcg_diff_pct < 1.0:
        print("=> NDCG diff < 1%: INT8 quality is acceptable.")
        print("   RECOMMENDATION: Set DEFAULT = INT8 quantized")
    elif ndcg_diff_pct < 3.0:
        print("=> NDCG diff 1-3%: INT8 quality is borderline.")
        print("   RECOMMENDATION: Keep DEFAULT = FP32, offer INT8 as FAST option")
    else:
        print("=> NDCG diff > 3%: INT8 quality is too degraded.")
        print("   RECOMMENDATION: DEFAULT = FP32 only")
    print()

    # Per-query breakdown
    print("=" * 70)
    print("PER-QUERY BREAKDOWN")
    print("=" * 70)
    print()
    print(f"{'Query':<40} | {'FP32 NDCG':>10} | {'INT8 NDCG':>10} | {'Diff':>8}")
    print("-" * 75)
    for i, (query, _, _) in enumerate(test_cases):
        fp32_n = fp32['ndcg_scores'][i]
        int8_n = int8['ndcg_scores'][i]
        diff = int8_n - fp32_n
        short_query = query[:37] + "..." if len(query) > 40 else query
        print(f"{short_query:<40} | {fp32_n:>10.4f} | {int8_n:>10.4f} | {diff:>+8.4f}")
    print()

    # Markdown summary
    print("## Results (for markdown)")
    print()
    print("| Metric | FP32 | INT8 | Diff |")
    print("|--------|------|------|------|")
    for metric in ['ndcg@10', 'mrr', 'recall@10']:
        fp32_val = fp32[metric]
        int8_val = int8[metric]
        diff = int8_val - fp32_val
        print(f"| {metric} | {fp32_val:.4f} | {int8_val:.4f} | {diff:+.4f} |")


if __name__ == '__main__':
    main()
