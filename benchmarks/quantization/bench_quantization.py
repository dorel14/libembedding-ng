"""Benchmark quantization modes for ONNX embedding models.

Compares FP32 (none), FP16 (static), and dynamic quantization
across different model sizes and batch sizes.

Auteur: David Orel
Version: 1.5.8

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# Add repo root to path for benchmarks
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class QuantizationResult:
    model_name: str
    quantization: str
    batch_size: int
    texts_per_second: float
    latency_ms_per_doc: float
    peak_memory_mb: float
    dim: int
    num_docs: int
    success: bool
    error: str = ""


def get_peak_memory_mb() -> float:
    if psutil:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def benchmark_quantization(
    model_name: str,
    quantization: str,
    texts: list[str],
    batch_size: int = 64,
    warmup: int = 2,
) -> QuantizationResult:
    """Benchmark a model with a given quantization mode."""
    from libembedding import TextEmbedding

    start_mem = get_peak_memory_mb()

    try:
        model = TextEmbedding(
            model_name,
            provider="cpu",
            batch_size=batch_size,
            quantization=quantization,
            show_download_progress=False,
        )
    except Exception as e:
        return QuantizationResult(
            model_name=model_name,
            quantization=quantization,
            batch_size=batch_size,
            texts_per_second=0,
            latency_ms_per_doc=0,
            peak_memory_mb=0,
            dim=0,
            num_docs=0,
            success=False,
            error=str(e),
        )

    dim = model.dim
    num_docs = len(texts)

    # Warmup
    try:
        for i in range(warmup):
            chunk = texts[i * 32 : (i + 1) * 32] if i < len(texts) // 32 else texts[:32]
            if chunk:
                model.embed(chunk)
    except Exception:
        pass

    # Timed run
    t0 = time.perf_counter()
    result = model.embed(texts, batch_size=batch_size)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    docs_per_sec = num_docs / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    latency_ms = elapsed_ms / num_docs if num_docs > 0 else 0

    end_mem = get_peak_memory_mb()
    peak_mem = max(end_mem - start_mem, end_mem)

    model.close()

    return QuantizationResult(
        model_name=model_name,
        quantization=quantization,
        batch_size=batch_size,
        texts_per_second=docs_per_sec,
        latency_ms_per_doc=latency_ms,
        peak_memory_mb=peak_mem,
        dim=dim,
        num_docs=num_docs,
        success=True,
    )


def generate_corpus(n: int = 100) -> list[str]:
    """Generate a representative corpus of varied text lengths."""
    templates = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
        "Embeddings are dense vector representations of text that capture semantic meaning and can be used for similarity search, classification, and clustering tasks in natural language processing applications.",
        "The transformer architecture, introduced in the paper 'Attention Is All You Need', revolutionized natural language processing by using self-attention mechanisms to process sequential data in parallel.",
        "Retrieval-augmented generation (RAG) combines the benefits of parametric and non-parametric memory to produce more accurate and factual responses by retrieving relevant information from a knowledge base before generating a response.",
    ]
    texts = []
    for i in range(n):
        base = templates[i % len(templates)]
        texts.append(f"[{i}] {base}")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantization modes")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
        help="Models to benchmark (HuggingFace IDs or local paths)",
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        choices=["none", "static", "dynamic"],
        default=["none", "static", "dynamic"],
        help="Quantization modes to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=100,
        help="Number of texts to embed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/quantization/results.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    texts = generate_corpus(args.num_texts)
    all_results: list[QuantizationResult] = []

    print(f"Benchmarking {len(args.models)} model(s) x {len(args.quantizations)} quantization(s)")
    print(f"Texts: {args.num_texts}, Batch size: {args.batch_size}")
    print("-" * 80)

    for model_name in args.models:
        for quant in args.quantizations:
            print(f"  {model_name} [{quant}] ...", end=" ", flush=True)
            result = benchmark_quantization(
                model_name, quant, texts, batch_size=args.batch_size
            )
            all_results.append(result)

            if result.success:
                print(
                    f"{result.texts_per_second:.1f} docs/s | "
                    f"{result.latency_ms_per_doc:.2f} ms/doc | "
                    f"{result.peak_memory_mb:.1f} MB"
                )
            else:
                print(f"FAILED: {result.error}")

    print("-" * 80)

    # Summary table
    print("\nSummary:")
    print(f"{'Model':<45} {'Quantization':<12} {'docs/s':>10} {'ms/doc':>10} {'MB':>10}")
    print("-" * 90)
    for r in all_results:
        if r.success:
            print(
                f"{r.model_name:<45} {r.quantization:<12} "
                f"{r.texts_per_second:>10.1f} {r.latency_ms_per_doc:>10.2f} "
                f"{r.peak_memory_mb:>10.1f}"
            )

    # Save results
    output_path = _REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [asdict(r) for r in all_results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print recommendations
    print("\nRecommendations:")
    for model_name in set(r.model_name for r in all_results if r.success):
        model_results = [r for r in all_results if r.model_name == model_name and r.success]
        if model_results:
            # Best throughput
            best_throughput = max(model_results, key=lambda r: r.texts_per_second)
            print(
                f"  {model_name}: best throughput = "
                f"{best_throughput.quantization} "
                f"({best_throughput.texts_per_second:.1f} docs/s)"
            )

            # Best latency
            best_latency = min(model_results, key=lambda r: r.latency_ms_per_doc)
            print(
                f"  {model_name}: best latency = "
                f"{best_latency.quantization} "
                f"({best_latency.latency_ms_per_doc:.2f} ms/doc)"
            )

            # Best memory
            best_mem = min(model_results, key=lambda r: r.peak_memory_mb)
            if best_mem.peak_memory_mb > 0:
                print(
                    f"  {model_name}: lowest memory = "
                    f"{best_mem.quantization} "
                    f"({best_mem.peak_memory_mb:.1f} MB)"
                )


if __name__ == "__main__":
    main()
