"""Benchmark quantization modes for ONNX embedding models.

Compares FP32 (none), FP16 (static), and dynamic quantization
across different model sizes and batch sizes.

Auteur: David Orel
Version: 1.5.8

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
    """Return peak resident memory of this process, in MB.

    On Windows the real peak is read from GetProcessMemoryInfo so the
    value reflects the high-water mark reached during inference rather
    than the instantaneous working set.
    """
    if not psutil:
        return 0.0

    if os.name == "nt":
        with contextlib.suppress(Exception):
            import ctypes

            kernel32 = ctypes.windll.kernel32
            process_query_information = 0x0400
            process_vm_read = 0x0010

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            handle = kernel32.OpenProcess(
                process_query_information | process_vm_read,
                False,
                os.getpid(),
            )
            if handle:
                try:
                    counters = PROCESS_MEMORY_COUNTERS()
                    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    ok = kernel32.GetProcessMemoryInfo(
                        handle, ctypes.byref(counters), counters.cb
                    )
                    if ok:
                        return counters.PeakWorkingSetSize / (1024 * 1024)
                finally:
                    kernel32.CloseHandle(handle)

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def benchmark_quantization(
    model_name: str,
    quantization: str,
    texts: list[str],
    batch_size: int = 64,
    warmup: int = 2,
) -> QuantizationResult:
    """Benchmark a model with a given quantization mode."""
    from libembedding import TextEmbedding

    def _failed(error: str, dim: int = 0) -> QuantizationResult:
        return QuantizationResult(
            model_name=model_name,
            quantization=quantization,
            batch_size=batch_size,
            texts_per_second=0.0,
            latency_ms_per_doc=0.0,
            peak_memory_mb=0.0,
            dim=dim,
            num_docs=0,
            success=False,
            error=error,
        )

    try:
        if quantization == "auto":
            model = TextEmbedding(
                model_name,
                provider="cpu",
                batch_size=batch_size,
                preferred_quantization="auto",
                show_download_progress=False,
            )
        else:
            model = TextEmbedding(
                model_name,
                provider="cpu",
                batch_size=batch_size,
                quantization=quantization,
                show_download_progress=False,
            )
    except Exception as exc:  # noqa: BLE001 - a benchmark records any failure
        return _failed(str(exc))

    dim = model.dim
    num_docs = len(texts)

    try:
        # Warmup (failures here are non-fatal: the timed run reports them)
        with contextlib.suppress(Exception):
            for i in range(warmup):
                chunk = (
                    texts[i * 32 : (i + 1) * 32]
                    if i < len(texts) // 32
                    else texts[:32]
                )
                if chunk:
                    model.embed(chunk)

        # Measure peak memory around the timed run
        peak_mem = get_peak_memory_mb()

        t0 = time.perf_counter()
        model.embed(texts, batch_size=batch_size)
        t1 = time.perf_counter()

        peak_mem = max(peak_mem, get_peak_memory_mb())

        elapsed_ms = (t1 - t0) * 1000
        docs_per_sec = num_docs / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
        latency_ms = elapsed_ms / num_docs if num_docs > 0 else 0.0
    except Exception as exc:  # noqa: BLE001 - a benchmark records any failure
        return _failed(str(exc), dim)
    finally:
        with contextlib.suppress(Exception):
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
        choices=["none", "static", "dynamic", "auto"],
        default=["none", "static", "dynamic"],
        help="Quantization modes to test (default: none static dynamic)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[8, 32, 64, 128],
        help="Batch sizes to test (default: 8 32 64 128)",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=1000,
        help="Number of texts to embed (default: 1000)",
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
    print(f"Texts: {args.num_texts}, Batch sizes: {args.batch_sizes}")
    print("-" * 80)

    for model_name in args.models:
        for quant in args.quantizations:
            for bs in args.batch_sizes:
                print(
                    f"  {model_name} [{quant}] batch={bs} ...",
                    end=" ", flush=True,
                )
                result = benchmark_quantization(
                    model_name, quant, texts, batch_size=bs
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
    for model_name in {r.model_name for r in all_results if r.success}:
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
