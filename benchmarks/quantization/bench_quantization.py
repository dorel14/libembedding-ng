"""Benchmark quantization modes for ONNX embedding models.

Compares FP32 (none), FP16 (static), and dynamic quantization
across different model sizes and batch sizes.

Auteur: David Orel
Version: 1.8.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
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

from libembedding import TextEmbedding


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


class MemoryMonitor:
    """Monitor peak memory usage during inference in a separate thread."""
    
    def __init__(self, interval_ms: int = 10):
        self.interval_ms = interval_ms
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
    
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _monitor(self):
        while not self._stop.is_set():
            try:
                mem = self._get_process_memory_mb()
                if mem > self.peak_mb:
                    self.peak_mb = mem
            except Exception:
                pass
            time.sleep(self.interval_ms / 1000.0)
    
    def _get_process_memory_mb(self) -> float:
        if not psutil:
            return 0.0
        
        if os.name == "nt":
            with contextlib.suppress(Exception):
                import ctypes
                psapi = ctypes.windll.psapi
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
                        ok = psapi.GetProcessMemoryInfo(
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

        # Start memory monitoring thread
        mem_monitor = MemoryMonitor(interval_ms=10)
        mem_monitor.start()

        # Timed run
        t0 = time.perf_counter()
        model.embed(texts, batch_size=batch_size)
        t1 = time.perf_counter()

        # Stop memory monitoring
        mem_monitor.stop()
        peak_mem = mem_monitor.peak_mb

        elapsed_ms = (t1 - t0) * 1000
        docs_per_sec = num_docs / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
        latency_ms = elapsed_ms / num_docs if num_docs > 0 else 0.0
    except Exception as exc:  # noqa: BLE001 - a benchmark records any failure
        mem_monitor.stop()
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


def generate_corpus(n: int = 1000) -> list[str]:
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


def format_justification(baseline_docs_per_sec: float, variant_docs_per_sec: float, 
                         baseline_mem: float, variant_mem: float) -> str:
    """Generate a justification string comparing variant to baseline."""
    if baseline_docs_per_sec <= 0:
        return "Baseline failed"
    
    speed_ratio = variant_docs_per_sec / baseline_docs_per_sec
    mem_ratio = variant_mem / baseline_mem if baseline_mem > 0 else 1.0
    
    parts = []
    if speed_ratio > 1.05:
        parts.append(f"{speed_ratio:.1f}x faster")
    elif speed_ratio < 0.95:
        parts.append(f"{1/speed_ratio:.1f}x slower")
    else:
        parts.append("similar speed")
    
    if mem_ratio < 0.95:
        parts.append(f"{1/mem_ratio:.1f}x less RAM")
    elif mem_ratio > 1.05:
        parts.append(f"{mem_ratio:.1f}x more RAM")
    else:
        parts.append("similar RAM")
    
    return ", ".join(parts)


def write_html_report(results: list[QuantizationResult], output_path: Path) -> None:
    """Write an HTML report with comparison tables."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Quantization Benchmark Results</title>
    <style>
        body { font-family: monospace; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
        th { background-color: #f2f2f2; text-align: center; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .model-header { text-align: left; }
        .justification { font-size: 0.9em; color: #555; max-width: 300px; }
        h1, h2 { color: #333; }
    </style>
</head>
<body>
    <h1>Quantization Benchmark Results</h1>
"""
    
    for model_name in sorted({r.model_name for r in results if r.success}):
        model_results = [r for r in results if r.model_name == model_name and r.success]
        baseline = next((r for r in model_results if r.quantization == "none"), None)
        
        html += f'<h2>{model_name}</h2>\n'
        html += '<table>\n'
        html += '<tr><th>Batch Size</th><th>Quantization</th><th>docs/s</th><th>ms/doc</th><th>Peak MB</th><th class="justification">Justification</th></tr>\n'
        
        for bs in sorted({r.batch_size for r in model_results}):
            batch_results = [r for r in model_results if r.batch_size == bs]
            for r in sorted(batch_results, key=lambda x: x.quantization):
                just = ""
                if baseline and r.quantization != "none":
                    just = format_justification(
                        baseline.texts_per_second, r.texts_per_second,
                        baseline.peak_memory_mb, r.peak_memory_mb
                    )
                html += f'<tr><td>{r.batch_size}</td><td>{r.quantization}</td><td>{r.texts_per_second:.1f}</td><td>{r.latency_ms_per_doc:.2f}</td><td>{r.peak_memory_mb:.1f}</td><td class="justification">{just}</td></tr>\n'
        
        html += '</table>\n'
    
    html += """</body>
</html>"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


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
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[8, 32, 64, 128],
        help="Batch sizes to test (default: 8 32 64 128)",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        choices=[1000, 5000],
        default=1000,
        help="Number of texts to embed (default: 1000, choices: 1000, 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quantization/results.json",
        help="Output JSON file for results (relative to repo root)",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="quantization/results.html",
        help="Output HTML file for results (relative to repo root)",
    )
    args = parser.parse_args()

    # Generate full corpus (including warmup portion)
    texts = generate_corpus(args.num_texts)
    
    # Explicit warmup: 5% of corpus, non-timed
    warmup_size = max(1, args.num_texts // 20)
    warmup_texts = texts[:warmup_size]
    timed_texts = texts[warmup_size:]
    
    all_results: list[QuantizationResult] = []

    print(f"Benchmarking {len(args.models)} model(s) x {len(args.quantizations)} quantization(s)")
    print(f"Texts: {len(timed_texts)} timed (+ {warmup_size} warmup), Batch sizes: {args.batch_sizes}")
    print("-" * 80)

    for model_name in args.models:
        for quant in args.quantizations:
            for bs in args.batch_sizes:
                print(
                    f"  {model_name} [{quant}] batch={bs} ...",
                    end=" ", flush=True,
                )
                # Run warmup first (not timed)
                try:
                    if quant == "auto":
                        warmup_model = TextEmbedding(
                            model_name,
                            provider="cpu",
                            batch_size=bs,
                            preferred_quantization="auto",
                            show_download_progress=False,
                        )
                    else:
                        warmup_model = TextEmbedding(
                            model_name,
                            provider="cpu",
                            batch_size=bs,
                            quantization=quant,
                            show_download_progress=False,
                        )
                    warmup_model.embed(warmup_texts)
                    warmup_model.close()
                except Exception:
                    pass  # Warmup failures are non-fatal
                
                # Timed benchmark
                result = benchmark_quantization(
                    model_name, quant, timed_texts, batch_size=bs
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

    # Save JSON results
    output_path = _REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [asdict(r) for r in all_results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Save HTML report
    html_path = _REPO_ROOT / args.output_html
    write_html_report(all_results, html_path)
    print(f"HTML report saved to {html_path}")

    # Print recommendations with justifications
    print("\nRecommendations:")
    for model_name in sorted({r.model_name for r in all_results if r.success}):
        model_results = [r for r in all_results if r.model_name == model_name and r.success]
        if model_results:
            baseline = next((r for r in model_results if r.quantization == "none"), None)
            
            # Best throughput
            best_throughput = max(model_results, key=lambda r: r.texts_per_second)
            just = ""
            if baseline and best_throughput.quantization != "none":
                just = f" ({format_justification(baseline.texts_per_second, best_throughput.texts_per_second, baseline.peak_memory_mb, best_throughput.peak_memory_mb)})"
            print(
                f"  {model_name}: best throughput = "
                f"{best_throughput.quantization} "
                f"({best_throughput.texts_per_second:.1f} docs/s){just}"
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
