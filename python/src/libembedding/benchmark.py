"""Unified Backend Benchmark for libembedding-ng.

Provides common protocol for comparing ONNX vs llama.cpp backends.
Self-contained module ready for integration.

Usage:
    from libembedding.benchmark import Benchmark, CorpusType, Objective

    bench = Benchmark()
    print(bench.hardware)
    result = bench.autotune("path/to/model.gguf", "llama.cpp")
    print(result)
Auteur: David Orel
Version: 1.6.0

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum

from ._binding import ffi, lib
from ._status import check_status


class CorpusType(IntEnum):
    """Test corpus categories."""

    SHORT = 0  # < 20 tokens
    MEDIUM = 1  # 20-80 tokens
    LONG = 2  # 80-200 tokens
    VERY_LONG = 3  # 200+ tokens
    MIXED = 4  # All lengths
    MULTILINGUAL = 5  # Multi-language
    EDGE_CASES = 6  # Edge cases


class Objective(IntEnum):
    """Optimization objectives (must match autotuner.h)."""

    LATENCY = 0
    THROUGHPUT = 1
    BALANCED = 2
    MEMORY = 3


@dataclass
class Metrics:
    """Benchmark metrics for one run."""

    throughput_docs_sec: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    load_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    dim: int = 0
    num_texts: int = 0
    num_errors: int = 0


@dataclass
class BenchmarkResult:
    """Result for one model x backend combination."""

    model_name: str
    backend: str
    throughput_docs_sec: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    peak_memory_mb: float = 0.0
    dim: int = 0
    sessions: int = 1
    threads: int = 1

    def __str__(self) -> str:
        return (
            f"{self.model_name} [{self.backend}]\n"
            f"  Sessions: {self.sessions}, Threads: {self.threads}\n"
            f"  Throughput: {self.throughput_docs_sec:.1f} docs/s\n"
            f"  Latency p50: {self.latency_p50_ms:.2f} ms\n"
            f"  Latency p95: {self.latency_p95_ms:.2f} ms\n"
            f"  Memory: {self.peak_memory_mb:.0f} MB"
        )


@dataclass
class ComparisonResult:
    """Full comparison results with recommendation."""

    results: list[BenchmarkResult] = field(default_factory=list)
    recommendation: BenchmarkResult | None = None

    def summary(self) -> str:
        """Generate formatted comparison table."""
        lines = [
            "=" * 72,
            "Unified Backend Comparison",
            "=" * 72,
            f"{'Model':<20} {'Backend':<12} {'Docs/s':>8} {'P50':>8} {'P95':>8} {'RAM':>8}",
            "-" * 72,
        ]
        for r in sorted(self.results, key=lambda x: -x.throughput_docs_sec):
            name = os.path.basename(r.model_name)[:19]
            lines.append(
                f"{name:<20} {r.backend:<12} "
                f"{r.throughput_docs_sec:>8.1f} "
                f"{r.latency_p50_ms:>8.1f} "
                f"{r.latency_p95_ms:>8.1f} "
                f"{r.peak_memory_mb:>8.0f}"
            )
        if self.recommendation:
            lines.extend(
                [
                    "-" * 72,
                    (
                        f"Recommendation: {os.path.basename(self.recommendation.model_name)} "
                        f"[{self.recommendation.backend}]"
                    ),
                    f"  Throughput: {self.recommendation.throughput_docs_sec:.1f} docs/s",
                    f"  Latency p50: {self.recommendation.latency_p50_ms:.2f} ms",
                    f"  Memory: {self.recommendation.peak_memory_mb:.0f} MB",
                ]
            )
        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass
class HardwareInfo:
    """Detected hardware information."""

    cpu_name: str
    physical_cores: int
    logical_cores: int
    os_name: str = ""
    ram_mb: int = 0
    features: str = ""


def detect_hardware() -> HardwareInfo:
    """Detect hardware fingerprint."""
    hw = ffi.new("lembed_cache_hardware_info_t*")
    check_status(lib.lembed_cache_detect_hardware(hw))
    return HardwareInfo(
        cpu_name=ffi.string(hw.cpu_name).decode(),
        physical_cores=hw.physical_cores,
        logical_cores=hw.logical_cores,
        os_name=ffi.string(hw.os_name).decode() if hw.os_name else "",
        ram_mb=hw.ram_mb,
        features=ffi.string(hw.features).decode() if hw.features else "",
    )


def clear_cache() -> None:
    """Clear all tuning cache entries."""
    check_status(lib.lembed_tune_cache_clear())


def cache_path() -> str:
    """Get tuning cache file path."""
    return ffi.string(lib.lembed_tune_cache_path()).decode()


class Benchmark:
    """Unified backend benchmark.

    Provides common protocol for comparing ONNX vs llama.cpp backends
    on the same corpus with auto-tuning.
    """

    def __init__(self):
        self._hardware: HardwareInfo | None = None

    @property
    def hardware(self) -> HardwareInfo:
        """Detect hardware fingerprint."""
        if self._hardware is None:
            self._hardware = detect_hardware()
        return self._hardware

    def autotune(
        self,
        model_path: str,
        backend: str,
        objective: Objective = Objective.BALANCED,
    ) -> BenchmarkResult:
        """Auto-tune a backend for a model."""
        result = ffi.new("lembed_benchmark_result_t*")
        check_status(
            lib.lembed_benchmark_autotune(
                model_path.encode(),
                backend.encode(),
                int(objective),
                result,
            )
        )
        return BenchmarkResult(
            model_name=ffi.string(result.model_name).decode(),
            backend=ffi.string(result.backend).decode(),
            throughput_docs_sec=result.metrics.throughput_docs_sec,
            latency_p50_ms=result.metrics.latency_p50_ms,
            latency_p95_ms=result.metrics.latency_p95_ms,
            peak_memory_mb=result.metrics.peak_memory_mb,
            dim=result.metrics.dim,
            sessions=result.config.batch_size,
            threads=result.config.num_threads,
        )

    def run(
        self,
        model_path: str,
        backend: str,
        corpus: CorpusType = CorpusType.MIXED,
        sessions: int = 4,
        threads: int = 1,
    ) -> BenchmarkResult:
        """Run benchmark for one model on one backend."""
        config = ffi.new("lembed_backend_config_t*")
        config.backend = backend.encode()
        config.num_threads = threads
        config.batch_size = sessions
        config.workers = 1

        result = ffi.new("lembed_benchmark_result_t*")
        check_status(
            lib.lembed_benchmark_run(
                model_path.encode(),
                backend.encode(),
                int(corpus),
                config,
                result,
            )
        )
        return BenchmarkResult(
            model_name=ffi.string(result.model_name).decode(),
            backend=ffi.string(result.backend).decode(),
            throughput_docs_sec=result.metrics.throughput_docs_sec,
            latency_p50_ms=result.metrics.latency_p50_ms,
            latency_p95_ms=result.metrics.latency_p95_ms,
            peak_memory_mb=result.metrics.peak_memory_mb,
            dim=result.metrics.dim,
            sessions=sessions,
            threads=threads,
        )

    def compare_all(
        self,
        onnx_path: str | None = None,
        gguf_path: str | None = None,
        objective: Objective = Objective.BALANCED,
    ) -> ComparisonResult:
        """Compare all available backends for a model."""
        comparison = ComparisonResult()

        if gguf_path and os.path.exists(gguf_path):
            result = self.autotune(gguf_path, "llama.cpp", objective)
            comparison.results.append(result)

        if onnx_path and os.path.exists(onnx_path):
            result = self.autotune(onnx_path, "onnx", objective)
            comparison.results.append(result)

        if comparison.results:
            comparison.recommendation = max(
                comparison.results, key=lambda r: r.throughput_docs_sec
            )

        return comparison

    def sweep(
        self,
        models: dict[str, str],
        corpora: list[CorpusType] | None = None,
    ) -> dict[str, list[BenchmarkResult]]:
        """Run benchmark sweep across models and corpora."""
        if corpora is None:
            corpora = [
                CorpusType.SHORT,
                CorpusType.MEDIUM,
                CorpusType.LONG,
                CorpusType.MIXED,
            ]

        results = {}
        for name, path in models.items():
            if not os.path.exists(path):
                continue
            # Determine backend from file extension
            if path.endswith(".onnx"):
                backend = "onnx"
                # ONNX expects a directory, not a file path
                model_path = os.path.dirname(path)
            else:
                backend = "llama.cpp"
                model_path = path
            model_results = []
            for corpus in corpora:
                result = self.run(model_path, backend, corpus)
                model_results.append(result)
            results[name] = model_results
        return results
