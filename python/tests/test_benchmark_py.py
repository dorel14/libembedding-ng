"""Unit tests for the unified benchmark Python API.

These tests must fail when the API is broken: every test asserts a real
property of the returned value (no bare `return True`).
"""

import os

import pytest


def test_import():
    """Test that benchmark module imports cleanly."""
    import libembedding.benchmark as bench

    assert bench is not None
    for name in ("Benchmark", "BenchmarkResult", "ComparisonResult", "Metrics"):
        assert hasattr(bench, name), f"Missing export: {name}"


def test_hardware_detection():
    """Test hardware detection."""
    from libembedding.benchmark import detect_hardware

    hw = detect_hardware()
    assert isinstance(hw.cpu_name, str)
    assert hw.cpu_name, "cpu_name must not be empty"
    assert hw.logical_cores >= 1
    assert hw.physical_cores >= 1
    assert hw.ram_mb >= 0
    assert hw.os_name, "os_name must not be empty"


def test_cache_path():
    """Test cache path retrieval."""
    from libembedding.benchmark import cache_path

    path = cache_path()
    assert isinstance(path, str)
    assert path, "cache path must not be empty"
    assert os.path.isabs(path), f"cache path must be absolute, got {path!r}"


def test_clear_cache():
    """Test cache clearing: removes the cache file and stays idempotent."""
    from libembedding.benchmark import cache_path, clear_cache, detect_hardware

    clear_cache()
    path = cache_path()
    assert not os.path.exists(path), f"cache file {path} still present after clear_cache()"

    # Clearing twice must not raise
    clear_cache()

    # Clearing an empty cache must not break hardware detection
    assert detect_hardware().logical_cores >= 1


def test_corpus_type_values():
    """CorpusType values must match the C enum."""
    from libembedding.benchmark import CorpusType

    assert CorpusType.SHORT == 0
    assert CorpusType.MEDIUM == 1
    assert CorpusType.LONG == 2
    assert CorpusType.VERY_LONG == 3
    assert CorpusType.MIXED == 4
    assert CorpusType.MULTILINGUAL == 5
    assert CorpusType.EDGE_CASES == 6


def test_objective_values():
    """Objective values must match the C enum in autotuner.h."""
    from libembedding.benchmark import Objective

    assert Objective.LATENCY == 0
    assert Objective.THROUGHPUT == 1
    assert Objective.BALANCED == 2
    assert Objective.MEMORY == 3


def test_benchmark_hardware_is_cached():
    """Benchmark.hardware must be detected once and reused."""
    from libembedding.benchmark import Benchmark, detect_hardware

    bench = Benchmark()
    first = bench.hardware
    assert first is bench.hardware, "hardware must be cached after the first access"
    assert first.cpu_name == detect_hardware().cpu_name


def test_benchmark_result_str():
    """BenchmarkResult.__str__ must render every metric."""
    from libembedding.benchmark import BenchmarkResult

    result = BenchmarkResult(
        model_name="/models/bge-small",
        backend="onnx",
        throughput_docs_sec=1234.5,
        latency_p50_ms=1.25,
        latency_p95_ms=2.5,
        peak_memory_mb=512.0,
    )
    rendered = str(result)
    for expected in ("bge-small", "onnx", "1234.5", "1.25", "2.50", "512"):
        assert expected in rendered, f"{expected!r} missing from {rendered!r}"


def test_comparison_summary_without_results():
    """A comparison with no result must not recommend anything."""
    from libembedding.benchmark import ComparisonResult

    summary = ComparisonResult().summary()
    assert "Unified Backend Comparison" in summary
    assert "Recommendation" not in summary


def test_comparison_summary_sorts_by_throughput():
    """Results must be listed from the fastest to the slowest."""
    from libembedding.benchmark import BenchmarkResult, ComparisonResult

    comparison = ComparisonResult(
        results=[
            BenchmarkResult(model_name="slow", backend="onnx", throughput_docs_sec=10.0),
            BenchmarkResult(model_name="fast", backend="onnx", throughput_docs_sec=99.0),
        ],
        recommendation=BenchmarkResult(
            model_name="fast", backend="onnx", throughput_docs_sec=99.0
        ),
    )
    summary = comparison.summary()
    assert "Recommendation: fast" in summary
    assert summary.index("fast") < summary.index("slow")


def test_autotune_missing_model_raises():
    """Autotuning a non-existent model must raise, not return a fake result."""
    from libembedding.benchmark import Benchmark
    from libembedding.exceptions import LembedError

    bench = Benchmark()
    missing = os.path.join(os.sep, "nonexistent", "libembedding", "model.onnx")
    with pytest.raises(LembedError):
        bench.run(missing, "onnx")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
