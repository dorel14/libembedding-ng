"""Unit tests for libembedding benchmark module."""

import pytest

from libembedding.benchmark import (
    Benchmark,
    BenchmarkResult,
    ComparisonResult,
    CorpusType,
    HardwareInfo,
    Metrics,
    Objective,
    cache_path,
    clear_cache,
    detect_hardware,
)


def test_corpus_type_enum():
    assert CorpusType.SHORT == 0
    assert CorpusType.MEDIUM == 1
    assert CorpusType.LONG == 2
    assert CorpusType.VERY_LONG == 3
    assert CorpusType.MIXED == 4
    assert CorpusType.MULTILINGUAL == 5
    assert CorpusType.EDGE_CASES == 6


def test_objective_enum():
    assert Objective.LATENCY == 0
    assert Objective.THROUGHPUT == 1
    assert Objective.BALANCED == 2
    assert Objective.MEMORY == 3


def test_metrics_defaults():
    m = Metrics()
    assert m.throughput_docs_sec == 0.0
    assert m.latency_p50_ms == 0.0
    assert m.dim == 0
    assert m.num_texts == 0


def test_benchmark_result_str():
    result = BenchmarkResult(
        model_name="test-model",
        backend="onnx",
        throughput_docs_sec=100.0,
        latency_p50_ms=50.0,
        latency_p95_ms=80.0,
        peak_memory_mb=200.0,
        dim=384,
        sessions=4,
        threads=2,
    )
    s = str(result)
    assert "test-model" in s
    assert "onnx" in s
    assert "100.0" in s


def test_comparison_result_summary():
    results = [
        BenchmarkResult(model_name="a.onnx", backend="onnx", throughput_docs_sec=50.0),
        BenchmarkResult(model_name="b.gguf", backend="llama.cpp", throughput_docs_sec=80.0),
    ]
    comparison = ComparisonResult(results=results)
    comparison.recommendation = results[1]
    summary = comparison.summary()
    assert "Unified Backend Comparison" in summary
    assert "b.gguf" in summary


def test_detect_hardware():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.physical_cores > 0
    assert hw.logical_cores > 0
    assert hw.cpu_name
    assert hw.ram_mb > 0


def test_cache_path():
    path = cache_path()
    assert isinstance(path, str)
    assert len(path) > 0


def test_clear_cache():
    clear_cache()


def test_benchmark_hardware_cached():
    bench = Benchmark()
    hw1 = bench.hardware
    hw2 = bench.hardware
    assert hw1 is hw2


def test_benchmark_compare_all_missing_paths():
    bench = Benchmark()
    result = bench.compare_all(onnx_path="/nonexistent/onnx", gguf_path="/nonexistent/gguf")
    assert result.results == []
    assert result.recommendation is None
