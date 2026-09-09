"""Unit tests for libembedding dataclasses."""

import numpy as np
import pytest

from libembedding.types import (
    SparseEmbedding,
    RerankResult,
    ModelInfo,
    ModelDesc,
    Stats,
    TuningResult,
    RerankerTuningResult,
    SparseTuningResult,
    ImageTuningResult,
    UnifiedTuningResult,
    ModelSelectionResult,
)


def test_sparse_embedding():
    indices = np.array([1, 5, 10], dtype=np.int32)
    values = np.array([0.5, 0.3, 0.8], dtype=np.float32)
    emb = SparseEmbedding(indices=indices, values=values)
    assert len(emb.indices) == 3
    assert len(emb.values) == 3
    assert emb.indices.dtype == np.int32
    assert emb.values.dtype == np.float32


def test_rerank_result():
    result = RerankResult(index=0, score=0.95)
    assert result.index == 0
    assert result.score == 0.95


def test_model_info():
    info = ModelInfo(
        model_name="test-model",
        model_code="test-code",
        model_file="model.onnx",
        description="Test model",
        dim=384,
        max_tokens=512,
        pooling="cls",
        quantization="none",
    )
    assert info.dim == 384
    assert info.pooling == "cls"
    assert info.quantization == "none"


def test_model_desc():
    desc = ModelDesc(
        name="test-model",
        dimension=384,
        max_length=512,
        pooling="cls",
        num_threads=4,
        batch_size=32,
        provider="cpu",
        device_id=0,
    )
    assert desc.dimension == 384
    assert desc.provider == "cpu"
    assert desc.device_id == 0


def test_stats():
    stats = Stats(texts_embedded=10, batches_run=2, avg_latency_ms=15.5)
    assert stats.texts_embedded == 10
    assert stats.batches_run == 2
    assert stats.avg_latency_ms == 15.5


def test_tuning_result():
    result = TuningResult(
        workers=4,
        threads=2,
        batch_size=32,
        throughput_docs_sec=100.0,
        latency_ms=50.0,
        memory_mb=200.0,
    )
    assert result.workers == 4
    assert result.throughput_docs_sec == 100.0


def test_reranker_tuning_result():
    result = RerankerTuningResult(
        threads=4,
        batch_size=32,
        max_tokens=512,
        throughput_docs_sec=100.0,
        latency_ms=50.0,
        p95_latency_ms=80.0,
        memory_mb=200.0,
    )
    assert result.max_tokens == 512
    assert result.p95_latency_ms == 80.0


def test_sparse_tuning_result():
    result = SparseTuningResult(
        top_k=100,
        min_weight=0.1,
        storage_format=1,
        threads=4,
        batch_size=32,
        throughput_docs_sec=80.0,
        latency_ms=40.0,
        memory_mb=150.0,
    )
    assert result.top_k == 100
    assert result.storage_format == 1


def test_image_tuning_result():
    result = ImageTuningResult(
        threads=2,
        batch_size=16,
        throughput_docs_sec=50.0,
        latency_ms=100.0,
        memory_mb=300.0,
    )
    assert result.threads == 2
    assert result.batch_size == 16


def test_unified_tuning_result():
    result = UnifiedTuningResult(
        task="embedding",
        threads=4,
        batch_size=32,
        workers=2,
        max_tokens=512,
        throughput_docs_sec=100.0,
        latency_ms=50.0,
        p95_latency_ms=80.0,
        memory_mb=200.0,
    )
    assert result.task == "embedding"
    assert result.workers == 2


def test_model_selection_result():
    result = ModelSelectionResult(
        model_code="bge-small",
        model_name="BAAI/bge-small-en-v1.5",
        dim=384,
        workers=4,
        threads=2,
        batch_size=32,
        throughput_docs_sec=100.0,
        latency_ms=50.0,
        memory_mb=200.0,
        score=0.9,
    )
    assert result.model_code == "bge-small"
    assert result.score == 0.9
