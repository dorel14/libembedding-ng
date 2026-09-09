"""Data types for libembedding results.

Auteur: David Orel
Version: 1.4.0
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseEmbedding:
    """Sparse embedding with token indices and weights."""

    indices: np.ndarray  # int32
    values: np.ndarray  # float32


@dataclass(frozen=True)
class RerankResult:
    """Single document reranking result."""

    index: int
    score: float


@dataclass(frozen=True)
class ModelInfo:
    """Information about an available model."""

    model_name: str
    model_code: str
    model_file: str
    description: str
    dim: int
    max_tokens: int
    pooling: str  # "cls" or "mean"
    quantization: str  # "none", "static", "dynamic"


@dataclass(frozen=True)
class ModelDesc:
    """Runtime descriptor of a created model context."""

    name: str
    dimension: int
    max_length: int
    pooling: str  # "cls" or "mean"
    num_threads: int
    batch_size: int
    provider: str
    device_id: int


@dataclass(frozen=True)
class Stats:
    """Runtime statistics for an embedding context."""

    texts_embedded: int
    batches_run: int
    avg_latency_ms: float


@dataclass(frozen=True)
class TuningResult:
    """Result of auto-tuning for optimal performance."""

    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float


@dataclass(frozen=True)
class RerankerTuningResult:
    """Result of reranker auto-tuning for optimal performance."""

    threads: int
    batch_size: int
    max_tokens: int
    throughput_docs_sec: float
    latency_ms: float
    p95_latency_ms: float
    memory_mb: float


@dataclass(frozen=True)
class SparseTuningResult:
    """Result of sparse auto-tuning for optimal performance."""

    top_k: int
    min_weight: float
    storage_format: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float


@dataclass(frozen=True)
class ImageTuningResult:
    """Result of image auto-tuning for optimal performance."""

    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float


@dataclass(frozen=True)
class UnifiedTuningResult:
    """Result of unified auto-tuning for any task type."""

    task: str
    threads: int
    batch_size: int
    workers: int
    max_tokens: int
    throughput_docs_sec: float
    latency_ms: float
    p95_latency_ms: float
    memory_mb: float


@dataclass(frozen=True)
class ModelSelectionResult:
    """Result of automatic model selection."""

    model_code: str
    model_name: str
    dim: int
    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
    score: float

