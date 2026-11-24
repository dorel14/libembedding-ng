"""Data types for libembedding results.

Auteur: David Orel
Version: 1.16.3
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
    # Download size in MB (model file plus external weights), -1 if unknown.
    download_size_mb: int = -1


@dataclass(frozen=True)
class ModelDesc:
    """Runtime descriptor of a created model context."""

    name: str
    dimension: int
    max_length: int
    pooling: str  # "cls" or "mean"
    num_threads: int
    batch_size: int
    provider: str = "cpu"
    device_id: int = 0
    quantization: str = "none"
    cache_size: int = 0
    # Runtime actually serving the model: "onnxruntime", "llama.cpp" or
    # "ggml-sparse" (the in-house SPLADE runtime). "" when not reported.
    backend: str = ""
    # Execution provider actually in use, lowercase: "cpu", "cuda", "coreml",
    # "directml", "tensorrt", or "gpu" for llama.cpp offload. Unlike
    # `provider` (the requested value, "llama" for every GGUF context), this
    # names the device. "" when not reported.
    effective_provider: str = ""


@dataclass(frozen=True)
class Stats:
    """Runtime statistics for an embedding context."""

    texts_embedded: int
    batches_run: int
    avg_latency_ms: float
    cache_hits: int = 0
    cache_misses: int = 0


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
    # GGUF runtime only (0 for ONNX models). workers: pass as ``workers=``
    # (0 = automatic); max_batch_tokens: apply with ``set_max_batch_tokens()``.
    workers: int = 0
    max_batch_tokens: int = 0
    # True when the result was read from the tune cache (no benchmark ran).
    from_cache: bool = False


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
    """Result of unified auto-tuning for any task type.

    ``workers`` is set for "embedding", ``max_tokens`` for "reranking", and
    ``top_k`` / ``min_weight`` / ``storage_format`` for "sparse" (the defaults a
    sparse model uses: they shape the output, so they are reported, not tuned).
    Latencies are per tuning call, except "embedding" where they are amortised
    per text.
    """

    task: str
    threads: int
    batch_size: int
    workers: int
    max_tokens: int
    throughput_docs_sec: float
    latency_ms: float
    p95_latency_ms: float
    memory_mb: float
    top_k: int = 0
    min_weight: float = 0.0
    storage_format: int = 0


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
