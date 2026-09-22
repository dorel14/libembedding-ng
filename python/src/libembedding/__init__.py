"""libembedding — Fast ONNX-based text, image, and sparse embeddings for Python."""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("libembedding-ng")
except PackageNotFoundError:
    __version__ = "0.0.0"

from ._binding import lib
from .autotune import (
    auto_select_model,
    autotune,
    autotune_unified,
    clear_autotune_cache,
)
from .backend import detect_backend
from .benchmark import (
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
from .cache import EmbeddingCache, cache_config_default
from .exceptions import LembedError, LlamaError
from .image_embedding import ImageEmbedding, image_autotune
from .models import (
    list_image_models,
    list_reranker_models,
    list_sparse_models,
    list_text_models,
)
from .reranker import (
    Reranker,
    clear_reranker_autotune_cache,
    reranker_auto_config,
    reranker_auto_config_profile,
    reranker_autotune,
    reranker_autotune_constrained,
)
from .similarity import cosine_similarity, dot_product, euclidean_distance
from .sparse_text_embedding import SparseTextEmbedding, sparse_autotune
from .text_embedding import (  # pyright: ignore[reportPrivateImportUsage]
    TextEmbedding,
    TextEmbeddingPool,  # pyright: ignore[reportPrivateImportUsage]
)
from .types import (
    ImageTuningResult,
    ModelDesc,
    ModelInfo,
    ModelSelectionResult,
    RerankerTuningResult,
    RerankResult,
    SparseEmbedding,
    SparseTuningResult,
    Stats,
    TuningResult,
    UnifiedTuningResult,
)

_logger = logging.getLogger(__name__)

# Version from C API (runtime)
try:
    __version__ = lib.lembed_version().decode("utf-8", errors="replace")
except (AttributeError, OSError) as e:
    _logger.debug("Could not get version from C API: %s", e)

# Autotune modes (re-exported for convenience)
LEMBED_AUTOTUNE_QUICK = lib.LEMBED_AUTOTUNE_QUICK
LEMBED_AUTOTUNE_FULL = lib.LEMBED_AUTOTUNE_FULL

# Optimization objectives
LEMBED_OBJECTIVE_LATENCY = lib.LEMBED_OBJECTIVE_LATENCY
LEMBED_OBJECTIVE_THROUGHPUT = lib.LEMBED_OBJECTIVE_THROUGHPUT
LEMBED_OBJECTIVE_BALANCED = lib.LEMBED_OBJECTIVE_BALANCED
LEMBED_OBJECTIVE_MEMORY = lib.LEMBED_OBJECTIVE_MEMORY

# Task types for unified auto-tuner
LEMBED_TASK_EMBEDDING = lib.LEMBED_TASK_EMBEDDING
LEMBED_TASK_RERANKING = lib.LEMBED_TASK_RERANKING
LEMBED_TASK_IMAGE = lib.LEMBED_TASK_IMAGE
LEMBED_TASK_SPARSE = lib.LEMBED_TASK_SPARSE

# Reranker profiles
LEMBED_PROFILE_INTERACTIVE = lib.LEMBED_PROFILE_INTERACTIVE
LEMBED_PROFILE_BALANCED = lib.LEMBED_PROFILE_BALANCED
LEMBED_PROFILE_QUALITY = lib.LEMBED_PROFILE_QUALITY

__all__ = [
    "LEMBED_AUTOTUNE_FULL",
    "LEMBED_AUTOTUNE_QUICK",
    "LEMBED_OBJECTIVE_BALANCED",
    "LEMBED_OBJECTIVE_LATENCY",
    "LEMBED_OBJECTIVE_MEMORY",
    "LEMBED_OBJECTIVE_THROUGHPUT",
    "LEMBED_PROFILE_BALANCED",
    "LEMBED_PROFILE_INTERACTIVE",
    "LEMBED_PROFILE_QUALITY",
    "LEMBED_TASK_EMBEDDING",
    "LEMBED_TASK_IMAGE",
    "LEMBED_TASK_RERANKING",
    "LEMBED_TASK_SPARSE",
    "Benchmark",
    "BenchmarkResult",
    "ComparisonResult",
    "CorpusType",
    "EmbeddingCache",
    "HardwareInfo",
    "ImageEmbedding",
    "ImageTuningResult",
    "LembedError",
    "LlamaError",
    "Metrics",
    "ModelDesc",
    "ModelInfo",
    "ModelSelectionResult",
    "Objective",
    "RerankResult",
    "Reranker",
    "RerankerTuningResult",
    "SparseEmbedding",
    "SparseTextEmbedding",
    "SparseTuningResult",
    "Stats",
    "TextEmbedding",
    "TextEmbeddingPool",
    "TuningResult",
    "UnifiedTuningResult",
    "auto_select_model",
    "autotune",
    "autotune_unified",
    "cache_config_default",
    "cache_path",
    "clear_autotune_cache",
    "clear_cache",
    "clear_reranker_autotune_cache",
    "cosine_similarity",
    "detect_backend",
    "detect_hardware",
    "dot_product",
    "euclidean_distance",
    "image_autotune",
    "list_image_models",
    "list_reranker_models",
    "list_sparse_models",
    "list_text_models",
    "reranker_auto_config",
    "reranker_auto_config_profile",
    "reranker_autotune",
    "reranker_autotune_constrained",
    "sparse_autotune",
]
