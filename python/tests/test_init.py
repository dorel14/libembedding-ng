"""Smoke tests for libembedding package imports and version."""

import pytest


def test_package_import():
    import libembedding

    assert libembedding.__version__


def test_public_api_exports():
    import libembedding

    expected = [
        "TextEmbedding",
        "TextEmbeddingPool",
        "SparseTextEmbedding",
        "ImageEmbedding",
        "Reranker",
        "Benchmark",
        "BenchmarkResult",
        "ComparisonResult",
        "CorpusType",
        "HardwareInfo",
        "Metrics",
        "Objective",
        "ModelDesc",
        "ModelInfo",
        "ModelSelectionResult",
        "RerankResult",
        "RerankerTuningResult",
        "SparseEmbedding",
        "SparseTuningResult",
        "Stats",
        "TuningResult",
        "UnifiedTuningResult",
        "LembedError",
        "LlamaError",
        "cosine_similarity",
        "dot_product",
        "euclidean_distance",
        "detect_backend",
        "list_text_models",
        "list_sparse_models",
        "list_image_models",
        "list_reranker_models",
        "autotune",
        "auto_select_model",
        "autotune_unified",
        "clear_autotune_cache",
        "clear_cache",
        "clear_reranker_autotune_cache",
        "reranker_auto_config",
        "reranker_auto_config_profile",
        "reranker_autotune",
        "reranker_autotune_constrained",
        "sparse_autotune",
        "image_autotune",
        "cache_path",
    ]
    for name in expected:
        assert hasattr(libembedding, name), f"Missing export: {name}"


def test_constants_exports():
    import libembedding

    assert hasattr(libembedding, "LEMBED_AUTOTUNE_QUICK")
    assert hasattr(libembedding, "LEMBED_AUTOTUNE_FULL")
    assert hasattr(libembedding, "LEMBED_OBJECTIVE_LATENCY")
    assert hasattr(libembedding, "LEMBED_OBJECTIVE_THROUGHPUT")
    assert hasattr(libembedding, "LEMBED_OBJECTIVE_BALANCED")
    assert hasattr(libembedding, "LEMBED_OBJECTIVE_MEMORY")
    assert hasattr(libembedding, "LEMBED_TASK_EMBEDDING")
    assert hasattr(libembedding, "LEMBED_TASK_RERANKING")
    assert hasattr(libembedding, "LEMBED_TASK_IMAGE")
    assert hasattr(libembedding, "LEMBED_TASK_SPARSE")
    assert hasattr(libembedding, "LEMBED_PROFILE_INTERACTIVE")
    assert hasattr(libembedding, "LEMBED_PROFILE_BALANCED")
    assert hasattr(libembedding, "LEMBED_PROFILE_QUALITY")


def test_binding_ffi_lib():
    from libembedding._binding import ffi, lib

    assert ffi is not None
    assert lib is not None
    # Basic C API sanity check
    version = lib.lembed_version()
    assert version
    assert len(ffi.string(version).decode()) > 0
