"""Smoke tests for libembedding package imports and version."""


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
    import re
    from pathlib import Path

    from libembedding._binding import ffi, lib

    # The shared library must expose the C entry points used by the bindings
    for symbol in (
        "lembed_version",
        "lembed_text_embedding_create_v2",
        "lembed_text_embedding_stats_v2",
        "lembed_sparse_text_embedding_stats_v2",
        "lembed_image_embedding_stats_v2",
        "lembed_cache_get",
    ):
        assert hasattr(lib, symbol), f"Missing C symbol: {symbol}"

    # C version (config.h) and the _cdefs.h stamp must be in sync
    c_version = ffi.string(lib.lembed_version()).decode()
    assert re.fullmatch(r"\d+\.\d+\.\d+", c_version), c_version

    cdefs = Path(__file__).parent.parent / "src" / "libembedding" / "_cdefs.h"
    if cdefs.exists():
        match = re.search(r"\(v(\d+\.\d+\.\d+)\)", cdefs.read_text(encoding="utf-8"))
        assert match, "no (vX.Y.Z) marker in _cdefs.h"
        assert match.group(1) == c_version, (
            f"_cdefs.h says v{match.group(1)}, runtime says {c_version}"
        )
