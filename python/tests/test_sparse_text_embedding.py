"""Integration tests for SparseTextEmbedding.

These tests require either a cached model or network access.
They are skipped if the model cannot be loaded.
"""

import numpy as np
import pytest


def _sparse_model(**kwargs):
    """Create a sparse embedding model, skipping if download fails."""
    from libembedding import SparseTextEmbedding
    from libembedding.exceptions import DownloadError

    kwargs.setdefault("show_download_progress", False)
    try:
        return SparseTextEmbedding("prithvida/SPLADE_PP_en_v1", **kwargs)
    except DownloadError:
        pytest.skip("sparse model download unavailable")


def test_sparse_list_supported_models():
    from libembedding import SparseTextEmbedding

    models = SparseTextEmbedding.list_supported_models()
    assert len(models) > 0
    for m in models:
        assert m.dim > 0
        assert m.model_code


def test_sparse_embed_basic():
    model = _sparse_model()
    result = model.embed(["Hello world", "How are you?"])
    assert isinstance(result, list)
    assert len(result) == 2
    for emb in result:
        assert isinstance(emb.indices, np.ndarray)
        assert isinstance(emb.values, np.ndarray)
        assert emb.indices.dtype == np.int32
        assert emb.values.dtype == np.float32
        assert len(emb.indices) == len(emb.values)
    model.close()


def test_sparse_embed_empty():
    model = _sparse_model()
    result = model.embed([])
    assert result == []
    model.close()


def test_sparse_embed_single():
    model = _sparse_model()
    result = model.embed(["single document"])
    assert len(result) == 1
    assert len(result[0].indices) > 0
    model.close()


def test_sparse_info():
    model = _sparse_model()
    info = model.info()
    assert info.dimension > 0
    assert info.max_length > 0
    assert info.batch_size > 0
    model.close()


def test_sparse_name():
    model = _sparse_model()
    assert "SPLADE" in model.name or "splade" in model.name.lower()
    model.close()


def test_sparse_stats():
    model = _sparse_model()
    model.embed(["Hello world"])
    stats = model.stats()
    assert stats.texts_embedded == 1
    assert stats.batches_run >= 1
    assert stats.avg_latency_ms > 0
    model.close()


def test_sparse_context_manager():
    try:
        with _sparse_model() as model:
            result = model.embed(["test"])
            assert len(result) == 1
    except pytest.skip.Exception:
        pass


def test_sparse_batch_size_override():
    model = _sparse_model()
    result = model.embed(["a", "b", "c"], batch_size=1)
    assert len(result) == 3
    model.close()


def test_sparse_top_terms():
    from libembedding import SparseTextEmbedding
    from libembedding.exceptions import DownloadError

    try:
        model = SparseTextEmbedding("prithvida/SPLADE_PP_en_v1", top_terms=10, show_download_progress=False)
        result = model.embed(["Hello world"])
        assert len(result[0].indices) <= 10
        model.close()
    except DownloadError:
        pytest.skip("sparse model download unavailable")


def test_sparse_min_weight():
    from libembedding import SparseTextEmbedding
    from libembedding.exceptions import DownloadError

    try:
        model = SparseTextEmbedding("prithvida/SPLADE_PP_en_v1", min_weight=0.5, show_download_progress=False)
        result = model.embed(["Hello world"])
        assert len(result) == 1
        model.close()
    except DownloadError:
        pytest.skip("sparse model download unavailable")
