"""Integration tests for SparseTextEmbedding.

Tests taking the ``sparse_model`` fixture require either a cached model or
network access; they are skipped when the model cannot be loaded.
"""

import numpy as np
import pytest


@pytest.fixture
def sparse_model():
    """Factory building a SparseTextEmbedding, skipping if download fails."""

    def _factory(**kwargs):
        from libembedding import SparseTextEmbedding
        from libembedding.exceptions import DownloadError

        kwargs.setdefault("show_download_progress", False)
        try:
            return SparseTextEmbedding("prithivida/Splade_PP_en_v1", **kwargs)
        except DownloadError:
            pytest.skip("sparse model download unavailable")

    return _factory


def test_sparse_list_supported_models():
    from libembedding import SparseTextEmbedding

    models = SparseTextEmbedding.list_supported_models()
    assert len(models) > 0
    for m in models:
        # Sparse models have dim = 0 (variable dimension)
        assert m.model_code
        assert m.max_tokens > 0


def test_sparse_embed_basic(sparse_model):
    model = sparse_model()
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


def test_sparse_embed_empty(sparse_model):
    model = sparse_model()
    result = model.embed([])
    assert result == []
    model.close()


def test_sparse_embed_single(sparse_model):
    model = sparse_model()
    result = model.embed(["single document"])
    assert len(result) == 1
    assert len(result[0].indices) > 0
    model.close()


def test_sparse_info(sparse_model):
    model = sparse_model()
    info = model.info()
    # Sparse models have dimension 0 (variable dimension)
    assert info.max_length > 0
    assert info.batch_size > 0
    model.close()


def test_sparse_name(sparse_model):
    model = sparse_model()
    assert "SPLADE" in model.name or "splade" in model.name.lower()
    model.close()


def test_sparse_stats(sparse_model):
    model = sparse_model()
    model.embed(["Hello world"])
    stats = model.stats()
    assert stats.texts_embedded == 1
    assert stats.batches_run >= 1
    assert stats.avg_latency_ms > 0
    model.close()


def test_sparse_context_manager(sparse_model):
    """Leaving the context must release the C context."""
    with sparse_model() as model:
        result = model.embed(["test"])
        assert len(result) == 1
        ctx = model._ctx
    assert model._ctx is None, "__exit__ must close the context"
    assert ctx is not None, "the context must be live inside the with block"


def test_sparse_batch_size_override(sparse_model):
    model = sparse_model()
    result = model.embed(["a", "b", "c"], batch_size=1)
    assert len(result) == 3
    model.close()


def test_sparse_top_terms():
    pytest.skip("top_terms feature not yet implemented in C API (P1)")


@pytest.mark.network
def test_sparse_min_weight():
    from libembedding import SparseTextEmbedding
    from libembedding.exceptions import DownloadError

    try:
        model = SparseTextEmbedding(
            "prithivida/Splade_PP_en_v1", min_weight=0.5, show_download_progress=False
        )
        result = model.embed(["Hello world"])
        assert len(result) == 1
        model.close()
    except DownloadError:
        pytest.skip("sparse model download unavailable")
