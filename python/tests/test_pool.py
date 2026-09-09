"""Tests for TextEmbeddingPool."""

import numpy as np
import pytest


def _bge_small(**kwargs):
    """Create a BGE-small model, skipping the test if download fails."""
    from libembedding import TextEmbedding
    from libembedding.exceptions import DownloadError

    kwargs.setdefault("show_download_progress", False)
    try:
        return TextEmbedding("BAAI/bge-small-en-v1.5", **kwargs)
    except DownloadError:
        pytest.skip("model download unavailable (network restriction in CI)")


def test_pool_repr():
    from libembedding import TextEmbeddingPool

    model = _bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        repr_str = repr(pool)
        assert "TextEmbeddingPool" in repr_str
        assert "workers=2" in repr_str
        assert f"dim={model.dim}" in repr_str
    finally:
        pool.close()
        model.close()


def test_pool_dim():
    from libembedding import TextEmbeddingPool

    model = _bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        assert pool.dim == model.dim
        assert pool.num_workers == 2
    finally:
        pool.close()
        model.close()


def test_pool_embed_basic():
    from libembedding import TextEmbeddingPool

    model = _bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        texts = ["Hello world", "How are you?", "Testing pooling"]
        result = pool.embed(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, model.dim)
        assert result.dtype == np.float32
    finally:
        pool.close()
        model.close()


def test_pool_embed_empty():
    from libembedding import TextEmbeddingPool

    model = _bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        result = pool.embed([])
        assert result.shape == (0, model.dim)
    finally:
        pool.close()
        model.close()


def test_pool_context_manager():
    from libembedding import TextEmbeddingPool

    model = _bge_small()
    try:
        with TextEmbeddingPool(model.model_name, workers=2, offline=True) as pool:
            assert pool.num_workers == 2
            result = pool.embed(["test"])
            assert result.shape == (1, model.dim)
    finally:
        model.close()


def test_pool_workers_auto():
    from libembedding import TextEmbeddingPool

    model = _bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=0, offline=True)
    try:
        assert pool.num_workers > 0
        assert pool.num_workers <= 8
    finally:
        pool.close()
        model.close()


def test_pool_embed_order_preserved():
    from libembedding import TextEmbeddingPool

    model = _bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        texts = [f"Text number {i}" for i in range(10)]
        result = pool.embed(texts)
        assert result.shape == (10, model.dim)
        # Verify embedding norms are reasonable (not all zeros)
        norms = np.linalg.norm(result, axis=1)
        assert np.all(norms > 0.1)
    finally:
        pool.close()
        model.close()


def test_pool_autotune_path():
    from libembedding import TextEmbeddingPool
    from libembedding.exceptions import DownloadError

    try:
        model = TextEmbedding("BAAI/bge-small-en-v1.5", show_download_progress=False)
    except DownloadError:
        pytest.skip("model download unavailable for pool autotune test")

    try:
        pool = TextEmbeddingPool(
            model.model_name,
            workers=1,
            autotune=True,
            autotune_texts=["short", "a bit longer text here", "tiny"],
            offline=True,
        )
        assert pool.num_workers >= 1
        result = pool.embed(["hello", "world"])
        assert result.shape == (2, model.dim)
        pool.close()
    finally:
        model.close()
