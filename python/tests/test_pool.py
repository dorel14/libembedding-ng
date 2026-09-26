"""Tests for TextEmbeddingPool."""

import numpy as np


def test_pool_repr(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        repr_str = repr(pool)
        assert "TextEmbeddingPool" in repr_str
        assert "workers=2" in repr_str
        assert f"dim={model.dim}" in repr_str
    finally:
        pool.close()
        model.close()


def test_pool_dim(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        assert pool.dim == model.dim
        assert pool.num_workers == 2
    finally:
        pool.close()
        model.close()


def test_pool_embed_basic(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
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


def test_pool_embed_empty(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=2, offline=True)
    try:
        result = pool.embed([])
        assert result.shape == (0, model.dim)
    finally:
        pool.close()
        model.close()


def test_pool_context_manager(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
    try:
        with TextEmbeddingPool(model.model_name, workers=2, offline=True) as pool:
            assert pool.num_workers == 2
            result = pool.embed(["test"])
            assert result.shape == (1, model.dim)
    finally:
        model.close()


def test_pool_workers_auto(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
    pool = TextEmbeddingPool(model.model_name, workers=0, offline=True)
    try:
        assert pool.num_workers > 0
        assert pool.num_workers <= 8
    finally:
        pool.close()
        model.close()


def test_pool_embed_order_preserved(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
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


def test_pool_autotune_path(bge_small):
    from libembedding import TextEmbeddingPool

    model = bge_small()
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
