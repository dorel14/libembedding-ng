"""Tests for TextEmbedding."""

import warnings

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


def test_list_text_models():
    from libembedding import list_text_models

    models = list_text_models()
    assert len(models) > 0
    assert any("bge-small" in m.model_name for m in models)
    for m in models:
        assert m.dim > 0
        assert m.model_code


def test_text_embedding_basic():
    model = _bge_small()
    assert model.dim == 384
    result = model.embed(["Hello world", "How are you?"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 384)
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    model.close()


def test_text_embedding_empty():
    model = _bge_small()
    result = model.embed([])
    assert result.shape == (0, model.dim)
    model.close()


def test_text_embedding_cosine_similarity():
    model = _bge_small()
    result = model.embed([
        "The cat sat on the mat",
        "A kitten was sitting on a rug",
        "Quantum physics is complex",
    ])
    sim_similar = np.dot(result[0], result[1])
    sim_different = np.dot(result[0], result[2])
    assert sim_similar > sim_different
    model.close()


def test_text_embedding_threads_param():
    model = _bge_small(threads=2)
    info = model.info()
    assert info.num_threads == 2
    model.close()


def test_text_embedding_num_threads_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _bge_small(num_threads=2)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "num_threads" in str(w[0].message)


def test_text_embedding_batch_size():
    model = _bge_small(batch_size=64)
    assert model.batch_size == 64
    info = model.info()
    assert info.batch_size == 64
    result = model.embed(["Hello", "World"], batch_size=1)
    assert result.shape == (2, 384)
    result2 = model.embed(["Hello", "World"])
    assert result2.shape == (2, 384)
    model.close()


def test_text_embedding_info():
    model = _bge_small()
    info = model.info()
    assert info.dimension == 384
    assert info.dimension > 0
    assert info.max_length > 0
    assert info.batch_size > 0
    model.close()


def test_text_embedding_name():
    model = _bge_small()
    assert "bge-small" in model.name
    model.close()


def test_text_embedding_offline_param():
    model = _bge_small(offline=False)
    assert model.dim == 384
    info = model.info()
    assert info.batch_size == 256
    model.close()


def test_text_embedding_offline_missing_model():
    from libembedding import TextEmbedding
    from libembedding.exceptions import LembedError

    with pytest.raises(LembedError):
        TextEmbedding("BAAI/bge-large-en-v1.5", offline=True)


def test_text_embedding_stats():
    model = _bge_small()
    model.embed(["Hello world", "Test sentence"])
    stats = model.stats()
    assert stats.texts_embedded == 2
    assert stats.batches_run >= 1
    assert stats.avg_latency_ms > 0
    model.close()


def test_text_embedding_max_length():
    model = _bge_small()
    assert model.info().max_length > 0
    model.close()


def test_similarity_functions():
    from libembedding import cosine_similarity, dot_product, euclidean_distance

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert abs(cosine_similarity(a, b) - 1.0) < 1e-5
    assert abs(dot_product(a, b) - 14.0) < 1e-5
    assert abs(euclidean_distance(a, b) - 0.0) < 1e-5

    c = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(c, d) - 0.0) < 1e-5
    assert abs(dot_product(c, d) - 0.0) < 1e-5
    assert abs(euclidean_distance(c, d) - 1.41421) < 1e-4


def test_text_embedding_from_mode():
    model = TextEmbedding.from_mode("balanced")
    assert model.dim == 384
    result = model.embed(["Hello world"])
    assert result.shape == (1, 384)
    model.close()


def test_text_embedding_from_mode_fast():
    model = TextEmbedding.from_mode("fast")
    assert model.dim > 0
    model.close()


def test_text_embedding_from_mode_invalid():
    with pytest.raises(ValueError, match="Unknown mode"):
        TextEmbedding.from_mode("invalid_mode")


def test_text_embedding_repr():
    model = _bge_small()
    r = repr(model)
    assert "TextEmbedding" in r
    assert f"dim={model.dim}" in r
    model.close()


def test_text_embedding_context_manager():
    with _bge_small() as model:
        assert model.dim == 384
        result = model.embed(["test"])
        assert result.shape == (1, 384)


def test_text_embedding_stream():
    model = _bge_small()
    texts = ["Hello", "World", "Test"]
    received = []

    def callback(arr, dim, userdata):
        received.append(arr)

    model.embed_stream(texts, callback)
    assert len(received) == 3
    for arr in received:
        assert arr.shape == (model.dim,)
    model.close()


def test_text_embedding_batched():
    model = _bge_small()
    texts = ["a", "b", "c", "d", "e"]
    batches = list(model.embed_batched(texts, batch_size=2))
    assert len(batches) == 5
    for emb in batches:
        assert emb.shape == (model.dim,)
    model.close()

