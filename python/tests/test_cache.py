"""Tests for the LRU embedding cache (EmbeddingCache).

Ownership contract under test
-----------------------------
``lembed_cache_get()`` returns a BORROWED pointer owned by the C cache: it is
freed by the cache on overwrite, on LRU eviction and on clear(). The Python
wrapper copies the data and never frees it.

The tests below therefore guard the double-free scenario: a value read from the
cache must stay usable after the entry has been evicted, and reading the same
key twice must not corrupt the heap.
"""

import numpy as np
import pytest

from libembedding import EmbeddingCache, cache_config_default


def _vec(seed: int, dim: int = 4) -> np.ndarray:
    return np.arange(seed, seed + dim, dtype=np.float32)


def test_cache_config_default():
    cfg = cache_config_default()
    assert cfg["capacity"] > 0
    assert cfg["ttl_seconds"] >= 0


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        EmbeddingCache(capacity=0)
    with pytest.raises(ValueError):
        EmbeddingCache(capacity=-1)


def test_miss_returns_none():
    cache = EmbeddingCache(capacity=8)
    assert cache.get("absent") is None
    assert cache.current_size == 0
    assert len(cache) == 0


def test_put_then_hit():
    cache = EmbeddingCache(capacity=8)
    expected = _vec(1)
    cache.put("a", expected)
    assert cache.current_size == 1

    got = cache.get("a")
    assert got is not None
    assert got.dtype == np.float32
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)
    assert "a" in cache


def test_put_2d_raises():
    cache = EmbeddingCache(capacity=8)
    with pytest.raises(ValueError):
        cache.put("a", np.zeros((2, 4), dtype=np.float32))


def test_get_twice_same_key_is_stable():
    """Re-reading a key must return the same data, not freed memory."""
    cache = EmbeddingCache(capacity=8)
    cache.put("a", _vec(1))

    first = cache.get("a")
    second = cache.get("a")

    assert first is not None and second is not None
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, _vec(1))
    # The two reads are independent copies
    assert first.ctypes.data != second.ctypes.data


def test_returned_array_is_a_private_copy():
    """The array returned by get() must survive eviction of its entry."""
    cache = EmbeddingCache(capacity=2)
    cache.put("a", _vec(10))
    snapshot = cache.get("a")
    assert snapshot is not None

    # Overflow the cache: "a" gets evicted and its buffer freed by the cache
    cache.put("b", _vec(20))
    cache.put("c", _vec(30))
    assert cache.get("a") is None
    assert cache.current_size == 2

    # The previously returned array is a copy: still valid, still correct
    np.testing.assert_array_equal(snapshot, _vec(10))
    snapshot[0] = -1.0
    assert cache.get("b") is not None


def test_lru_eviction_order():
    """Least recently used entry is evicted first."""
    cache = EmbeddingCache(capacity=2)
    cache.put("a", _vec(1))
    cache.put("b", _vec(2))
    # Touch "a" so "b" becomes the least recently used entry
    assert cache.get("a") is not None
    cache.put("c", _vec(3))

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None
    assert cache.current_size == 2


def test_overwrite_same_key_keeps_size():
    cache = EmbeddingCache(capacity=4)
    cache.put("a", _vec(1))
    cache.put("a", _vec(100))
    assert cache.current_size == 1
    got = cache.get("a")
    assert got is not None
    np.testing.assert_array_equal(got, _vec(100))


def test_clear_releases_all_entries():
    cache = EmbeddingCache(capacity=4)
    cache.put("a", _vec(1))
    cache.put("b", _vec(2))
    assert cache.current_size == 2

    cache.clear()
    assert cache.current_size == 0
    assert cache.get("a") is None
    assert cache.get("b") is None

    # The cache is still usable after clear()
    cache.put("c", _vec(3))
    assert cache.get("c") is not None


def test_dim_mismatch_returns_none():
    cache = EmbeddingCache(capacity=4, dim=4)
    cache.put("a", _vec(1, dim=8))
    assert cache.get("a") is None
    # Explicit dim argument overrides the constructor dim
    assert cache.get("a", dim=8) is not None


def test_empty_vector_roundtrip():
    cache = EmbeddingCache(capacity=4)
    cache.put("empty", np.zeros(0, dtype=np.float32))
    got = cache.get("empty")
    assert got is not None
    assert got.shape == (0,)


def test_unicode_key():
    cache = EmbeddingCache(capacity=4)
    key = "héllo wörld — ünïcode ✓"
    cache.put(key, _vec(7))
    got = cache.get(key)
    assert got is not None
    np.testing.assert_array_equal(got, _vec(7))


def test_stats_and_capacity():
    cache = EmbeddingCache(capacity=16)
    stats = cache.stats()
    assert stats["capacity"] == 16
    assert stats["current_size"] == 0
    cache.put("a", _vec(1))
    assert cache.stats()["current_size"] == 1
    assert cache.capacity == 16


def test_context_manager():
    with EmbeddingCache(capacity=4) as cache:
        cache.put("a", _vec(1))
        assert cache.get("a") is not None
    # close() drops the ffi.gc handle; the C cache is freed by the gc finalizer
    assert cache._cache is None


def test_many_entries_with_small_capacity():
    """Hammer the cache to exercise eviction against the allocator."""
    cache = EmbeddingCache(capacity=4)
    for i in range(200):
        key = f"text-{i}"
        cache.put(key, _vec(i))
        got = cache.get(key)
        assert got is not None
        np.testing.assert_array_equal(got, _vec(i))
    assert cache.current_size == 4

    # The last 4 keys must still be present, the older ones evicted
    for i in range(196, 200):
        assert cache.get(f"text-{i}") is not None
    for i in range(0, 100):
        assert cache.get(f"text-{i}") is None


def test_text_embedding_cache_hits(bge_small):
    """TextEmbedding(cache_size=N) must serve repeated texts from the cache."""
    model = bge_small(cache_size=16)
    try:
        assert model.info().cache_size == 16

        texts = ["cache me", "cache me too"]
        first = model.embed(texts)
        stats = model.stats()
        assert stats.cache_misses == 2
        assert stats.cache_hits == 0

        second = model.embed(texts)
        stats = model.stats()
        assert stats.cache_hits == 2
        assert stats.cache_misses == 2

        # Cached values must be identical to the freshly computed ones
        np.testing.assert_allclose(first, second, atol=1e-6)
    finally:
        model.close()


def test_text_embedding_partial_cache_hits(bge_small):
    """A batch mixing cached and new texts must only embed the new ones."""
    model = bge_small(cache_size=16)
    try:
        model.embed(["alpha", "beta"])
        assert model.stats().cache_misses == 2

        result = model.embed(["beta", "gamma", "alpha"])
        stats = model.stats()
        assert stats.cache_hits == 2
        assert stats.cache_misses == 3
        assert result.shape == (3, model.dim)
    finally:
        model.close()
