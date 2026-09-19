"""LRU cache for embeddings.

Provides both a standalone cache class and integration with TextEmbedding
via the ``cache_size`` constructor parameter.

Auteur: David Orel
Version: 1.6.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import numpy as np

from ._binding import ffi, lib
from .models import _QUANTIZATION_NAMES


class EmbeddingCache:
    """Thread-safe LRU cache for dense embeddings.

    Can be used standalone or is automatically created when
    ``cache_size`` is set on a :class:`TextEmbedding` or :class:`Reranker`
    instance.

    Args:
        capacity: Maximum number of entries (default 4096).
        ttl_seconds: Optional time-to-live in seconds (0 = no expiry).
        dim: Expected embedding dimensionality (for validation).

    Example:
        >>> cache = EmbeddingCache(capacity=1024)
        >>> cache.put("hello world", np.array([0.1, 0.2, 0.3], dtype=np.float32))
        >>> vec = cache.get("hello world")
        >>> print(vec.shape)
        (3,)
    """

    def __init__(self, capacity: int = 4096, ttl_seconds: int = 0, dim: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._dim = dim
        cfg = ffi.new("lembed_cache_config_t *")
        cfg.capacity = capacity
        cfg.ttl_seconds = ttl_seconds
        self._cfg = cfg
        self._cache = ffi.gc(
            lib.lembed_cache_create(ffi.addressof(cfg)),
            lib.lembed_cache_free,
        )

    @property
    def capacity(self) -> int:
        """Maximum number of entries in the cache."""
        return lib.lembed_cache_capacity(self._cache)

    @property
    def current_size(self) -> int:
        """Current number of entries in the cache."""
        return lib.lembed_cache_size(self._cache)

    def get(self, text: str, dim: int | None = None) -> np.ndarray | None:
        """Look up a cached embedding by text.

        Args:
            text: The text string to look up.
            dim: Expected dimensionality (for validation; overrides constructor dim).

        Returns:
            Numpy float32 array of shape (dim,) if found, else None.
        """
        expected_dim = dim if dim is not None else self._dim
        out_vec = ffi.new("float **")
        out_dim = ffi.new("int *")
        c_text = text.encode("utf-8")
        c_buf = ffi.new("char[]", c_text)

        hit = lib.lembed_cache_get(self._cache, c_buf, out_vec, out_dim)
        if not hit:
            return None

        cached_dim = out_dim[0]
        if expected_dim > 0 and cached_dim != expected_dim:
            lib.free(out_vec[0])
            return None

        arr = np.frombuffer(
            ffi.buffer(out_vec[0], cached_dim * 4), dtype=np.float32
        ).copy()
        lib.free(out_vec[0])
        return arr

    def put(self, text: str, vec: np.ndarray) -> None:
        """Store an embedding in the cache.

        Args:
            text: The text string key.
            vec: Numpy float32 array (1-D).
        """
        vec = np.ascontiguousarray(vec, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("vec must be a 1-D array")

        c_text = text.encode("utf-8")
        c_buf = ffi.new("char[]", c_text)
        c_vec = ffi.from_buffer("float[]", vec)

        lib.lembed_cache_put(self._cache, c_buf, c_vec, vec.shape[0])

    def clear(self) -> None:
        """Remove all entries from the cache."""
        lib.lembed_cache_clear(self._cache)

    def __len__(self) -> int:
        return self.current_size

    def __contains__(self, text: str) -> bool:
        """Return True if text is in the cache."""
        return self.get(text) is not None

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "capacity": int(self.capacity),
            "current_size": int(self.current_size),
        }

    def close(self) -> None:
        """Release the underlying C resources."""
        self._cache = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def cache_config_default() -> dict:
    """Return the default cache configuration."""
    cfg = lib.lembed_cache_config_default()
    return {
        "capacity": int(cfg.capacity),
        "ttl_seconds": int(cfg.ttl_seconds),
    }
