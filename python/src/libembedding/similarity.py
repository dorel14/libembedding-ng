"""Native similarity functions operating on raw float arrays.

Auteur: David Orel
Version: 1.4.0
"""

from __future__ import annotations

import numpy as np

from ._binding import ffi, lib


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D float arrays of equal length."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b must be 1-D arrays of the same length")
    return lib.lembed_cosine_similarity(
        ffi.from_buffer("float[]", a), ffi.from_buffer("float[]", b), a.shape[0]
    )


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product between two 1-D float arrays of equal length."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b must be 1-D arrays of the same length")
    return lib.lembed_dot_product(
        ffi.from_buffer("float[]", a), ffi.from_buffer("float[]", b), a.shape[0]
    )


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute euclidean (L2) distance between two 1-D float arrays of equal length."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b must be 1-D arrays of the same length")
    return lib.lembed_euclidean_distance(
        ffi.from_buffer("float[]", a), ffi.from_buffer("float[]", b), a.shape[0]
    )

