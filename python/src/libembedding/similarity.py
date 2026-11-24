"""Native similarity functions operating on raw float arrays.

Auteur: David Orel
Version: 1.16.2
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._binding import ffi, lib
from ._status import check_status


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


def _as_matrix(x: np.ndarray, name: str) -> np.ndarray:
    """Return x as a C-contiguous float32 2-D array, or raise ValueError."""
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be a float32 array, got {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (rows, dim)")
    return np.ascontiguousarray(arr)


def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    """MaxSim late-interaction score (ColBERT).

    Sum over the query rows of the maximum dot product with any document row.
    ``query`` is (nq, dim) and ``doc`` is (nd, dim), both float32. L2-normalised
    rows give the cosine MaxSim. An empty query or document scores 0.0.
    """
    q = _as_matrix(query, "query")
    d = _as_matrix(doc, "doc")
    if q.shape[1] != d.shape[1]:
        raise ValueError("query and doc must have the same dim")
    if q.shape[1] <= 0:
        raise ValueError("dim must be > 0")
    out = ffi.new("float *")
    check_status(
        lib.lembed_maxsim(
            ffi.from_buffer("float[]", q) if q.size else ffi.NULL,
            q.shape[0],
            ffi.from_buffer("float[]", d) if d.size else ffi.NULL,
            d.shape[0],
            q.shape[1],
            out,
        )
    )
    return float(out[0])


def _ragged_docs(
    docs: tuple[np.ndarray, np.ndarray], dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a ragged ``(data, offsets)`` pair without copying it."""
    data_in, offsets_in = docs
    buf = _as_matrix(data_in, "docs data")
    if buf.shape[1] != dim:
        raise ValueError(f"docs data has dim {buf.shape[1]}, expected {dim}")
    offsets = np.asarray(offsets_in)
    if offsets.dtype != np.int64 or offsets.ndim != 1 or offsets.shape[0] < 1:
        raise ValueError("docs offsets must be a non-empty 1-D int64 array")
    offsets = np.ascontiguousarray(offsets)
    if offsets[0] != 0:
        raise ValueError("docs offsets[0] must be 0")
    if offsets.shape[0] > 1 and bool(np.any(np.diff(offsets) < 0)):
        raise ValueError("docs offsets must be non-decreasing")
    if offsets[-1] > buf.shape[0]:
        raise ValueError(
            f"docs offsets end at row {int(offsets[-1])}, data has {buf.shape[0]} rows"
        )
    return buf, offsets


def maxsim_batch(
    query: np.ndarray,
    docs: Sequence[np.ndarray] | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """MaxSim of one query against several documents.

    ``docs`` is either a sequence of (n_i, dim) float32 arrays (n_i may differ),
    which is concatenated once before the native call, or a ragged pair
    ``(data, offsets)`` passed to the native code without any copy: ``data`` is a
    C-contiguous float32 array ``(total_rows, dim)`` and ``offsets`` an int64
    array of ``num_docs + 1`` non-decreasing row offsets starting at 0 (document
    ``i`` is ``data[offsets[i]:offsets[i + 1]]``). ``MultiVectorEmbedding.embed_ragged``
    returns that pair. Returns a float32 array with one score per document, in
    input order.
    """
    q = _as_matrix(query, "query")
    dim = q.shape[1]
    if dim <= 0:
        raise ValueError("dim must be > 0")
    # A tuple of two documents is a list of 2-D matrices; the ragged pair has a
    # 1-D offsets array as its second item.
    if (
        isinstance(docs, tuple)
        and len(docs) == 2
        and isinstance(docs[1], np.ndarray)
        and docs[1].ndim == 1
    ):
        buf, offsets = _ragged_docs(docs, dim)
        num_docs = offsets.shape[0] - 1
    else:
        mats = []
        for i, doc in enumerate(docs):
            m = _as_matrix(doc, f"docs[{i}]")
            if m.shape[1] != dim:
                raise ValueError(f"docs[{i}] has dim {m.shape[1]}, expected {dim}")
            mats.append(m)
        num_docs = len(mats)
        if num_docs == 0:
            return np.zeros(0, dtype=np.float32)
        offsets = np.zeros(num_docs + 1, dtype=np.int64)
        np.cumsum([m.shape[0] for m in mats], out=offsets[1:])
        buf = np.concatenate(mats, axis=0)
    scores = np.zeros(num_docs, dtype=np.float32)
    if num_docs == 0:
        return scores
    check_status(
        lib.lembed_maxsim_batch(
            ffi.from_buffer("float[]", q) if q.size else ffi.NULL,
            q.shape[0],
            ffi.from_buffer("float[]", buf) if buf.size else ffi.NULL,
            ffi.from_buffer("int64_t[]", offsets),
            num_docs,
            dim,
            ffi.from_buffer("float[]", scores, require_writable=True),
        )
    )
    return scores


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute euclidean (L2) distance between two 1-D float arrays of equal length."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b must be 1-D arrays of the same length")
    return lib.lembed_euclidean_distance(
        ffi.from_buffer("float[]", a), ffi.from_buffer("float[]", b), a.shape[0]
    )
