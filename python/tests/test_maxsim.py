"""Tests for maxsim / maxsim_batch (no model, no network)."""

from __future__ import annotations

import numpy as np
import pytest

import libembedding
from libembedding import maxsim, maxsim_batch

Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
D = np.array([[0.5, 0.5], [0.9, 0.1], [-1.0, 2.0]], dtype=np.float32)


def test_exported() -> None:
    assert "maxsim" in libembedding.__all__
    assert "maxsim_batch" in libembedding.__all__


def test_maxsim_hand_computed() -> None:
    assert maxsim(Q, D) == pytest.approx(2.9, abs=1e-5)


def test_maxsim_empty() -> None:
    empty = np.zeros((0, 2), dtype=np.float32)
    assert maxsim(empty, D) == 0.0
    assert maxsim(Q, empty) == 0.0


def test_maxsim_non_contiguous() -> None:
    wide = np.asfortranarray(D)
    assert maxsim(Q, wide) == pytest.approx(2.9, abs=1e-5)


def test_maxsim_batch_matches_loop() -> None:
    docs = [D, np.zeros((0, 2), dtype=np.float32), np.array([[0.0, 3.0]], dtype=np.float32)]
    scores = maxsim_batch(Q, docs)
    assert scores.dtype == np.float32
    assert scores.shape == (3,)
    for s, d in zip(scores, docs):
        assert s == pytest.approx(maxsim(Q, d), abs=1e-6)
    assert scores[1] == 0.0


def test_maxsim_batch_no_docs() -> None:
    assert maxsim_batch(Q, []).shape == (0,)


def test_maxsim_batch_all_empty_docs() -> None:
    empty = np.zeros((0, 2), dtype=np.float32)
    assert maxsim_batch(Q, [empty, empty]).tolist() == [0.0, 0.0]


def _ragged(docs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(docs) + 1, dtype=np.int64)
    np.cumsum([len(d) for d in docs], out=offsets[1:])
    return np.concatenate(docs), offsets


def test_maxsim_batch_ragged_matches_list() -> None:
    rng = np.random.default_rng(3)
    q = rng.standard_normal((5, 17)).astype(np.float32)
    docs = [rng.standard_normal((n, 17)).astype(np.float32) for n in (4, 0, 33, 1, 70)]
    data, offsets = _ragged(docs)
    from_list = maxsim_batch(q, docs)
    from_ragged = maxsim_batch(q, (data, offsets))
    assert from_ragged.dtype == np.float32
    assert from_ragged.tolist() == from_list.tolist()
    for s, d in zip(from_ragged, docs):
        assert s == pytest.approx(maxsim(q, d), abs=1e-6)
        # float64 numpy reference
        ref = 0.0 if len(d) == 0 else float((q.astype(np.float64) @ d.astype(np.float64).T).max(1).sum())
        assert s == pytest.approx(ref, rel=1e-5, abs=1e-5)


def test_maxsim_batch_ragged_ignores_rows_past_the_end() -> None:
    data, offsets = _ragged([D, D])
    padded = np.concatenate([data, np.full((3, 2), 9.0, dtype=np.float32)])
    assert maxsim_batch(Q, (padded, offsets)).tolist() == maxsim_batch(Q, (data, offsets)).tolist()


def test_maxsim_batch_two_docs_tuple_is_a_list() -> None:
    # a tuple of two 2-D matrices is two documents, not a ragged pair
    assert maxsim_batch(Q, (D, D)).shape == (2,)


def test_maxsim_batch_ragged_validation() -> None:
    data, offsets = _ragged([D, D])
    with pytest.raises(ValueError):
        maxsim_batch(Q, (data, offsets.astype(np.int32)))
    with pytest.raises(ValueError):
        maxsim_batch(Q, (data, np.array([1, 3, 6], dtype=np.int64)))
    with pytest.raises(ValueError):
        maxsim_batch(Q, (data, np.array([0, 4, 2], dtype=np.int64)))
    with pytest.raises(ValueError):
        maxsim_batch(Q, (data, np.array([0, 3, 7], dtype=np.int64)))  # past the data
    with pytest.raises(ValueError):
        maxsim_batch(Q, (np.zeros((6, 3), dtype=np.float32), offsets))
    empty = (np.zeros((0, 2), dtype=np.float32), np.zeros(1, dtype=np.int64))
    assert maxsim_batch(Q, empty).shape == (0,)


def test_validation() -> None:
    with pytest.raises(ValueError):
        maxsim(Q.astype(np.float64), D)
    with pytest.raises(ValueError):
        maxsim(Q[0], D)
    with pytest.raises(ValueError):
        maxsim(Q, np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        maxsim_batch(Q, [D, np.zeros((2, 5), dtype=np.float32)])
    with pytest.raises(ValueError):
        maxsim_batch(Q, [D.astype(np.float64)])
