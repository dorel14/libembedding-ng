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
