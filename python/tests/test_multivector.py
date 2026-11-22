"""Tests for MultiVectorEmbedding (ColBERT late interaction).

Offline unit tests need no model. Model tests use the local cache only (they
skip when the answerai-colbert-small-v1 files are absent) and never download.

Auteur: David Orel
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import libembedding
from libembedding import (
    LateInteractionTextEmbedding,
    LembedError,
    MultiVectorEmbedding,
    maxsim,
)

MODEL_DIR = (
    Path.home() / ".cache" / "libembedding" / "models--answerdotai-answerai-colbert-small-v1"
)
GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests" / "data" / "colbert" / "answerai_small_golden.json"
)
_FILES = ("vespa_colbert.onnx", "tokenizer.json", "onnx_config.json")
_have_model = all((MODEL_DIR / f).is_file() for f in _FILES) and GOLDEN.is_file()
needs_model = pytest.mark.skipif(
    not _have_model, reason="answerai-colbert-small-v1 not in the local cache"
)


# ---------------------------------------------------------------- offline ----


def test_alias_identity() -> None:
    assert LateInteractionTextEmbedding is MultiVectorEmbedding
    assert "MultiVectorEmbedding" in libembedding.__all__
    assert "LateInteractionTextEmbedding" in libembedding.__all__


def test_missing_dir_raises_lembed_error(tmp_path: Path) -> None:
    with pytest.raises(LembedError):
        MultiVectorEmbedding(str(tmp_path / "does-not-exist"))


def test_empty_dir_raises_lembed_error(tmp_path: Path) -> None:
    with pytest.raises(LembedError):
        MultiVectorEmbedding(str(tmp_path))


def test_gguf_path_refused(tmp_path: Path) -> None:
    with pytest.raises(LembedError):
        MultiVectorEmbedding(str(tmp_path / "m.gguf"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_dir": ""},
        {"model_dir": "x", "provider": "nope"},
        {"model_dir": "x", "query_maxlen": 0},
        {"model_dir": "x", "doc_maxlen": -3},
        {"model_dir": "x", "threads": -1},
    ],
)
def test_bad_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MultiVectorEmbedding(**kwargs)


def test_non_path_model_dir() -> None:
    with pytest.raises(TypeError):
        MultiVectorEmbedding(123)  # type: ignore[arg-type]


# ------------------------------------------------------------------ model ----


@pytest.fixture(scope="module")
def mv():
    if not _have_model:
        pytest.skip("answerai-colbert-small-v1 not in the local cache")
    with MultiVectorEmbedding(str(MODEL_DIR)) as m:
        yield m


@needs_model
def test_close_idempotent_and_closed_use() -> None:
    m = MultiVectorEmbedding(str(MODEL_DIR))
    m.close()
    m.close()
    with pytest.raises(ValueError):
        m.embed(["x"])
    with pytest.raises(ValueError):
        _ = m.dim


@needs_model
def test_bad_embed_arguments(mv: MultiVectorEmbedding) -> None:
    with pytest.raises(TypeError):
        mv.embed("a single string")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mv.embed(["x"], mode="bogus")
    assert mv.embed([]) == []


@needs_model
def test_shapes(mv: MultiVectorEmbedding) -> None:
    assert mv.dim == 96
    q = mv.embed(["what is the capital of France?"], mode="query")
    assert q[0].shape == (32, 96)
    assert q[0].dtype == np.float32
    docs = mv.embed(["Paris is the capital of France.", "Bananas grow in tropical climates."])
    assert len(docs) == 2
    assert all(d.ndim == 2 and d.shape[1] == 96 and d.shape[0] > 0 for d in docs)
    assert docs[0].shape[0] != 32 or docs[1].shape[0] != 32


@needs_model
def test_norms(mv: MultiVectorEmbedding) -> None:
    for m in mv.embed(["Paris is the capital of France."]) + mv.embed(["a query"], mode="query"):
        np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-3)


@needs_model
def test_iterators_match_embed(mv: MultiVectorEmbedding) -> None:
    q = list(mv.query_embed("how do bananas grow"))
    assert len(q) == 1 and q[0].shape == (32, 96)
    np.testing.assert_array_equal(q[0], mv.embed(["how do bananas grow"], mode="query")[0])
    two = list(mv.query_embed(["a", "b"]))
    assert len(two) == 2
    p = list(mv.passage_embed(["Hamlet is a tragedy."]))
    np.testing.assert_array_equal(p[0], mv.embed(["Hamlet is a tragedy."])[0])


@needs_model
def test_maxsim_relevance(mv: MultiVectorEmbedding) -> None:
    q = mv.embed(["what is the capital of France?"], mode="query")[0]
    rel, unrel = mv.embed(
        ["Paris is the capital of France.", "Bananas grow in tropical climates."]
    )
    assert maxsim(q, rel) > maxsim(q, unrel)


@needs_model
def test_info_stats(mv: MultiVectorEmbedding) -> None:
    mv.embed(["x"])
    assert mv.backend == "onnxruntime"
    assert mv.info().dimension == 96
    assert mv.stats().texts_embedded >= 1


@needs_model
def test_golden_parity(mv: MultiVectorEmbedding) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for mode, key in (("query", "queries"), ("document", "docs")):
        items = golden[key]
        got = mv.embed([it["text"] for it in items], mode=mode)
        for it, vecs in zip(items, got, strict=True):
            ref = np.asarray(it["vectors"], dtype=np.float32)
            assert vecs.shape == ref.shape, (mode, it["text"])
            cos = np.sum(vecs * ref, axis=1) / (
                np.linalg.norm(vecs, axis=1) * np.linalg.norm(ref, axis=1)
            )
            assert cos.min() >= 0.999, (mode, it["text"], float(cos.min()))
