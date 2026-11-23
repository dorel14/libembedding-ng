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
from libembedding.exceptions import DownloadError

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


REGISTRY_ID = "answerdotai/answerai-colbert-small-v1"


def test_list_supported_models() -> None:
    models = MultiVectorEmbedding.list_supported_models()
    assert [m.model_code for m in models] == [REGISTRY_ID]
    m = models[0]
    assert m.model_name == REGISTRY_ID
    assert m.model_file == "vespa_colbert.onnx"
    assert m.dim == 96
    assert m.max_tokens == 300
    assert m.download_size_mb == 127
    assert libembedding.list_multivector_models() == models


def test_registry_id_offline_empty_cache(tmp_path: Path) -> None:
    """A registry id with an empty cache and offline=True is a download error,
    not a 'directory not found' one: the id was recognised."""
    with pytest.raises(DownloadError):
        MultiVectorEmbedding(REGISTRY_ID, cache_dir=str(tmp_path), offline=True)
    assert not any(tmp_path.iterdir())  # offline never creates the model dir


def test_unknown_id_is_not_a_registry_model(tmp_path: Path) -> None:
    with pytest.raises(LembedError) as exc:
        MultiVectorEmbedding("some-org/not-a-colbert", cache_dir=str(tmp_path), offline=True)
    assert not isinstance(exc.value, DownloadError)


def test_local_directory_shadows_registry(tmp_path: Path) -> None:
    """An existing directory named like a registry id is loaded as a directory
    (here an empty one, hence the I/O error, never a download)."""
    local = tmp_path / "answerdotai" / "answerai-colbert-small-v1"
    local.mkdir(parents=True)
    with pytest.raises(LembedError) as exc:
        MultiVectorEmbedding(str(local), cache_dir=str(tmp_path / "cache"), offline=True)
    assert not isinstance(exc.value, DownloadError)


@pytest.mark.network
def test_registry_id_downloads_and_embeds() -> None:
    try:
        m = MultiVectorEmbedding(REGISTRY_ID)
    except DownloadError:
        pytest.skip("model download unavailable")
    with m:
        assert m.dim == 96
        assert m.info().name == REGISTRY_ID
        docs = m.embed(["ColBERT keeps one vector per token."], mode="document")
        q = m.embed(["what does colbert keep?"], mode="query")
        assert docs[0].shape[1] == 96 and docs[0].shape[0] > 3
        assert q[0].shape == (32, 96)
        assert maxsim(q[0], docs[0]) > 0


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
def test_embed_ragged_matches_embed(mv: MultiVectorEmbedding) -> None:
    from libembedding import maxsim_batch

    texts = ["Paris is the capital of France.", "Bananas grow in tropical climates."]
    docs = mv.embed(texts)
    data, offsets = mv.embed_ragged(texts)
    assert data.dtype == np.float32 and data.flags["C_CONTIGUOUS"]
    assert offsets.dtype == np.int64 and offsets.shape == (3,)
    for i, d in enumerate(docs):
        assert np.array_equal(data[offsets[i] : offsets[i + 1]], d)
    q = mv.embed(["capital of France"], mode="query")[0]
    assert maxsim_batch(q, (data, offsets)).tolist() == maxsim_batch(q, docs).tolist()
    empty_data, empty_offsets = mv.embed_ragged([])
    assert empty_data.shape == (0, 96) and empty_offsets.tolist() == [0]


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
