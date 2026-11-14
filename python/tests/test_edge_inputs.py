# SPDX-License-Identifier: MIT
"""Edge-case inputs across dense, sparse and reranking (roadmap T23).

Only the tokenizer had these cases (tests/test_tokenizer.cpp). Every task must
take them without crashing, return the expected shapes and finite values, and
give the same result for a text whether it is embedded alone or in a batch.

Dense uses bge-small (downloaded if missing, like the other dense tests);
sparse and reranking use their default models only when already cached, as
they are large.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.network

EDGE_TEXTS = [
    "",                                   # empty
    "   ",                                # spaces only
    "\t\t",                               # tabs only
    "line one\r\nline two\r\n",           # CRLF
    "\U0001F600\U0001F680 \U0001F1EB\U0001F1F7",   # emoji and a flag
    "مرحبا שלום",  # Arabic and Hebrew (RTL)
    chr(0x200B).join(["zero", "width", "space"]),  # zero-width spaces (U+200B)
    "été café",          # combining accents (NFD)
    "x" * 1000,                           # one 1000-character word
    " ".join(["token"] * 2000),           # far beyond 512 tokens
    "normal sentence about search engines",  # a regular control text
]


def _cached(factory):
    from libembedding.exceptions import DownloadError, ModelNotFoundError

    try:
        return factory()
    except (DownloadError, ModelNotFoundError) as exc:
        pytest.skip(f"model not in the local cache: {exc}")


def test_dense_edge_inputs(bge_small):
    model = bge_small()
    try:
        batch = model.embed(EDGE_TEXTS)
        assert batch.shape == (len(EDGE_TEXTS), model.dim)
        assert np.all(np.isfinite(batch))
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-4)
        for i, text in enumerate(EDGE_TEXTS):
            alone = model.embed([text])[0]
            assert float(np.dot(alone, batch[i])) >= 0.9999, repr(text[:40])
    finally:
        model.close()


def test_sparse_edge_inputs():
    from libembedding import SparseTextEmbedding

    model = _cached(lambda: SparseTextEmbedding(offline=True, show_download_progress=False))
    try:
        batch = model.embed(EDGE_TEXTS)
        assert len(batch) == len(EDGE_TEXTS)
        for i, (text, emb) in enumerate(zip(EDGE_TEXTS, batch)):
            assert emb.indices.shape == emb.values.shape, repr(text[:40])
            assert np.all(np.isfinite(emb.values)), repr(text[:40])
            assert np.all(emb.values >= 0), repr(text[:40])
            alone = model.embed([text])[0]
            assert dict(zip(alone.indices.tolist(), alone.values.tolist())) == pytest.approx(
                dict(zip(emb.indices.tolist(), emb.values.tolist())), rel=1e-4, abs=1e-5
            ), repr(text[:40])
        # The control text must activate terms; the regular sentence is the check
        # that the edge cases did not corrupt the batch around them.
        assert batch[-1].indices.size > 0
    finally:
        model.close()


def test_rerank_edge_inputs():
    from libembedding import Reranker

    model = _cached(lambda: Reranker(offline=True, show_download_progress=False))
    try:
        for query in ("", "search engines", "\U0001F600", "x" * 1000):
            results = model.rerank(query, EDGE_TEXTS)
            assert sorted(r.index for r in results) == list(range(len(EDGE_TEXTS)))
            scores = [r.score for r in results]
            assert all(np.isfinite(scores)), repr(query[:40])
            assert scores == sorted(scores, reverse=True)
    finally:
        model.close()
