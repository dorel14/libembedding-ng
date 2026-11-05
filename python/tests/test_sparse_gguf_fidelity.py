"""Fidelity of the sparse GGUF runtime against the ONNX reference.

Both backends run the same SPLADE++ checkpoint (prithivida/Splade_PP_en_v1):
ONNX Runtime on the FP32 export, the ggml runtime on the Q8_0 GGUF. Q8_0
rounding means the vectors cannot be identical, so these tests pin how close
they are and, more to the point, that the difference does not change what a
search returns.

The thresholds are measured, not guessed: each one sits below the value
observed on this text set, with margin for platform-dependent rounding (see
docs/gguf_sparse_runtime.md). A drop below them is a regression of the
runtime, not noise.

Both models must be in the local cache; the tests skip otherwise.

Auteur: David Orel

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.network

ONNX_MODEL = "prithivida/Splade_PP_en_v1"
GGUF_REPO = "cstr/splade-pp-en-v1-GGUF"
GGUF_FILE = "splade-pp-en-v1-q8_0.gguf"

# Plain ASCII English: the tokenizer is shared by both backends, so accented
# text would measure the tokenizer approximation rather than the runtime.
TEXTS = [
    "a fast brown fox jumps over the lazy dog",
    "how to bake sourdough bread at home",
    "vector databases store embeddings and search them",
    "machine learning models learn from data",
    "the weather today is cold and rainy",
    "installing a package manager on windows",
    "what is the capital city of france",
    "the mitochondria is the powerhouse of the cell",
    (
        "Sparse retrieval models such as SPLADE expand a query into weighted "
        "vocabulary terms, which lets an inverted index answer semantic "
        "queries without dense vectors."
    ),
    (
        "The central bank raised interest rates by a quarter point on Tuesday, "
        "citing persistent inflation in services and a tight labour market."
    ),
]

DOCS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Python programming language emphasizes code readability.",
    "Mount Everest is the highest mountain above sea level, in the Himalayas.",
    "Insulin is a hormone that regulates blood glucose levels.",
    "The Great Wall of China stretches across northern China.",
    "Black holes are regions where gravity prevents light from escaping.",
    "Sourdough bread is leavened with a fermented starter of flour and water.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Vaccines train the immune system to recognize pathogens.",
    "Rust is a systems programming language focused on memory safety.",
    "Coral reefs are underwater ecosystems built by colonies of tiny animals.",
]

# (query, index of the one document that answers it)
QUERIES = [
    ("which tower is in paris", 0),
    ("how do plants make energy from sunlight", 1),
    ("tallest mountain in the world", 3),
    ("hormone controlling blood sugar", 4),
    ("how to make bread with a starter", 7),
    ("memory safe systems language", 10),
]

# Measured on an i7-1065G7 (2026-11-04): cosine min 0.99980 / mean 0.99989,
# top-10 overlap 1.0, score correlation >= 0.99997, batch 1 vs 64 cosine min
# 0.99992. Each bound leaves about ten times the observed gap as margin.
COSINE_MIN = 0.998
COSINE_MEAN_MIN = 0.999
TOP10_OVERLAP_MEAN_MIN = 0.9
SCORE_CORRELATION_MIN = 0.9995
BATCHING_COSINE_MIN = 0.999


def _load(factory):
    from libembedding.exceptions import DownloadError, ModelNotFoundError

    try:
        return factory()
    except (DownloadError, ModelNotFoundError) as exc:
        pytest.skip(f"model not available locally: {exc}")


@pytest.fixture(scope="module")
def onnx_model():
    from libembedding import SparseTextEmbedding

    model = _load(lambda: SparseTextEmbedding(
        ONNX_MODEL, offline=True, show_download_progress=False, threads=4,
    ))
    yield model
    model.close()


@pytest.fixture(scope="module")
def gguf_model():
    from libembedding import SparseTextEmbedding

    model = _load(lambda: SparseTextEmbedding.from_gguf_model(
        GGUF_REPO, GGUF_FILE, offline=True, show_download_progress=False, threads=4,
    ))
    yield model
    model.close()


def _as_dict(emb) -> dict[int, float]:
    return dict(zip(emb.indices.tolist(), emb.values.tolist()))


def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
    dot = sum(w * b[t] for t, w in a.items() if t in b)
    na = np.sqrt(sum(w * w for w in a.values()))
    nb = np.sqrt(sum(w * w for w in b.values()))
    return float(dot / (na * nb)) if na and nb else 0.0


def _top_terms(emb, k: int = 10) -> set[int]:
    return set(emb.indices[np.argsort(-emb.values)[:k]].tolist())


def _scores(query, docs) -> np.ndarray:
    q = _as_dict(query)
    return np.array([sum(w * q.get(t, 0.0) for t, w in _as_dict(d).items())
                     for d in docs])


def test_vectors_match_onnx(onnx_model, gguf_model):
    ref = onnx_model.embed(TEXTS)
    got = gguf_model.embed(TEXTS)

    cosines = [_cosine(_as_dict(a), _as_dict(b)) for a, b in zip(ref, got)]
    overlaps = [len(_top_terms(a) & _top_terms(b)) / 10 for a, b in zip(ref, got)]

    worst = int(np.argmin(cosines))
    assert min(cosines) >= COSINE_MIN, (
        f"cosine {cosines[worst]:.5f} on {TEXTS[worst]!r}")
    assert np.mean(cosines) >= COSINE_MEAN_MIN, f"mean cosine {np.mean(cosines):.5f}"
    assert np.mean(overlaps) >= TOP10_OVERLAP_MEAN_MIN, (
        f"mean top-10 term overlap {np.mean(overlaps):.3f}")


def test_retrieval_matches_onnx(onnx_model, gguf_model):
    """Same answers, and nearly the same scores, on a search with known answers."""
    ref_docs, got_docs = onnx_model.embed(DOCS), gguf_model.embed(DOCS)
    ref_q = onnx_model.embed([q for q, _ in QUERIES])
    got_q = gguf_model.embed([q for q, _ in QUERIES])

    for i, (query, answer) in enumerate(QUERIES):
        ref_scores = _scores(ref_q[i], ref_docs)
        got_scores = _scores(got_q[i], got_docs)
        assert int(np.argmax(ref_scores)) == answer, f"ONNX misses {query!r}"
        assert int(np.argmax(got_scores)) == answer, f"GGUF misses {query!r}"
        corr = float(np.corrcoef(ref_scores, got_scores)[0, 1])
        assert corr >= SCORE_CORRELATION_MIN, f"score correlation {corr:.4f} on {query!r}"


def test_gguf_workers_keep_vectors(gguf_model):
    """Several workers only change which graph computes a batch, not the result."""
    from libembedding import SparseTextEmbedding

    pooled = _load(lambda: SparseTextEmbedding.from_gguf_model(
        GGUF_REPO, GGUF_FILE, offline=True, show_download_progress=False,
        threads=4, workers=4,
    ))
    try:
        texts = TEXTS + DOCS
        ref = gguf_model.embed(texts, batch_size=2)
        got = pooled.embed(texts, batch_size=2)
        assert pooled.info().num_threads == 4
        for a, b in zip(ref, got):
            assert _cosine(_as_dict(a), _as_dict(b)) >= BATCHING_COSINE_MIN
    finally:
        pooled.close()


def test_gguf_batching_keeps_vectors(gguf_model):
    """The token budget only changes the batch shape, never the result.

    Not bit-exact on Q8_0: the GEMM shape changes with the batch, and with it
    the accumulation order.
    """
    texts = TEXTS + DOCS
    singles = gguf_model.embed(texts, batch_size=1)
    batched = gguf_model.embed(texts, batch_size=64)
    cosines = [_cosine(_as_dict(a), _as_dict(b)) for a, b in zip(singles, batched)]
    assert min(cosines) >= BATCHING_COSINE_MIN, f"min cosine {min(cosines):.6f}"
