"""GGUF dense batching regression (1.16.6).

embed() used to run one llama_encode() per text. Texts are now packed into
shared llama_encode() calls, one sequence per text. A packed batch of
mixed-length texts must give the same vectors as embedding each text alone
(same attention isolation, same positions, same order), including a text longer
than the context that is truncated.

Needs the MiniLM GGUF in the local cache; skips when it is absent.

Author: David Orel
SPDX-License-Identifier: MIT
"""

from pathlib import Path

import numpy as np
import pytest

from libembedding import TextEmbedding

GGUF = Path.home() / ".cache" / "libembedding" / "gguf" / "all-MiniLM-L6-v2-Q8_0.gguf"

TEXTS = [
    "Hi.",
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings map text to vectors.",
    " ".join(["performance"] * 40) + ".",
    "Short one",
    "A medium sentence about batching several sequences into a single encode call.",
    "",
    "Café naïve résumé",
    " ".join(["word"] * 900),  # longer than the 512-token context: truncated
    "Another short sentence.",
    "Last text of the batch, a bit longer than the previous one to vary the lengths.",
] * 3  # 33 texts: several packed groups


@pytest.fixture(scope="module")
def model():
    if not GGUF.is_file():
        pytest.skip(f"{GGUF.name} not in local cache")
    with TextEmbedding(str(GGUF), offline=True, show_download_progress=False) as m:
        yield m


def test_packed_batch_matches_per_text(model):
    batch = np.asarray(model.embed(TEXTS))
    single = np.vstack([np.asarray(model.embed([t])) for t in TEXTS])
    assert batch.shape == single.shape
    cos = np.sum(batch * single, axis=1) / (
        np.linalg.norm(batch, axis=1) * np.linalg.norm(single, axis=1)
    )
    # Q8_0 re-quantizes activations between layers, so a different batch shape
    # moves the shortest texts by up to ~5e-4 in cosine (measured: same spread
    # vs ONNX FP32 as the per-text path). A packing bug (leaked attention,
    # wrong positions or order) gives cosines far below 0.999.
    for text, c in zip(TEXTS, cos):
        assert c >= 0.999, f"cosine {c:.6f} for {text[:40]!r}"
    assert cos.mean() >= 0.9998


def test_packed_batch_is_order_independent(model):
    forward = np.asarray(model.embed(TEXTS))
    backward = np.asarray(model.embed(TEXTS[::-1]))[::-1]
    cos = np.sum(forward * backward, axis=1) / (
        np.linalg.norm(forward, axis=1) * np.linalg.norm(backward, axis=1)
    )
    assert cos.min() >= 0.999
