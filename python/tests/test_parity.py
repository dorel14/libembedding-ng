"""Cross-platform parity: every platform embeds like the x86_64 reference.

data/parity_bge_small_en_v15.json holds bge-small-en-v1.5 embeddings of three
texts, computed on x86_64. Each platform must give the same dimension and a
cosine of at least 0.999 per text: the build (compiler, SIMD kernels, ONNX
Runtime build, tokenizer) may move the last digits, never the vector. The
Android wheel job selects this test through ``download_smoke``; the native
Android and iOS builds run tests/test_embedding_parity.cpp on the same file.

Auteur: David Orel
Version: 1.13.0

SPDX-License-Identifier: MIT
"""

import json
from pathlib import Path

import numpy as np
import pytest

from libembedding import TextEmbedding

REFERENCE = Path(__file__).parent / "data" / "parity_bge_small_en_v15.json"


@pytest.mark.network
@pytest.mark.download_smoke
def test_embeddings_match_the_x86_64_reference():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = np.asarray(ref["embeddings"], dtype=np.float32)

    model = TextEmbedding(ref["model"], batch_size=4, show_download_progress=False)
    try:
        got = model.embed(ref["texts"])
    finally:
        model.close()

    assert got.shape == expected.shape == (len(ref["texts"]), ref["dim"])
    cosines = np.sum(got * expected, axis=1) / (
        np.linalg.norm(got, axis=1) * np.linalg.norm(expected, axis=1)
    )
    worst = float(cosines.min())
    print(f"parity cosines vs x86_64 reference: {np.round(cosines, 6).tolist()}")
    assert worst >= ref["min_cosine"], (
        f"embedding drifted from the x86_64 reference: cosines {cosines.tolist()}"
    )
