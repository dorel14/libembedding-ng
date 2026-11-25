"""GGUF BERT (WPM) Hangul regression (1.16.5).

llama.cpp's WordPiece tokenizer decomposes with a 1:1 NFD table, so every Hangul
syllable collapsed to its leading consonant jamo (cosine vs ONNX ~0.73). The
GGUF path now decomposes Hangul with the library's HF-exact BERT normalization
before llama.cpp tokenizes. Other scripts must stay unchanged.

Needs the MiniLM GGUF and the MiniLM ONNX model in the local cache; skips when
either is absent.

Author: David Orel
SPDX-License-Identifier: MIT
"""

from pathlib import Path

import numpy as np
import pytest

from libembedding import TextEmbedding
from libembedding.exceptions import LembedError

GGUF_DIR = Path.home() / ".cache" / "libembedding" / "gguf"
ONNX_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# (file, minimum cosine vs ONNX). Q4_K_M quantization noise alone costs ~1 % on
# every text (0.987 on Chinese, which was never affected); the bug gave 0.2-0.7.
GGUF_FILES = [
    (GGUF_DIR / "all-MiniLM-L6-v2-Q8_0.gguf", 0.999),
    (GGUF_DIR / "all-MiniLM-L6-v2-Q4_K_M.gguf", 0.985),
]

HANGUL_TEXTS = ["안녕하세요 세계.", "Hello 세계 world"]
CONTROL_TEXTS = [
    "你好世界",
    "こんにちは世界",
    "Café naïve résumé",
    "Ünïcödé façade",
    "The quick brown fox jumps.",
]


@pytest.fixture(scope="module")
def onnx_vectors():
    try:
        model = TextEmbedding(ONNX_MODEL, offline=True, show_download_progress=False)
    except LembedError as exc:  # model absent from the local cache
        pytest.skip(f"MiniLM ONNX not in local cache: {exc}")
    with model:
        return np.asarray(model.embed(HANGUL_TEXTS + CONTROL_TEXTS))


@pytest.mark.parametrize("gguf,floor", GGUF_FILES, ids=[p.stem for p, _ in GGUF_FILES])
def test_gguf_matches_onnx_on_hangul_and_controls(gguf, floor, onnx_vectors):
    if not gguf.is_file():
        pytest.skip(f"{gguf.name} not in local cache")
    with TextEmbedding(str(gguf), offline=True, show_download_progress=False) as model:
        got = np.asarray(model.embed(HANGUL_TEXTS + CONTROL_TEXTS))
    cos = np.sum(got * onnx_vectors, axis=1)
    texts = HANGUL_TEXTS + CONTROL_TEXTS
    for text, c in zip(texts, cos):
        assert c >= floor, f"{gguf.name}: cosine {c:.4f} for {text!r}"
