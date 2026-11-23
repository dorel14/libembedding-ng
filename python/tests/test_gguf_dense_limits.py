"""GGUF dense regressions (1.16.1): long inputs and decoder-only models.

- An input above n_ubatch (512) tokens used to hit GGML_ASSERT and abort the
  whole process; it is now truncated.
- A decoder-only GGUF (Qwen3-Embedding) used to load then crash with an access
  violation at the first embed; it is now refused at construction.

Both need a GGUF file already in the local cache; they skip when it is absent.

Author: David Orel
SPDX-License-Identifier: MIT
"""

from pathlib import Path

import numpy as np
import pytest

from libembedding import TextEmbedding
from libembedding.exceptions import UnsupportedError

GGUF_DIR = Path.home() / ".cache" / "libembedding" / "gguf"
ENCODER_GGUF = GGUF_DIR / "all-MiniLM-L6-v2-Q4_K_M.gguf"
DECODER_GGUF = GGUF_DIR / "qwen3-emb" / "Qwen3-Embedding-0.6B-Q8_0.gguf"


@pytest.mark.skipif(not ENCODER_GGUF.is_file(), reason="MiniLM GGUF not in local cache")
@pytest.mark.parametrize("max_length", [0, 64])
def test_long_input_is_truncated_not_fatal(max_length):
    model = TextEmbedding(
        str(ENCODER_GGUF), offline=True, show_download_progress=False, max_length=max_length
    )
    long_text = " ".join(f"word{i % 97}" for i in range(3000))
    vecs = np.asarray(model.embed([long_text, "word0 word1 word2"]))
    assert vecs.shape[0] == 2
    assert np.isfinite(vecs).all()
    assert abs(float(np.linalg.norm(vecs[0])) - 1.0) < 0.01


@pytest.mark.skipif(not ENCODER_GGUF.is_file(), reason="MiniLM GGUF not in local cache")
def test_max_length_truncates_the_prefix():
    """With max_length=16 only the first tokens count, so a long text and its
    own prefix give (nearly) the same vector."""
    model = TextEmbedding(
        str(ENCODER_GGUF), offline=True, show_download_progress=False, max_length=16
    )
    head = " ".join(f"alpha{i}" for i in range(40))
    tail = head + " " + " ".join(f"omega{i}" for i in range(400))
    a, b = np.asarray(model.embed([head, tail]))
    assert float(np.dot(a, b)) > 0.999


@pytest.mark.skipif(not DECODER_GGUF.is_file(), reason="Qwen3-Embedding GGUF not in local cache")
def test_decoder_only_gguf_is_refused():
    with pytest.raises(UnsupportedError) as info:
        TextEmbedding(str(DECODER_GGUF), offline=True, show_download_progress=False)
    assert "decoder-only" in str(info.value)
