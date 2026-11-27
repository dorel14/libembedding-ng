# Auteur: David Orel
# SPDX-License-Identifier: MIT
"""ONNX dense batching from Python.

The binding used to build the C options with a zero-filled ffi.new() struct.
batch_strategy 0 is LEMBED_BATCH_SEQUENTIAL: one ORT call per text, with
batch_size silently ignored (measured: 1000 ORT calls for 1000 texts, 0.57x the
throughput of the batched path on all-MiniLM-L6-v2). The binding now starts
from lembed_text_options_default(), whose strategy is LEMBED_BATCH_LENGTH_BUCKET.

Quantized variants keep one text per run: dynamic int8 scales are computed over
the whole batch, so batching moves their vectors (cosine 0.95 to 0.98 against
the same text alone, measured with bare onnxruntime).
"""

import numpy as np
import pytest

TEXTS = [
    "Hi.",
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings map text to vectors.",
    " ".join(["performance"] * 40) + ".",
    "Short one",
    "A medium sentence about batching several sequences into a single run.",
] * 8  # 48 texts


def test_onnx_dense_runs_batches_not_one_call_per_text(bge_small):
    with bge_small(offline=True, batch_size=32) as m:
        before = m.stats().batches_run
        batch = np.asarray(m.embed(TEXTS, batch_size=32))
        runs = m.stats().batches_run - before
        # 48 short texts, batch 32: a handful of ORT runs, far from 48.
        assert 1 <= runs <= len(TEXTS) // 4, f"{runs} ORT runs for {len(TEXTS)} texts"

        single = np.vstack([np.asarray(m.embed([t])) for t in TEXTS])

    cos = np.sum(batch * single, axis=1) / (
        np.linalg.norm(batch, axis=1) * np.linalg.norm(single, axis=1)
    )
    # FP32, padding masked: batch composition does not move the vectors.
    assert cos.min() >= 0.99999


def test_quantized_variant_keeps_one_text_per_run():
    from libembedding import TextEmbedding, models
    from libembedding.exceptions import LembedError

    quantized = [
        m
        for m in models.list_text_models()
        if m.model_name == "BAAI/bge-small-en-v1.5"
        and m.quantization in ("static", "dynamic")
    ]
    if not quantized:
        pytest.skip("no quantized bge-small entry in the registry")
    try:
        m = TextEmbedding(
            quantized[0].model_name,
            quantization=quantized[0].quantization,
            offline=True,
            show_download_progress=False,
        )
    except LembedError:
        pytest.skip("quantized bge-small is not in the local cache")
    with m:
        before = m.stats().batches_run
        m.embed(TEXTS, batch_size=32)
        assert m.stats().batches_run - before == len(TEXTS)
