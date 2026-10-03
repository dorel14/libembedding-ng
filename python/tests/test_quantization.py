"""Tests for quantization mode selection and model resolution.

Two regressions are locked down here, both of which silently returned the wrong
weights instead of failing:

- a HuggingFace name must select its own registry entry. Leaving
  ``opts.base.model`` at its zero-initialised value pointed every name at entry 0,
  so ``TextEmbedding("BAAI/bge-small-en-v1.5")`` ran all-MiniLM-L6-v2 weights and
  ``info()`` reported the wrong model.
- ``quantization=`` must select the registry entry that provides that mode, since
  a quantized model is a different file rather than a session option.

Auteur: David Orel
Version: 1.9.0

SPDX-License-Identifier: MIT
"""

import numpy as np
import pytest

from libembedding import TextEmbedding, list_text_models
from libembedding.exceptions import LembedError

BGE_SMALL = "BAAI/bge-small-en-v1.5"
MINILM_L6 = "sentence-transformers/all-MiniLM-L6-v2"
PROBE_TEXT = ["machine learning is a subset of artificial intelligence"]


def _variants(base_name):
    return [m for m in list_text_models() if m.model_name == base_name]


def _family_with_kinds(*kinds):
    """First registry family that provides all of `kinds`."""
    families = {}
    for m in list_text_models():
        families.setdefault(m.model_name, set()).add(m.quantization)
    for name, have in families.items():
        if all(k in have for k in kinds):
            return name
    return None


class TestModelResolution:
    """A model name must load its own weights."""

    def test_distinct_models_give_distinct_embeddings(self):
        small = TextEmbedding(
            BGE_SMALL, provider="cpu", batch_size=4, show_download_progress=False
        )
        l6 = TextEmbedding(
            MINILM_L6, provider="cpu", batch_size=4, show_download_progress=False
        )
        try:
            assert small.dim == l6.dim
            # Both are 384-dimensional, so only the values can tell them apart.
            assert np.abs(small.embed(PROBE_TEXT) - l6.embed(PROBE_TEXT)).max() > 1e-4
        finally:
            small.close()
            l6.close()

    def test_info_reports_the_requested_model(self, bge_small):
        model = bge_small()
        assert model.info().name == BGE_SMALL
        model.close()


class TestQuantizationSelection:
    """quantization= must select the weights, not just session options."""

    def test_registry_tags_round_trip(self):
        for info in list_text_models():
            if info.quantization == "none":
                continue
            model = TextEmbedding(
                info.model_code,
                quantization=info.quantization,
                provider="cpu",
                batch_size=4,
                show_download_progress=False,
            )
            try:
                assert model.quantization == info.quantization
            finally:
                model.close()

    def test_family_name_selects_the_quantized_variant(self):
        base = _family_with_kinds("none", "static") or _family_with_kinds(
            "none", "dynamic"
        )
        if base is None:
            pytest.skip("no registry family offers a quantized sibling")
        target = next(v for v in _variants(base) if v.quantization != "none")

        model = TextEmbedding(
            base,
            quantization=target.quantization,
            provider="cpu",
            batch_size=4,
            show_download_progress=False,
        )
        try:
            assert model.quantization == target.quantization
            assert model.info().name == base
        finally:
            model.close()

    def test_unavailable_mode_is_refused_with_available_modes(self):
        base = "sentence-transformers/all-MiniLM-L6-v2"  # none + dynamic only
        with pytest.raises(LembedError) as excinfo:
            TextEmbedding(
                base,
                quantization="static",
                provider="cpu",
                show_download_progress=False,
            )
        message = str(excinfo.value)
        assert "static" in message
        assert "dynamic" in message

    def test_dynamic_variant_accepts_batching(self):
        base = _family_with_kinds("none", "dynamic")
        if base is None:
            pytest.skip("no registry family offers a dynamic sibling")

        batched = TextEmbedding(
            base,
            quantization="dynamic",
            provider="cpu",
            batch_size=4,
            show_download_progress=False,
        )
        unbatched = TextEmbedding(
            base,
            quantization="dynamic",
            provider="cpu",
            batch_size=0,
            show_download_progress=False,
        )
        try:
            # Batching must not change the result. A dynamic variant used to be
            # forced into a single batch over the whole input.
            np.testing.assert_allclose(
                batched.embed(PROBE_TEXT), unbatched.embed(PROBE_TEXT), atol=1e-5
            )
        finally:
            batched.close()
            unbatched.close()