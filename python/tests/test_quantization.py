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

# Instantiating a registry entry downloads its weights, and the quantized half of
# the registry is not homogeneous: gte-large, bge-large, mxbai-embed-large and
# arctic-embed-l are each 1.3-1.7 GB. Walking every entry turned one test into a
# ~10 GB download that CI cannot finish reliably -- it failed on the runner with
# "Failed to download: Alibaba-NLP/gte-large-en-v1.5". The round-trip is checked
# over a cheap subset instead, still covering each quantization mode the registry
# declares. `declared` below fails loudly if one of these stops being quantized,
# so the subset cannot rot into a vacuous pass.
CHEAP_QUANTIZED = (
    "Xenova/all-MiniLM-L6-v2",  # dynamic
    "Xenova/all-MiniLM-L12-v2",  # dynamic
    "Qdrant/bge-small-en-v1.5-onnx-Q",  # fp16
)


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
        declared = {m.model_code: m.quantization for m in list_text_models()}
        for code in CHEAP_QUANTIZED:
            quantization = declared.get(code)
            assert quantization not in (None, "none"), (
                f"{code} is no longer a quantized registry entry; "
                f"CHEAP_QUANTIZED needs updating (declared={quantization!r})"
            )
            model = TextEmbedding(
                code,
                quantization=quantization,
                provider="cpu",
                batch_size=4,
                show_download_progress=False,
            )
            try:
                assert model.quantization == quantization
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


class TestAutoQuantization:
    """preferred_quantization="auto" and what it exposes afterwards.

    This is the path that was reworked to delegate the whole decision to
    ``lembed_quantization_auto_select``: the benchmark, the decision rule and the
    cache entry all live on the C side now. What is left to check on this side is
    the wiring -- that the resolved mode is the one actually loaded, that the
    rationale is attached to *this* instance, and that the fallbacks are FP32.
    """

    def test_auto_reports_a_real_mode_and_a_reason(self, require_cached_model):
        base = _family_with_kinds("none", "dynamic")
        if base is None:
            pytest.skip("no registry family offers a quantized sibling")
        require_cached_model(base)

        model = TextEmbedding(
            base,
            preferred_quantization="auto",
            provider="cpu",
            batch_size=4,
            show_download_progress=False,
        )
        try:
            # Not "auto": the AUTO sentinel is a request, and reporting it back
            # would mean the context kept a mode nothing implements.
            assert model.quantization in ("none", "dynamic", "static", "fp16"), (
                f"auto resolved to {model.quantization!r}"
            )
            # The decision happened, so the caller can be told why. An empty
            # reason here would mean the measurement ran but its outcome was lost.
            assert model.quantization_reason, "auto-selection reported no reason"
            # And the mode reported must be the weights that were loaded.
            assert model.info().name == base
        finally:
            model.close()

    def test_auto_on_a_model_without_siblings_keeps_fp32(self, require_cached_model):
        """A family with nothing to choose between resolves to itself, quietly."""
        families = {}
        for m in list_text_models():
            families.setdefault(m.model_name, set()).add(m.quantization)
        single = next((n for n, k in families.items() if len(k) == 1), None)
        if single is None:
            pytest.skip("every registry family has a quantized sibling")

        require_cached_model(single)
        model = TextEmbedding(
            single,
            preferred_quantization="auto",
            provider="cpu",
            show_download_progress=False,
        )
        try:
            assert model.quantization == "none"
            # Nothing was measured, so there is nothing to justify: a reason
            # here would be claiming a benchmark that never ran.
            assert model.quantization_reason == ""
        finally:
            model.close()

    def test_reason_is_not_shared_between_instances(self, require_cached_model):
        """Each construction carries its own rationale.

        The reason used to travel through a class attribute that the constructor
        read back and reset. Any path returning early -- an unresolvable name, an
        OSError, nothing measurable -- left the previous value in place, so a
        later instance could report another model's decision.
        """
        base = _family_with_kinds("none", "dynamic")
        if base is None:
            pytest.skip("no registry family offers a quantized sibling")
        require_cached_model(base)

        first = TextEmbedding(
            base,
            preferred_quantization="auto",
            provider="cpu",
            batch_size=4,
            show_download_progress=False,
        )
        try:
            auto_reason = first.quantization_reason
            # A plain construction resets the reason: no auto, no rationale.
            plain = TextEmbedding(
                base,
                quantization="none",
                provider="cpu",
                show_download_progress=False,
            )
            try:
                assert plain.quantization_reason == ""
            finally:
                plain.close()
            # The auto instance keeps its own rationale: constructing another
            # model must not overwrite it.
            assert first.quantization_reason == auto_reason
        finally:
            first.close()

    def test_explicit_mode_leaves_the_reason_empty(self, bge_small):
        model = bge_small(quantization="none")
        try:
            assert model.quantization == "none"
            # An explicit choice has nothing to justify.
            assert model.quantization_reason == ""
        finally:
            model.close()

    def test_unknown_mode_is_refused_before_any_work(self):
        with pytest.raises(ValueError, match="Unknown quantization"):
            TextEmbedding(
                MINILM_L6,
                quantization="int4",
                provider="cpu",
                show_download_progress=False,
            )