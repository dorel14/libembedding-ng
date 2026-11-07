"""Tests for autotune module.

Some tests require the C library and/or network access to download models.
They are marked with @pytest.mark.network or skipped gracefully.
"""

from __future__ import annotations

import pytest

from libembedding.autotune import (
    _do_autotune,
    _run_autotune,
    _sample_workload,
    _to_c_string_array,
    auto_select_model,
    autotune,
    autotune_unified,
    clear_autotune_cache,
    clear_autotune_cache_unified,
)
from libembedding.exceptions import LembedError, ModelNotFoundError
from libembedding.sampling import _sample_corpus


def _skip_if_nothing_to_tune() -> None:
    """Skip unless at least one auto-select candidate can be benchmarked.

    ``auto_select_impl`` walks its candidate list, skips every candidate whose
    autotune fails, and answers ``MODEL_NOT_FOUND`` when none survived. Because
    the benchmark sessions are created with ``offline = 1``, a runner with an
    empty model cache lands on exactly that answer. It means "nothing cached",
    not "the selector is broken", so the test reports as skipped.

    The probe itself benchmarks every candidate, so its verdict is memoised: one
    probe per session, not one per test.
    """
    global _NOTHING_TO_TUNE
    if _NOTHING_TO_TUNE is None:
        try:
            auto_select_model("balanced")
            _NOTHING_TO_TUNE = False
        except ModelNotFoundError:
            _NOTHING_TO_TUNE = True
    if _NOTHING_TO_TUNE:
        pytest.skip(
            "no auto-select candidate is in the local cache and the "
            "autotuner does not download"
        )


_NOTHING_TO_TUNE: bool | None = None


def test_clear_autotune_cache_no_args():
    """Clearing every entry must reach the C layer and stay idempotent."""
    from libembedding._binding import lib

    assert hasattr(lib, "lembed_autotune_clear_cache")
    assert clear_autotune_cache() is None
    assert clear_autotune_cache() is None


def test_clear_autotune_cache_with_model():
    """Clearing a single model must also be idempotent."""
    from libembedding._binding import lib

    assert hasattr(lib, "lembed_autotune_clear_cache")
    assert clear_autotune_cache("BAAI/bge-small-en-v1.5") is None
    assert clear_autotune_cache("BAAI/bge-small-en-v1.5") is None


@pytest.mark.network
def test_auto_select_model():
    _skip_if_nothing_to_tune()
    result = auto_select_model("balanced")
    assert result.model_code
    assert result.model_name
    assert result.dim > 0
    assert result.workers >= 1
    assert result.threads >= 1
    assert result.batch_size >= 1
    assert result.throughput_docs_sec >= 0
    assert result.latency_ms >= 0
    assert result.memory_mb >= 0
    assert result.score >= 0.0


@pytest.mark.network
def test_auto_select_model_speed():
    _skip_if_nothing_to_tune()
    result = auto_select_model("speed")
    assert result.model_code


@pytest.mark.network
def test_auto_select_model_quality():
    _skip_if_nothing_to_tune()
    result = auto_select_model("quality")
    assert result.model_code


@pytest.mark.network
def test_auto_select_model_invalid():
    with pytest.raises((LembedError, ValueError)):
        auto_select_model("invalid_use_case_xyz")


@pytest.mark.skip(reason="requires model download (network)")
def test_autotune_unified_embedding():
    result = autotune_unified("embedding", "BAAI/bge-small-en-v1.5")
    assert result.task == "embedding"
    assert result.threads >= 1
    assert result.batch_size >= 1
    assert result.throughput_docs_sec >= 0


def test_autotune_unified_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        autotune_unified("invalid_task")


@pytest.mark.skip(reason="requires model download (network)")
def test_autotune_unified_reranking_no_default():
    result = autotune_unified("reranking")
    assert result.task == "reranking"
    assert result.max_tokens >= 1


def test_autotune_unified_sparse_no_default():
    with pytest.raises(ValueError, match="No default model"):
        autotune_unified("sparse")


class TestCustomCorpus:
    """The custom-corpus path must validate its input and bound its work."""

    def test_empty_list_falls_back_to_synthetic(self):
        """An empty corpus means "no corpus", not "tune on nothing"."""
        assert _sample_corpus([], 10) == []

    def test_non_str_entry_is_rejected(self):
        with pytest.raises(TypeError, match="texts must contain str"):
            _run_autotune("BAAI/bge-small-en-v1.5", 0, ["ok", 42])

    def test_corpus_is_sampled_to_the_cap(self):
        """The C layer benchmarks every config against the whole corpus, so an
        unbounded list would make one call arbitrarily long."""
        texts = [f"document number {i} " * (i % 20 + 1) for i in range(5000)]
        assert len(_sample_corpus(texts, 100)) <= 100

    def test_c_string_array_keeps_its_owners_alive(self):
        """The char[] owners must stay reachable for the whole C call."""
        texts = ["alpha", "beta", "gamma"]
        c_array, owners = _to_c_string_array(texts)
        assert len(owners) == 3
        assert all(owner is not None for owner in owners)
        assert c_array is not None

    def test_unicode_round_trip(self):
        texts = ["héllo wörld", "naïve café", "日本語テキスト"]
        _, owners = _to_c_string_array(texts)
        assert len(owners) == 3


@pytest.mark.network
def test_autotune_with_custom_corpus(require_cached_model):
    require_cached_model("BAAI/bge-small-en-v1.5")
    result = autotune("BAAI/bge-small-en-v1.5", texts=["short text", "a longer document"])
    assert result.workers >= 1
    assert result.threads >= 1
    assert result.batch_size >= 1


@pytest.mark.network
def test_do_autotune_with_custom_corpus(require_cached_model):
    require_cached_model("BAAI/bge-small-en-v1.5")
    result = _do_autotune("BAAI/bge-small-en-v1.5", texts=["short text", "another one"])
    assert result.workers >= 1
    assert result.batch_size >= 1


class TestUnifiedOptions:
    """autotune_unified validates its options before reaching the C layer, and
    the C layer rejects what Python cannot check (unknown models)."""

    def test_unknown_objective(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            autotune_unified("embedding", "BAAI/bge-small-en-v1.5", objective="speed")

    def test_image_rejects_custom_workload(self):
        with pytest.raises(ValueError, match="not supported for task 'image'"):
            autotune_unified("image", "Qdrant/clip-ViT-B-32-vision", texts=["x"])

    def test_min_tokens_is_reranking_only(self):
        with pytest.raises(ValueError, match="min_tokens"):
            autotune_unified("embedding", "BAAI/bge-small-en-v1.5", min_tokens=64)

    def test_negative_budget(self):
        with pytest.raises(ValueError, match=">= 0"):
            autotune_unified("reranking", max_latency_ms=-1.0)

    def test_non_str_workload(self):
        with pytest.raises(TypeError, match="texts must contain str"):
            autotune_unified("sparse", "Qdrant/Splade_PP_en_v1", texts=["ok", 3])

    @pytest.mark.parametrize("task", ["embedding", "reranking", "sparse", "image"])
    def test_unknown_model_is_reported(self, task):
        """Resolved before any benchmark, so this needs no model and no network."""
        with pytest.raises(LembedError):
            autotune_unified(task, "no-such-org/no-such-model")

    def test_reranking_sample_keeps_query_last(self):
        docs = [f"document {i}" for i in range(500)]
        sampled = _sample_workload("reranking", [*docs, "the query"], 50)
        assert sampled[-1] == "the query"
        assert len(sampled) <= 50

    def test_options_defaults_follow_the_task(self):
        from libembedding._binding import lib

        assert lib.lembed_autotune_options_default(0).objective == lib.LEMBED_OBJECTIVE_THROUGHPUT
        assert lib.lembed_autotune_options_default(1).objective == lib.LEMBED_OBJECTIVE_BALANCED
        opts = lib.lembed_autotune_options_default(3)
        assert opts.mode == lib.LEMBED_AUTOTUNE_QUICK
        assert opts.n_texts == 0 and opts.max_latency_ms == 0.0 and opts.min_tokens == 0


class TestUnifiedCacheClear:
    @pytest.mark.parametrize("task", ["embedding", "reranking", "sparse", "image"])
    def test_clear_is_idempotent_for_every_task(self, task):
        assert clear_autotune_cache_unified(task, "no-such-org/no-such-model") is None
        assert clear_autotune_cache_unified(task, "no-such-org/no-such-model") is None

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            clear_autotune_cache_unified("video")


@pytest.mark.network
@pytest.mark.parametrize(
    ("task", "model", "objective"),
    [
        ("sparse", "Qdrant/Splade_PP_en_v1", "throughput"),
        ("reranking", "jinaai/jina-reranker-v1-turbo-en", "latency"),
    ],
)
def test_autotune_unified_is_cached(task, model, objective):
    """Second call is a cache hit returning the same configuration.

    The tuners never download: a model missing from the local cache makes every
    configuration fail, which lembed_last_error() reports as "could run".
    """
    try:
        first = autotune_unified(task, model, objective=objective)
    except LembedError as exc:
        if "could run" in str(exc):
            pytest.skip(f"{model} is not in the local cache ({exc})")
        raise
    again = autotune_unified(task, model, objective=objective)
    assert first.threads >= 1 and first.batch_size >= 1
    assert first.throughput_docs_sec > 0 and first.p95_latency_ms >= first.latency_ms
    assert (again.threads, again.batch_size, again.max_tokens) == (
        first.threads,
        first.batch_size,
        first.max_tokens,
    )
