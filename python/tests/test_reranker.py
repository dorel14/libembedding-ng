"""Tests for Reranker API."""

import pytest

from libembedding.exceptions import LembedError


def test_reranker_list_supported_models():
    from libembedding import Reranker

    models = Reranker.list_supported_models()
    assert len(models) > 0
    assert any("bge-reranker" in m.model_name for m in models)
    for m in models:
        # Reranker models have dim = 0 (they output scores, not embeddings)
        assert m.model_code
        assert m.max_tokens > 0


@pytest.mark.network
def test_reranker_auto_profile_map():
    """Every documented profile must build a usable reranker (or skip offline)."""
    from libembedding.reranker import Reranker_auto

    for profile in ("fast", "balanced", "quality"):
        try:
            reranker = Reranker_auto(profile, offline=True)
        except (OSError, RuntimeError, ValueError, LembedError):
            pytest.skip(f"model unavailable for reranker profile {profile!r}")
        try:
            desc = reranker.info()
            assert reranker.name, f"empty name for profile {profile!r}"
            assert desc.max_length > 0, f"invalid max_length for profile {profile!r}"
            assert desc.batch_size > 0, f"invalid batch_size for profile {profile!r}"
        finally:
            reranker.close()


def test_reranker_rerank_result_dataclass():
    from libembedding.types import RerankResult

    result = RerankResult(index=0, score=0.95)
    assert result.index == 0
    assert result.score == 0.95


def test_reranker_tuning_result_dataclass():
    from libembedding.types import RerankerTuningResult

    result = RerankerTuningResult(
        threads=4,
        batch_size=32,
        max_tokens=512,
        throughput_docs_sec=100.0,
        latency_ms=50.0,
        p95_latency_ms=80.0,
        memory_mb=200.0,
    )
    assert result.threads == 4
    assert result.batch_size == 32
    assert result.max_tokens == 512


def test_reranker_stats_dataclass():
    from libembedding.types import Stats

    stats = Stats(texts_embedded=10, batches_run=2, avg_latency_ms=15.5)
    assert stats.texts_embedded == 10
    assert stats.batches_run == 2
    assert stats.avg_latency_ms == 15.5


def test_reranker_model_desc_dataclass():
    from libembedding.types import ModelDesc

    desc = ModelDesc(
        name="test-model",
        dimension=384,
        max_length=512,
        pooling=0,
        num_threads=4,
        batch_size=32,
        provider=0,
        device_id=0,
    )
    assert desc.name == "test-model"
    assert desc.dimension == 384


def test_reranker_clear_cache():
    """Clearing the reranker autotune cache must reach the C layer, twice."""
    from libembedding._binding import lib
    from libembedding.reranker import clear_reranker_autotune_cache

    assert hasattr(lib, "lembed_autotune_unified_clear_cache")
    assert clear_reranker_autotune_cache() is None
    assert clear_reranker_autotune_cache() is None
    assert (
        clear_reranker_autotune_cache("jinaai/jina-reranker-v1-turbo-en-quantized")
        is None
    )


def test_reranker_auto_config_invalid_objective():
    from libembedding.reranker import reranker_auto_config

    with pytest.raises(ValueError, match="Unknown objective"):
        reranker_auto_config(
            "jinaai/jina-reranker-v1-turbo-en-quantized", objective="invalid"
        )


def test_reranker_autotune_constrained_invalid_objective():
    from libembedding.reranker import reranker_autotune_constrained

    with pytest.raises(ValueError, match="Unknown objective"):
        reranker_autotune_constrained(
            "jinaai/jina-reranker-v1-turbo-en-quantized", objective="invalid"
        )


def test_reranker_auto_config_profile_invalid():
    from libembedding.reranker import reranker_auto_config_profile

    with pytest.raises(ValueError, match="Unknown profile"):
        reranker_auto_config_profile("invalid")


def test_reranker_info_offline_missing():
    from libembedding import Reranker

    with pytest.raises(LembedError):
        Reranker("nonexistent-reranker-model", offline=True)
