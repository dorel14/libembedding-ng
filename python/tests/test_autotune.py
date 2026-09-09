"""Tests for autotune module.

Some tests require the C library and/or network access to download models.
They are marked with @pytest.mark.network or skipped gracefully.
"""

import pytest

from libembedding.autotune import (
    _do_autotune,
    autotune,
    auto_select_model,
    clear_autotune_cache,
    autotune_unified,
)
from libembedding.exceptions import LembedError


def test_clear_autotune_cache_no_args():
    clear_autotune_cache()


def test_clear_autotune_cache_with_model():
    clear_autotune_cache("BAAI/bge-small-en-v1.5")


def test_auto_select_model():
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
    assert 0.0 <= result.score <= 1.0


def test_auto_select_model_speed():
    result = auto_select_model("speed")
    assert result.model_code


def test_auto_select_model_quality():
    result = auto_select_model("quality")
    assert result.model_code


def test_auto_select_model_invalid():
    with pytest.raises((LembedError, ValueError)):
        auto_select_model("invalid_use_case_xyz")


def test_autotune_unified_embedding():
    result = autotune_unified("embedding", "BAAI/bge-small-en-v1.5")
    assert result.task == "embedding"
    assert result.threads >= 1
    assert result.batch_size >= 1
    assert result.throughput_docs_sec >= 0


def test_autotune_unified_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        autotune_unified("invalid_task")


def test_autotune_unified_reranking_no_default():
    with pytest.raises(ValueError, match="No default model"):
        autotune_unified("reranking")


def test_autotune_unified_sparse_no_default():
    with pytest.raises(ValueError, match="No default model"):
        autotune_unified("sparse")
