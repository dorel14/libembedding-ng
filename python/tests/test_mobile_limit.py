"""The mobile model limit: light registry models only on Android and iOS.

On a phone, a registry model above 150 MB is refused by name before anything is
downloaded, and list_*_models() hides it. LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT=1
applies the same rule here, so the behaviour is tested on every platform;
offline=True keeps the tests off the network.

Auteur: David Orel
Version: 1.13.0

SPDX-License-Identifier: MIT
"""

import sys

import pytest

from libembedding import (
    TextEmbedding,
    list_reranker_models,
    list_text_models,
    mobile_model_limit_mb,
)
from libembedding.exceptions import LembedError, UnsupportedError

ON_MOBILE = sys.platform in ("android", "ios")


@pytest.fixture
def mobile_rules(monkeypatch):
    monkeypatch.delenv("LIBEMBEDDING_ALLOW_LARGE_MODELS", raising=False)
    monkeypatch.setenv("LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT", "1")


def test_limit_depends_on_the_platform(monkeypatch):
    monkeypatch.delenv("LIBEMBEDDING_ALLOW_LARGE_MODELS", raising=False)
    monkeypatch.delenv("LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT", raising=False)
    assert mobile_model_limit_mb() == (150 if ON_MOBILE else 0)


def test_every_listed_model_has_a_size(monkeypatch):
    monkeypatch.setenv("LIBEMBEDDING_ALLOW_LARGE_MODELS", "1")
    models = list_text_models()
    assert models
    assert all(m.download_size_mb > 0 for m in models)


def test_lists_keep_only_light_models(mobile_rules):
    assert mobile_model_limit_mb() == 150
    text = list_text_models()
    assert text
    assert all(0 < m.download_size_mb <= 150 for m in text)
    assert any(m.model_code == "Xenova/bge-small-en-v1.5" for m in text)
    assert not any(m.model_code == "Xenova/bge-large-en-v1.5" for m in text)
    assert all(m.download_size_mb <= 150 for m in list_reranker_models())


def test_heavy_model_is_refused_with_alternatives(mobile_rules):
    with pytest.raises(UnsupportedError) as excinfo:
        TextEmbedding("BAAI/bge-large-en-v1.5", offline=True)
    message = str(excinfo.value)
    assert "limit for models on mobile devices" in message
    assert "Xenova/bge-small-en-v1.5" in message
    assert "LIBEMBEDDING_ALLOW_LARGE_MODELS=1" in message


def test_light_model_is_not_refused(mobile_rules):
    try:
        TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", offline=True).close()
    except UnsupportedError:  # pragma: no cover - the failure being tested
        pytest.fail("a light model was refused by the mobile limit")
    except LembedError:
        pass  # not in the local cache: offline mode, expected


def test_allow_large_models_lifts_the_limit(mobile_rules, monkeypatch):
    monkeypatch.setenv("LIBEMBEDDING_ALLOW_LARGE_MODELS", "1")
    assert mobile_model_limit_mb() == 0
    try:
        TextEmbedding("BAAI/bge-large-en-v1.5", offline=True).close()
    except UnsupportedError:  # pragma: no cover - the failure being tested
        pytest.fail("LIBEMBEDDING_ALLOW_LARGE_MODELS=1 did not lift the limit")
    except LembedError:
        pass  # not in the local cache: offline mode, expected
