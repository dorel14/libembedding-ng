"""Unit tests for libembedding model registry and resolution."""

import pytest


def test_list_text_models():
    from libembedding.models import list_text_models

    models = list_text_models()
    assert len(models) > 0
    assert any("bge-small" in m.model_name for m in models)
    for m in models:
        assert m.dim > 0
        assert m.model_code
        assert m.model_file
        assert m.description
        assert m.max_tokens > 0
        assert m.pooling in ("cls", "mean")
        assert m.quantization in ("none", "static", "dynamic")


def test_list_sparse_models():
    from libembedding.models import list_sparse_models

    models = list_sparse_models()
    assert len(models) > 0
    for m in models:
        assert m.dim > 0
        assert m.model_code


def test_list_image_models():
    from libembedding.models import list_image_models

    models = list_image_models()
    assert len(models) > 0
    for m in models:
        assert m.dim > 0
        assert m.model_code


def test_list_reranker_models():
    from libembedding.models import list_reranker_models

    models = list_reranker_models()
    assert len(models) > 0
    for m in models:
        assert m.dim > 0
        assert m.model_code


def test_resolve_text_model_by_code():
    from libembedding.models import resolve_text_model

    idx = resolve_text_model("bge-small-en-v1.5")
    assert idx >= 0


def test_resolve_text_model_by_name():
    from libembedding.models import resolve_text_model

    idx = resolve_text_model("BAAI/bge-small-en-v1.5")
    assert idx >= 0


def test_resolve_text_model_not_found():
    from libembedding.exceptions import ModelNotFoundError
    from libembedding.models import resolve_text_model

    with pytest.raises(ModelNotFoundError):
        resolve_text_model("nonexistent-model-xyz-123")


def test_resolve_sparse_model_by_code():
    from libembedding.models import resolve_sparse_model

    idx = resolve_sparse_model("SPLADE_PP_en_v1")
    assert idx >= 0


def test_resolve_sparse_model_not_found():
    from libembedding.exceptions import ModelNotFoundError
    from libembedding.models import resolve_sparse_model

    with pytest.raises(ModelNotFoundError):
        resolve_sparse_model("nonexistent-sparse-model")


def test_resolve_image_model_by_code():
    from libembedding.models import resolve_image_model

    idx = resolve_image_model("clip-vit-b32")
    assert idx >= 0


def test_resolve_image_model_by_name():
    from libembedding.models import resolve_image_model

    idx = resolve_image_model("Qdrant/clip-ViT-B-32-vision")
    assert idx >= 0


def test_resolve_image_model_not_found():
    from libembedding.exceptions import ModelNotFoundError
    from libembedding.models import resolve_image_model

    with pytest.raises(ModelNotFoundError):
        resolve_image_model("nonexistent-image-model")


def test_resolve_reranker_model_by_code():
    from libembedding.models import resolve_reranker_model

    idx = resolve_reranker_model("bge-reranker-base")
    assert idx >= 0


def test_resolve_reranker_model_not_found():
    from libembedding.exceptions import ModelNotFoundError
    from libembedding.models import resolve_reranker_model

    with pytest.raises(ModelNotFoundError):
        resolve_reranker_model("nonexistent-reranker-model")


def test_is_local_path_file():
    from libembedding.models import _is_local_path

    assert _is_local_path("/path/to/model.gguf") is True
    assert _is_local_path("C:\\path\\to\\model.onnx") is True


def test_is_local_path_missing():
    from libembedding.models import _is_local_path

    assert _is_local_path("/nonexistent/path") is False
    assert _is_local_path("BAAI/bge-small-en-v1.5") is False


def test_is_gguf_model():
    from libembedding.models import _is_gguf_model

    assert _is_gguf_model("model.gguf") is True
    assert _is_gguf_model("model.GGUF") is True
    assert _is_gguf_model("model.onnx") is False
    assert _is_gguf_model("BAAI/bge-small") is False
