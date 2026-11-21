"""Effective backend and provider introspection (T9, decision D3).

Hermetic tests use fake model files: an explicit provider other than "cpu"
that the build cannot attach is refused before the model is parsed, so they
need no model and no network. The tests marked ``network`` load real models
(downloaded on first use, then served from the cache).

Auteur: David Orel
Version: 1.14.0

SPDX-License-Identifier: MIT
"""

import sys

import pytest

from libembedding import (
    ImageEmbedding,
    Reranker,
    SparseTextEmbedding,
    TextEmbedding,
)
from libembedding._binding import ffi, lib
from libembedding.exceptions import (
    DownloadError,
    OnnxRuntimeError,
    UnsupportedError,
)
from libembedding.types import ModelDesc

# Providers no shipped build compiles in. CoreML is real on Apple.
UNAVAILABLE = ["cuda", "directml", "tensorrt"] + (
    [] if sys.platform == "darwin" else ["coreml"]
)

# These tests download whole models (the ONNX SPLADE and CLIP vision files are
# several hundred MB). The suite-wide --timeout=120 is for compute, not for a
# slow runner download, which made release wheel jobs fail at random.
DOWNLOAD_TIMEOUT_S = 600


def download_test(func):
    """Marks a test as needing the network, with a download-sized timeout."""
    return pytest.mark.network(pytest.mark.timeout(DOWNLOAD_TIMEOUT_S)(func))


@pytest.fixture
def fake_model_dir(tmp_path):
    """A model directory whose files are not a real model."""
    (tmp_path / "model.onnx").write_bytes(b"not an onnx model")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "config.json").write_text(
        '{"hidden_size": 8, "max_position_embeddings": 16}'
    )
    return str(tmp_path)


def test_model_desc_new_fields_default_empty():
    """Positional construction and defaults stay backward compatible."""
    desc = ModelDesc("m", 8, 16, "mean", 1, 32)
    assert desc.provider == "cpu"
    assert desc.backend == ""
    assert desc.effective_provider == ""


@pytest.mark.parametrize("provider", UNAVAILABLE)
def test_text_explicit_provider_unavailable_raises(fake_model_dir, provider):
    with pytest.raises(UnsupportedError) as exc:
        TextEmbedding(fake_model_dir, provider=provider, dim=8)
    assert "not available" in exc.value.detail


@pytest.mark.parametrize("provider", UNAVAILABLE)
def test_reranker_explicit_provider_unavailable_raises(fake_model_dir, provider):
    with pytest.raises(UnsupportedError):
        Reranker(fake_model_dir, provider=provider, show_download_progress=False)


@pytest.mark.parametrize("provider", UNAVAILABLE)
def test_sparse_explicit_provider_unavailable_raises(fake_model_dir, provider):
    with pytest.raises(UnsupportedError):
        SparseTextEmbedding(
            fake_model_dir, provider=provider, show_download_progress=False
        )


@pytest.mark.parametrize("provider", UNAVAILABLE)
def test_image_explicit_provider_unavailable_raises(fake_model_dir, provider):
    with pytest.raises(UnsupportedError):
        ImageEmbedding(
            fake_model_dir, provider=provider, dim=8, show_download_progress=False
        )


def test_llama_provider_on_onnx_model_raises(fake_model_dir):
    """llama.cpp is not an ONNX Runtime provider: no silent CPU run either."""
    with pytest.raises(UnsupportedError) as exc:
        TextEmbedding(fake_model_dir, provider="llama", dim=8)
    assert "llama.cpp" in exc.value.detail


def test_cpu_provider_is_never_unsupported(fake_model_dir):
    """Same fake files on CPU: the model is what fails, not the provider."""
    with pytest.raises(OnnxRuntimeError):
        TextEmbedding(fake_model_dir, provider="cpu", dim=8)


# ---- Real models ----------------------------------------------------------


def _gguf_path(registry_name: str) -> str:
    """Local path of a GGUF registry entry, downloading it if needed."""
    info = lib.lembed_find_gguf_model(registry_name.encode("utf-8"))
    assert info != ffi.NULL, registry_name
    return _download_gguf(ffi.string(info.gguf_url).decode("utf-8"), registry_name)


def _download_gguf(url: str, label: str) -> str:
    """Local path of a https://huggingface.co/<repo>/resolve/main/<file> URL."""
    repo, filename = url.split("huggingface.co/", 1)[1].split("/resolve/main/", 1)
    out = ffi.new("char **")
    status = lib.lembed_ensure_gguf_model(
        repo.encode("utf-8"), filename.encode("utf-8"), ffi.NULL, 0, 0, out
    )
    if status != 0 or out[0] == ffi.NULL:
        pytest.skip(f"{label} GGUF unavailable (network restriction?)")
    try:
        return ffi.string(out[0]).decode("utf-8")
    finally:
        lib.lembed_free_string(out[0])


def test_onnx_text_reports_onnxruntime_cpu(bge_small):
    model = bge_small()
    assert model.backend == "onnxruntime"
    assert model.provider == "cpu"
    info = model.info()
    assert info.backend == "onnxruntime"
    assert info.effective_provider == "cpu"
    assert info.provider == "cpu"


@download_test
def test_gguf_text_reports_llamacpp():
    model = TextEmbedding(_gguf_path("Snowflake-XS-Q4"), show_download_progress=False)
    assert model.backend == "llama.cpp"
    assert model.provider == "cpu"
    info = model.info()
    assert info.backend == "llama.cpp"
    assert info.effective_provider == "cpu"
    # The requested-provider field keeps its historical value.
    assert info.provider == "llama"


@download_test
def test_reranker_auto_falling_back_to_gguf_reports_llamacpp():
    """backend="auto" with a name the ONNX registry does not know: the
    reranker is served by llama.cpp, and says so."""
    model = Reranker(
        _gguf_path("Jina-Reranker-v1-Turbo-En"),
        backend="auto",
        show_download_progress=False,
    )
    assert model.backend == "llama.cpp"
    assert model.provider == "cpu"
    info = model.info()
    assert info.backend == "llama.cpp"
    assert info.effective_provider == "cpu"
    assert info.name  # desc_v2 is filled for GGUF rerankers too


@download_test
def test_onnx_reranker_reports_onnxruntime():
    model = Reranker(
        "jinaai/jina-reranker-v1-turbo-en-quantized", show_download_progress=False
    )
    assert model.backend == "onnxruntime"
    assert model.provider == "cpu"
    assert model.info().backend == "onnxruntime"


@download_test
def test_onnx_sparse_reports_onnxruntime():
    try:
        model = SparseTextEmbedding(
            "prithivida/Splade_PP_en_v1", show_download_progress=False
        )
    except DownloadError:
        pytest.skip("sparse model download unavailable")
    assert model.backend == "onnxruntime"
    assert model.provider == "cpu"
    assert model.info().backend == "onnxruntime"


@download_test
def test_gguf_sparse_reports_ggml_sparse():
    info = lib.lembed_find_sparse_gguf_model(b"SPLADE-PP-En-v1-GGUF")
    assert info != ffi.NULL
    path = _download_gguf(ffi.string(info.gguf_url).decode("utf-8"), "SPLADE")
    model = SparseTextEmbedding.from_gguf(path, show_download_progress=False)
    assert model.backend == "ggml-sparse"
    assert model.provider == "cpu"
    assert model.info().backend == "ggml-sparse"


@download_test
def test_image_reports_onnxruntime():
    try:
        model = ImageEmbedding(
            "Qdrant/clip-ViT-B-32-vision", show_download_progress=False
        )
    except DownloadError:
        pytest.skip("image model download unavailable")
    assert model.backend == "onnxruntime"
    assert model.provider == "cpu"
    assert model.info().effective_provider == "cpu"
