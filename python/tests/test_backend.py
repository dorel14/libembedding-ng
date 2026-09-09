"""Unit tests for backend auto-detection."""

import os
import pytest

from libembedding.backend import detect_backend, backend_to_enum


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("model.gguf", "llama"),
        ("model.GGUF", "llama"),
        ("model.onnx", "onnx"),
        ("BAAI/bge-small-en-v1.5", "onnx"),
        ("Xenova/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2-Q4_K_M.gguf", "llama"),
    ],
)
def test_detect_backend_auto(model_name, expected):
    assert detect_backend(model_name) == expected


def test_detect_backend_explicit_onnx():
    assert detect_backend("anything", "onnx") == "onnx"


def test_detect_backend_explicit_llama():
    assert detect_backend("anything", "llama") == "llama"


def test_detect_backend_local_dir_with_onnx(tmp_path):
    onnx_dir = tmp_path / "model_dir"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_text("fake")
    assert detect_backend(str(onnx_dir)) == "onnx"


def test_detect_backend_local_dir_with_gguf(tmp_path):
    gguf_dir = tmp_path / "model_dir"
    gguf_dir.mkdir()
    (gguf_dir / "model.gguf").write_text("fake")
    assert detect_backend(str(gguf_dir)) == "llama"


def test_detect_backend_local_path_onnx_file(tmp_path):
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_text("fake")
    assert detect_backend(str(onnx_file)) == "onnx"


def test_backend_to_enum():
    assert backend_to_enum("cpu") == 0
    assert backend_to_enum("onnx") == 0
    assert backend_to_enum("llama") == 5
    assert backend_to_enum("llamacpp") == 5
    assert backend_to_enum("auto") == 2
    assert backend_to_enum("unknown") == 2


def test_backend_to_enum_case_insensitive():
    assert backend_to_enum("CPU") == 0
    assert backend_to_enum("LLAMA") == 5
    assert backend_to_enum("Auto") == 2
