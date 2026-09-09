"""Unit tests for libembedding exception hierarchy."""

import pytest

from libembedding.exceptions import (
    LembedError,
    InvalidArgumentError,
    OutOfMemoryError,
    OnnxRuntimeError,
    TokenizerError,
    DownloadError,
    IOError,
    ModelNotFoundError,
    UnsupportedError,
    BatchSizeError,
    LlamaError,
)


def test_lembed_error_base():
    err = LembedError(1, "test message", "test detail")
    assert err.status_code == 1
    assert err.message == "test message"
    assert err.detail == "test detail"
    assert str(err) == "test message: test detail"


def test_lembed_error_no_detail():
    err = LembedError(2, "test message")
    assert str(err) == "test message"


def test_invalid_argument_error():
    err = InvalidArgumentError(1, "bad arg")
    assert isinstance(err, LembedError)
    assert err.status_code == 1


def test_out_of_memory_error():
    err = OutOfMemoryError(2, "oom")
    assert isinstance(err, LembedError)


def test_onnx_runtime_error():
    err = OnnxRuntimeError(3, "onnx failed")
    assert isinstance(err, LembedError)


def test_tokenizer_error():
    err = TokenizerError(4, "tokenizer failed")
    assert isinstance(err, LembedError)


def test_download_error():
    err = DownloadError(5, "download failed", "network timeout")
    assert isinstance(err, LembedError)
    assert err.detail == "network timeout"


def test_io_error():
    err = IOError(6, "io failed")
    assert isinstance(err, LembedError)


def test_model_not_found_error():
    err = ModelNotFoundError(7, "not found", "model xyz")
    assert isinstance(err, LembedError)
    assert "model xyz" in str(err)


def test_unsupported_error():
    err = UnsupportedError(8, "unsupported")
    assert isinstance(err, LembedError)


def test_batch_size_error():
    err = BatchSizeError(9, "batch size error")
    assert isinstance(err, LembedError)


def test_llama_error():
    err = LlamaError(10, "llama error")
    assert isinstance(err, LembedError)


def test_exception_hierarchy():
    assert issubclass(InvalidArgumentError, LembedError)
    assert issubclass(OutOfMemoryError, LembedError)
    assert issubclass(OnnxRuntimeError, LembedError)
    assert issubclass(TokenizerError, LembedError)
    assert issubclass(DownloadError, LembedError)
    assert issubclass(IOError, LembedError)
    assert issubclass(ModelNotFoundError, LembedError)
    assert issubclass(UnsupportedError, LembedError)
    assert issubclass(BatchSizeError, LembedError)
    assert issubclass(LlamaError, LembedError)
