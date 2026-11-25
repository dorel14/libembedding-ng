# SPDX-License-Identifier: MIT
# Auteur: David Orel
"""Out-of-range enum values reach the C API through cffi, which takes any int.

The Python wrappers map names to enum values, so they never produce one; this
test goes through the raw binding the way a third-party caller would. No model
and no network: validation runs before either.
"""

import pytest

from libembedding._binding import ffi, lib

# cffi gives these enums an unsigned underlying type and refuses -1, so the
# all-ones pattern (what a C caller's -1 becomes) stands in for it.
BAD_VALUES = [999, 0xFFFFFFFF]


def _message():
    return ffi.string(lib.lembed_last_error()).decode("utf-8", "replace")


@pytest.mark.parametrize("value", BAD_VALUES)
@pytest.mark.parametrize(
    "getter",
    [
        "lembed_get_text_model_info",
        "lembed_get_sparse_model_info",
        "lembed_get_image_model_info",
        "lembed_get_reranker_model_info",
        "lembed_get_multivector_model_info",
    ],
)
def test_registry_rejects_out_of_range_model(getter, value):
    info = ffi.new("lembed_model_info_t *")
    assert getattr(lib, getter)(value, info) == lib.LEMBED_ERROR_INVALID_ARGUMENT
    assert "Invalid" in _message()


@pytest.mark.parametrize("value", BAD_VALUES)
def test_text_create_rejects_out_of_range_fields(value):
    out = ffi.new("lembed_text_embedding_t **")
    for field in ("model", "backend", "batch_strategy"):
        opts = ffi.new("lembed_text_options_t *", lib.lembed_text_options_default())
        # batch_strategy is a plain int field in the C struct: use the signed value.
        signed = value - (1 << 32) if value >= (1 << 31) else value
        setattr(opts, field, signed if field == "batch_strategy" else value)
        status = lib.lembed_text_embedding_create(opts, out)
        assert status == lib.LEMBED_ERROR_INVALID_ARGUMENT, field
        assert "Invalid" in _message(), field


@pytest.mark.parametrize("value", BAD_VALUES)
def test_unknown_provider_is_unsupported_not_cpu(value):
    out = ffi.new("lembed_reranker_t **")
    opts = ffi.new("lembed_reranker_options_t *", lib.lembed_reranker_options_default())
    opts.provider = value
    status = lib.lembed_reranker_create(opts, out)
    assert status == lib.LEMBED_ERROR_UNSUPPORTED
    assert "unknown lembed_execution_provider_t" in _message()


@pytest.mark.parametrize("value", BAD_VALUES)
def test_multivector_create_unknown_provider_is_unsupported(value):
    # model_dir is a registry name: the provider check must fire before any
    # lookup or download.
    out = ffi.new("lembed_multivector_t **")
    opts = ffi.new("lembed_multivector_options_t *", lib.lembed_multivector_options_default())
    model_dir = ffi.new("char[]", b"answerdotai/answerai-colbert-small-v1")
    opts.model_dir = model_dir
    opts.provider = value
    status = lib.lembed_multivector_create(opts, out)
    assert status == lib.LEMBED_ERROR_UNSUPPORTED
    assert out[0] == ffi.NULL
    assert "unknown lembed_execution_provider_t" in _message()


@pytest.mark.parametrize("value", BAD_VALUES)
def test_reranker_create_rejects_out_of_range_backend_and_model(value):
    out = ffi.new("lembed_reranker_t **")
    for field in ("backend", "model"):
        opts = ffi.new("lembed_reranker_options_t *", lib.lembed_reranker_options_default())
        setattr(opts, field, value)
        assert lib.lembed_reranker_create(opts, out) == lib.LEMBED_ERROR_INVALID_ARGUMENT, field
        assert "Invalid" in _message(), field


@pytest.mark.parametrize("value", BAD_VALUES)
def test_autotune_rejects_out_of_range_mode(value):
    result = ffi.new("lembed_tuning_result_t *")
    status = lib.lembed_autotune(b"sentence-transformers/all-MiniLM-L6-v2", value, result)
    assert status == lib.LEMBED_ERROR_INVALID_ARGUMENT
    assert "Invalid" in _message()
