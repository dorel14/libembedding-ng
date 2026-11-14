# SPDX-License-Identifier: MIT
"""Sparse GGUF registry (lembed_list_sparse_gguf_models), no network.

The model itself is checked against ONNX in test_sparse_gguf_fidelity.py;
these tests pin the registry entry and the name resolution rules.
"""

import pytest

from libembedding._binding import ffi, lib
from libembedding.sparse_text_embedding import _split_hf_resolve_url


def _entries():
    out = ffi.new("lembed_gguf_model_info_t const **")
    count = ffi.new("int *")
    assert lib.lembed_list_sparse_gguf_models(out, count) == 0
    return [out[0][i] for i in range(count[0])]


def test_splade_entry():
    entries = {ffi.string(e.name).decode(): e for e in _entries()}
    entry = entries["SPLADE-PP-En-v1-GGUF"]
    url = ffi.string(entry.gguf_url).decode()
    assert _split_hf_resolve_url(url) == (
        "cstr/splade-pp-en-v1-GGUF", "splade-pp-en-v1-q8_0.gguf")
    assert ffi.string(entry.model_code).decode() == "prithivida/Splade_PP_en_v1"
    assert entry.dim == 30522          # BERT vocabulary: the sparse width
    assert 0 < entry.file_size_mb <= 150  # loadable on mobile without override


def test_find_is_exact_and_case_insensitive():
    assert lib.lembed_find_sparse_gguf_model(b"splade-pp-en-v1-gguf") != ffi.NULL
    # Names of the ONNX model must keep resolving to ONNX.
    for onnx_name in (b"prithivida/Splade_PP_en_v1", b"Splade_PP_en_v1", b"SPLADE-PP-En-v1"):
        assert lib.lembed_find_sparse_gguf_model(onnx_name) == ffi.NULL
    assert lib.lembed_find_sparse_gguf_model(ffi.NULL) == ffi.NULL


def test_dense_registry_does_not_list_it():
    out = ffi.new("lembed_gguf_model_info_t const **")
    count = ffi.new("int *")
    assert lib.lembed_list_gguf_models(out, count) == 0
    names = {ffi.string(out[0][i].name).decode() for i in range(count[0])}
    assert "SPLADE-PP-En-v1-GGUF" not in names


def test_split_rejects_other_urls():
    with pytest.raises(ValueError):
        _split_hf_resolve_url("https://example.com/a/b.gguf")
