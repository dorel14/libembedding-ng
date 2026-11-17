"""Sparse autotune with the GGUF runtime (name/path routing, result fields).

The grid, the cache hit and the default fallback are covered by the C++ tests
(``test_sparse_gguf``, ``test_autotune_core``) on a generated fixture. Here: the
Python entry point carries a GGUF name or path to the C layer unchanged and
exposes the GGUF fields. No timing is asserted.

Author: David Orel
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os

import pytest

import libembedding.sparse_text_embedding as ste
from libembedding.exceptions import LembedError
from libembedding.sparse_text_embedding import sparse_autotune


class _FakeLib:
    """Records the model name and fills the result like the C layer would."""

    LEMBED_AUTOTUNE_FULL = 1
    LEMBED_AUTOTUNE_QUICK = 0

    def __init__(self) -> None:
        self.seen: list[bytes] = []

    def lembed_sparse_autotune(self, name, mode, result):
        self.seen.append(bytes(name))
        result.threads = 4
        result.batch_size = 16
        result.workers = 2
        result.max_batch_tokens = 1024
        result.from_cache = len(self.seen) > 1
        result.throughput_docs_sec = 10.0
        return 0  # LEMBED_OK


@pytest.mark.parametrize(
    "name", ["SPLADE-PP-En-v1-GGUF", "some/dir/model.gguf"]
)
def test_gguf_name_reaches_the_c_layer_unchanged(monkeypatch, name):
    fake = _FakeLib()
    monkeypatch.setattr(ste, "lib", fake)

    first = sparse_autotune(name)
    second = sparse_autotune(name)

    assert fake.seen == [name.encode(), name.encode()]
    assert (first.workers, first.max_batch_tokens) == (2, 1024)
    assert first.from_cache is False
    assert second.from_cache is True


def test_missing_gguf_path_is_refused(tmp_path):
    with pytest.raises(LembedError):
        sparse_autotune(str(tmp_path / "no_such_model.gguf"))


@pytest.mark.skipif(
    not os.environ.get("LEMBED_TEST_SPARSE_GGUF"),
    reason="set LEMBED_TEST_SPARSE_GGUF to a sparse GGUF file to run",
)
def test_real_gguf_second_call_comes_from_cache(tmp_path, monkeypatch):
    # A private cache, so no earlier run can turn the first call into a hit.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    path = os.environ["LEMBED_TEST_SPARSE_GGUF"]

    first = sparse_autotune(path)
    second = sparse_autotune(path)

    assert first.from_cache is False
    assert second.from_cache is True
    assert (second.threads, second.workers, second.batch_size) == (
        first.threads,
        first.workers,
        first.batch_size,
    )
    assert second.max_batch_tokens == first.max_batch_tokens
