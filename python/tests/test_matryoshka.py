"""Matryoshka truncation: C helper and TextEmbedding(truncate_dim=...).

The reference is always numpy: keep the first ``k`` components, then divide by
their L2 norm. The C helper and the Python wrapper must reproduce it.

Auteur: David Orel
Version: 1.14.0

SPDX-License-Identifier: MIT
"""

import numpy as np
import pytest

from libembedding._binding import ffi, lib
from libembedding.exceptions import InvalidArgumentError


def _reference(arr: np.ndarray, k: int, renormalize: bool = True) -> np.ndarray:
    out = arr[:, :k].astype(np.float64)
    if renormalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        out = out / norms
    return out.astype(np.float32)


class _CResult:
    """An lembed_embeddings_t over a cffi-owned buffer.

    lembed_embeddings_truncate never reallocates nor frees the buffer, so a
    buffer owned by Python is fine here (and lembed_embeddings_free must not be
    called on it).
    """

    def __init__(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.buf = ffi.new("float[]", arr.size)
        ffi.buffer(self.buf, arr.nbytes)[:] = arr.tobytes()
        self.res = ffi.new("lembed_embeddings_t *")
        self.res.data = self.buf
        self.res.num_embeddings = arr.shape[0]
        self.res.dim = arr.shape[1]

    def to_numpy(self) -> np.ndarray:
        n, d = self.res.num_embeddings, self.res.dim
        flat = np.frombuffer(ffi.buffer(self.res.data, n * d * 4), dtype=np.float32)
        return flat.copy().reshape(n, d)


def _random_unit(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, d)).astype(np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# C helper, synthetic data (no model)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [1, 64, 128, 256, 512, 768])
def test_c_truncate_matches_numpy_reference(k):
    arr = _random_unit(7, 768, seed=k)
    c = _CResult(arr)
    assert lib.lembed_embeddings_truncate(c.res, k, 1) == 0
    assert c.res.dim == k
    assert c.res.num_embeddings == 7
    out = c.to_numpy()
    np.testing.assert_allclose(out, _reference(arr, k), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_c_truncate_without_renormalize_is_a_plain_prefix():
    arr = _random_unit(5, 32, seed=3)
    c = _CResult(arr)
    assert lib.lembed_embeddings_truncate(c.res, 10, 0) == 0
    # Bit-exact: no arithmetic at all, only a move.
    np.testing.assert_array_equal(c.to_numpy(), arr[:, :10])


def test_c_truncate_zero_row_stays_zero():
    arr = _random_unit(3, 8, seed=4)
    arr[1, :4] = 0.0
    c = _CResult(arr)
    assert lib.lembed_embeddings_truncate(c.res, 4, 1) == 0
    out = c.to_numpy()
    np.testing.assert_array_equal(out[1], np.zeros(4, dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_c_truncate_same_dim_renormalizes_only():
    arr = _random_unit(4, 16, seed=5) * 3.0
    c = _CResult(arr)
    assert lib.lembed_embeddings_truncate(c.res, 16, 1) == 0
    np.testing.assert_allclose(c.to_numpy(), _reference(arr, 16), atol=1e-6)


@pytest.mark.parametrize("new_dim", [0, -1, 17])
def test_c_truncate_rejects_out_of_range_dim(new_dim):
    arr = _random_unit(2, 16, seed=6)
    c = _CResult(arr)
    status = lib.lembed_embeddings_truncate(c.res, new_dim, 1)
    assert status == lib.LEMBED_ERROR_INVALID_ARGUMENT
    assert b"new_dim" in ffi.string(lib.lembed_last_error())
    # Rejected calls leave the result untouched.
    assert c.res.dim == 16
    np.testing.assert_array_equal(c.to_numpy(), arr)


def test_c_truncate_rejects_null_result():
    assert lib.lembed_embeddings_truncate(ffi.NULL, 4, 1) == lib.LEMBED_ERROR_INVALID_ARGUMENT
    assert ffi.string(lib.lembed_last_error()) != b""


def test_c_truncate_rejects_null_data():
    res = ffi.new("lembed_embeddings_t *")
    res.num_embeddings = 2
    res.dim = 8
    assert lib.lembed_embeddings_truncate(res, 4, 1) == lib.LEMBED_ERROR_INVALID_ARGUMENT


def test_c_truncate_empty_result_is_ok():
    res = ffi.new("lembed_embeddings_t *")
    res.num_embeddings = 0
    res.dim = 8
    assert lib.lembed_embeddings_truncate(res, 4, 1) == 0
    assert res.dim == 4


# --------------------------------------------------------------------------- #
# TextEmbedding(truncate_dim=...)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0, -8])
def test_truncate_dim_non_positive_rejected_before_loading(bad):
    from libembedding import TextEmbedding

    # The model name is bogus on purpose: the check must fire before any load.
    with pytest.raises(ValueError, match="truncate_dim"):
        TextEmbedding("does-not/exist", truncate_dim=bad, offline=True)


@pytest.mark.parametrize("bad", [True, 64.0, "64"])
def test_truncate_dim_wrong_type_rejected(bad):
    from libembedding import TextEmbedding

    with pytest.raises(TypeError, match="truncate_dim"):
        TextEmbedding("does-not/exist", truncate_dim=bad, offline=True)


TEXTS = [
    "Matryoshka embeddings keep their meaning in a prefix.",
    "The quick brown fox jumps over the lazy dog.",
    "libembedding runs ONNX Runtime and llama.cpp backends.",
]


def test_truncate_dim_matches_numpy_reference(bge_small):
    full = bge_small()
    native = full.dim
    ref = full.embed(TEXTS)
    full.close()

    k = 128
    model = bge_small(truncate_dim=k)
    try:
        assert model.dim == k
        assert model.native_dim == native
        assert model.truncate_dim == k
        assert "native_dim" in repr(model)

        out = model.embed(TEXTS)
        assert out.shape == (len(TEXTS), k)
        np.testing.assert_allclose(out, _reference(ref, k), rtol=0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

        # Every output path agrees with embed().
        iter_out = np.stack(list(model.embed_iter(TEXTS)))
        np.testing.assert_array_equal(iter_out, out)
        raw = np.frombuffer(model.embed_bytes(TEXTS), dtype=np.float32)
        np.testing.assert_array_equal(raw.reshape(len(TEXTS), k), out)

        streamed = []
        model.embed_stream(TEXTS, lambda arr, dim, _ud: streamed.append((arr, dim)))
        assert [d for _, d in streamed] == [k] * len(TEXTS)
        np.testing.assert_allclose(
            np.stack([a for a, _ in streamed]), out, rtol=0, atol=1e-6
        )

        assert model.embed([]).shape == (0, k)
    finally:
        model.close()


def test_truncate_dim_full_size_is_identity(bge_small):
    full = bge_small()
    ref = full.embed(TEXTS)
    native = full.dim
    full.close()

    model = bge_small(truncate_dim=native)
    try:
        assert model.dim == native
        np.testing.assert_allclose(model.embed(TEXTS), ref, rtol=0, atol=1e-6)
    finally:
        model.close()


def test_truncate_dim_above_native_rejected(bge_small):
    probe = bge_small()
    native = probe.dim
    probe.close()
    with pytest.raises(ValueError, match="exceeds the model dimension"):
        bge_small(truncate_dim=native + 1)


def test_invalid_argument_error_type():
    """A Python caller of the raw helper gets InvalidArgumentError via check_status."""
    from libembedding._status import check_status

    c = _CResult(_random_unit(1, 8))
    with pytest.raises(InvalidArgumentError):
        check_status(lib.lembed_embeddings_truncate(c.res, 9, 1))
