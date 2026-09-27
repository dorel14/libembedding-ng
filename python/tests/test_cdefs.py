"""ABI parity between the C headers and the cffi declarations.

``_cdefs.h`` is a hand-written copy of the C declarations. A silent drift
(``float`` vs ``double``, ``int`` vs ``uint64_t``) is invisible to both the C
and the Python compiler, and corrupts memory at runtime.

The expected sizes below are frozen by ``tests/test_struct_sizes.cpp`` on the C
side: the two tables must be changed together.
"""

import struct
import sys

import pytest

from libembedding._binding import ffi

# 64-bit ABI (LP64 and Win64 agree: these structs are flat, no unions/bits)
EXPECTED_SIZES_64 = {
    "lembed_stats_t": 24,
    "lembed_stats_v2_t": 48,
    "lembed_model_desc_t": 40,
    "lembed_model_desc_v2_t": 48,
    "lembed_embeddings_t": 16,
    "lembed_sparse_embedding_t": 24,
    "lembed_rerank_result_t": 8,
    "lembed_cache_config_t": 24,
    "lembed_text_options_t": 88,
    "lembed_text_options_v2_t": 96,
    "lembed_tuning_result_t": 40,
    "lembed_sparse_tuning_result_t": 48,
    "lembed_reranker_tuning_result_t": 48,
    "lembed_unified_tuning_result_t": 64,
    "lembed_cache_hardware_info_t": 460,
    "lembed_tune_config_result_t": 24,
    "lembed_tune_cache_entry_t": 1104,
}

IS_64_BIT = struct.calcsize("P") == 8

requires_64_bit = pytest.mark.skipif(
    not IS_64_BIT, reason="frozen sizes only cover the 64-bit ABI"
)


@requires_64_bit
@pytest.mark.parametrize("type_name", sorted(EXPECTED_SIZES_64))
def test_struct_sizes_match_c(type_name):
    expected = EXPECTED_SIZES_64[type_name]
    actual = ffi.sizeof(type_name)
    assert actual == expected, (
        f"sizeof({type_name}) = {actual} in _cdefs.h, expected {expected} "
        f"(see tests/test_struct_sizes.cpp)"
    )


@pytest.mark.parametrize("type_name", sorted(EXPECTED_SIZES_64))
def test_struct_declared_in_cdefs(type_name):
    """Every frozen struct must still be declared in _cdefs.h."""
    assert ffi.sizeof(type_name) > 0  # raises if the declaration is missing


def test_versioned_stats_have_cache_fields():
    """The v2 stats structs must expose the cache counters used by Python."""
    stats_v2 = ffi.new("lembed_stats_v2_t *")
    for field in ("texts_embedded", "batches_run", "avg_latency_ms"):
        assert hasattr(stats_v2.base, field)
    for field in ("cache_hits", "cache_misses"):
        assert hasattr(stats_v2, field)


def test_tuning_results_use_double_not_float():
    """A float/double drift in a tuning result would corrupt every metric."""
    tuning = ffi.new("lembed_tuning_result_t *")
    tuning.throughput_docs_sec = 1.5
    tuning.latency_ms = 2.5
    tuning.memory_mb = 3.5
    assert tuning.throughput_docs_sec == 1.5
    assert tuning.latency_ms == 2.5
    assert tuning.memory_mb == 3.5

    unified = ffi.new("lembed_unified_tuning_result_t *")
    unified.throughput_docs_sec = 1234.5678
    assert unified.throughput_docs_sec == pytest.approx(1234.5678, rel=1e-9)


def test_float_fields_roundtrip():
    """float fields must not be widened/narrowed by cffi."""
    sparse = ffi.new("lembed_sparse_tuning_result_t *")
    sparse.min_weight = 0.125
    assert sparse.min_weight == pytest.approx(0.125, rel=1e-6)
    assert sparse.top_k == 0
    assert sparse.storage_format == 0


def test_cache_config_defaults_match_c():
    """cache_config_default() must report exactly what the C default returns."""
    from libembedding._binding import lib
    from libembedding.cache import cache_config_default

    c_cfg = lib.lembed_cache_config_default()
    defaults = cache_config_default()
    assert defaults["capacity"] == c_cfg.capacity
    assert defaults["ttl_seconds"] == c_cfg.ttl_seconds
    assert defaults["capacity"] > 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only binding path")
def test_ffi_buffer_roundtrip_float_array():
    """ffi.buffer over a float* must produce the exact byte count used by C."""
    import struct

    values = [1.5, -2.25, 3.125]
    c_array = ffi.new("float[]", values)
    raw = bytes(ffi.buffer(c_array, 3 * 4))
    assert len(raw) == 12
    assert list(struct.unpack("3f", raw)) == values
