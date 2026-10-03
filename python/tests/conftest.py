"""Shared pytest configuration for libembedding Python tests."""

import os
import platform
import sys
from pathlib import Path

import pytest

# Ensure the src layout is importable when running tests from the repo root.
# conftest.py lives in <repo>/python/tests, so the repository root is three
# levels up (tests -> python -> repo).
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "python" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BGE_SMALL_MODEL = "BAAI/bge-small-en-v1.5"

# Fixtures that build a model, i.e. that need the model cache or the network.
# pytest ignores marks applied to a fixture, so the `network` marker is added at
# collection time from these names (see pytest_collection_modifyitems).
NETWORK_FIXTURES = frozenset({"bge_small", "image_model", "sparse_model"})


def pytest_collection_modifyitems(items):
    """Mark every test requesting a model fixture with `network`."""
    for item in items:
        if "network" in item.keywords:
            continue
        if NETWORK_FIXTURES.intersection(getattr(item, "fixturenames", ())):
            item.add_marker("network")


@pytest.fixture
def bge_small():
    """Factory building a BGE-small TextEmbedding, skipping if unavailable.

    Usage:
        def test_something(bge_small):
            model = bge_small(cache_size=16)
    """

    def _factory(**kwargs):
        from libembedding import TextEmbedding
        from libembedding.exceptions import DownloadError

        kwargs.setdefault("show_download_progress", False)
        try:
            return TextEmbedding(BGE_SMALL_MODEL, **kwargs)
        except DownloadError:
            pytest.skip("model download unavailable (network restriction in CI)")

    return _factory


def skip_unless_cached(repo_code: str) -> None:
    """Skip the calling test unless *repo_code* is already in the local cache.

    The autotuner builds its benchmark sessions with ``offline = 1``
    (``detail/autotune_bench_text.hpp``): it measures a model, it never fetches
    one. A fresh CI runner therefore has nothing to tune, and the tuner answers
    ``Model file not in cache (offline mode)`` -- which is a missing
    precondition rather than a regression.

    The check goes through the library instead of re-deriving the on-disk cache
    layout, so it follows the registry mapping too (the tuner is handed
    ``BAAI/bge-small-en-v1.5`` but the weights live under ``Xenova/``).
    """
    from libembedding._binding import ffi, lib
    from libembedding.models import resolve_text_model

    index = resolve_text_model(repo_code)
    model_dir = ffi.new("char **")
    status = lib.lembed_ensure_text_model(index, ffi.NULL, 0, 1, model_dir)
    if model_dir[0] != ffi.NULL:
        lib.lembed_free_string(model_dir[0])
    if status != lib.LEMBED_OK:
        pytest.skip(
            f"{repo_code} is not in the local cache and the autotuner "
            "does not download (offline benchmarking)"
        )


@pytest.fixture
def require_cached_model():
    """Expose :func:`skip_unless_cached` to test modules as a fixture."""

    def _require(repo_code: str) -> None:
        skip_unless_cached(repo_code)

    return _require


# Locate the built shared library and ensure its directory is on PATH
# so Windows can resolve the runtime DLLs (onnxruntime, libcurl, MSVC runtime).
# The import is deliberately placed here: it needs sys.path to be patched first.
from libembedding._binding import _find_library

_lib_path = _find_library()
if os.path.exists(_lib_path):
    _lib_dir = str(Path(_lib_path).parent)
    if _lib_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _lib_dir + os.pathsep + os.environ.get("PATH", "")
else:
    # Fallback candidates (in case _binding.py doesn't know about them).
    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(
        platform.system(), ".so"
    )
    for candidate in [
        ROOT / "build" / "python" / f"libembedding{ext}",
        ROOT / "build" / f"libembedding{ext}",
        ROOT / "build" / "Release" / f"libembedding{ext}",
        ROOT / "build3" / "Release" / f"libembedding{ext}",
        ROOT / "build3" / f"libembedding{ext}",
    ]:
        if candidate.exists():
            _lib_dir = str(candidate.parent)
            if _lib_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _lib_dir + os.pathsep + os.environ.get("PATH", "")
            break
