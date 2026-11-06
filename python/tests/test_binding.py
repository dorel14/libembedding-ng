"""Tests for the shared-library lookup in _binding.

Auteur: David Orel

SPDX-License-Identifier: MIT
"""

import os
import warnings

import pytest

from libembedding._binding import (
    _DEPENDENT_DLLS,
    _find_library,
    _pick_library,
    _runtime_dirs,
    _select_dependencies,
)


def _touch(path, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_library_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "libembedding-custom.bin"
    fake.write_bytes(b"")
    monkeypatch.setenv("LIBEMBEDDING_LIBRARY", str(fake))
    assert _find_library() == str(fake.resolve())


def test_library_override_missing_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBEMBEDDING_LIBRARY", str(tmp_path / "absent.dll"))
    with pytest.raises(OSError, match="LIBEMBEDDING_LIBRARY"):
        _find_library()


def test_library_lookup_without_override(monkeypatch):
    monkeypatch.delenv("LIBEMBEDDING_LIBRARY", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a stale local copy is not this test's subject
        assert "libembedding" in _find_library()


def test_pick_library_prefers_the_package_copy(tmp_path):
    packaged = _touch(tmp_path / "pkg" / "libembedding.dll", mtime=2_000)
    build = _touch(tmp_path / "build" / "Release" / "libembedding.dll", mtime=1_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _pick_library([packaged, build], packaged_count=1) == packaged


def test_pick_library_warns_on_a_stale_package_copy(tmp_path):
    packaged = _touch(tmp_path / "pkg" / "libembedding.dll", mtime=1_000)
    build = _touch(tmp_path / "build" / "Release" / "libembedding.dll", mtime=2_000)
    with pytest.warns(UserWarning, match="older than the build output"):
        assert _pick_library([packaged, build], packaged_count=1) == packaged


def test_pick_library_does_not_compare_build_trees(tmp_path):
    debug = _touch(tmp_path / "build" / "Debug" / "libembedding.dll", mtime=1_000)
    release = _touch(tmp_path / "build" / "Release" / "libembedding.dll", mtime=2_000)
    absent = tmp_path / "pkg" / "libembedding.dll"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _pick_library([absent, debug, release], packaged_count=1) == debug


def test_pick_library_without_any_candidate(tmp_path):
    assert _pick_library([tmp_path / "libembedding.dll"], packaged_count=1) is None


def test_runtime_dirs_start_with_the_library_dir_and_skip_other_builds(tmp_path):
    lib = _touch(tmp_path / "custom" / "libembedding.dll")
    dirs = _runtime_dirs(lib)
    assert dirs[0] == lib.parent
    assert not any(d.parts[-2:] in (("bin", "Debug"), ("bin", "Release")) for d in dirs)


def test_ggml_cpu_features_names_the_loaded_backend():
    # Loading from the library's own directory is the path a wheel takes: ggml's
    # default search would look next to python.exe and find nothing.
    from libembedding import ggml_cpu_features

    features = ggml_cpu_features()
    assert features, "no ggml CPU backend was loaded next to libembedding"
    assert all(f and f == f.upper() for f in features)
    assert ggml_cpu_features() == features


def test_ggml_cpu_backend_is_not_preloaded():
    # ggml scores and loads its own CPU variant; preloading one would choose it.
    assert not any(name.startswith("ggml-cpu") for name in _DEPENDENT_DLLS)


def test_select_dependencies_loads_one_copy_per_name(tmp_path):
    own, fallback = tmp_path / "own", tmp_path / "fallback"
    for name in ("ggml.dll", "ggml-base.dll"):
        _touch(own / name)
        _touch(fallback / name)
    _touch(fallback / "libcurl-x64.dll")

    selected = _select_dependencies([own, fallback])

    assert own / "ggml.dll" in selected
    assert own / "ggml-base.dll" in selected
    assert fallback / "libcurl-x64.dll" in selected  # missing next to the library
    assert len(selected) == len({p.name for p in selected}) == 3
    assert all(p.name in _DEPENDENT_DLLS for p in selected)
