"""Tests for the shared-library lookup in _binding.

Auteur: David Orel

SPDX-License-Identifier: MIT
"""

import pytest

from libembedding._binding import _find_library


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
    assert "libembedding" in _find_library()
