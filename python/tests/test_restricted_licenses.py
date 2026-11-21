# SPDX-License-Identifier: MIT
"""Registry models under a licence that is not MIT-compatible (decision D4).

jinaai/jina-reranker-v2-base-multilingual (CC-BY-NC-4.0) and
google/embeddinggemma-300m (Gemma Terms of Use) stay in the enums for ABI
reasons but are refused unless LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1.
No network: the refusal happens before any download, and the opt-in case runs
offline with an empty cache, so it fails later, on the download.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import libembedding
from libembedding._binding import ffi, lib
from libembedding.models import resolve_reranker_model, resolve_text_model

CASES = [
    ("reranker", "jinaai/jina-reranker-v2-base-multilingual", "CC-BY-NC-4.0"),
    ("text", "google/embeddinggemma-300m", "Gemma Terms of Use"),
]


def _ensure(kind, name, cache_dir):
    out = ffi.new("char **")
    if kind == "reranker":
        status = lib.lembed_ensure_reranker_model(
            resolve_reranker_model(name), str(cache_dir).encode(), 0, 1, out)
    else:
        status = lib.lembed_ensure_text_model(
            resolve_text_model(name), str(cache_dir).encode(), 0, 1, out)
    message = ffi.string(lib.lembed_last_error()).decode("utf-8", "replace")
    return status, message


@pytest.mark.parametrize("kind,name,licence", CASES)
def test_refused_by_default(kind, name, licence, tmp_path, monkeypatch):
    monkeypatch.delenv("LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES", raising=False)
    status, message = _ensure(kind, name, tmp_path)
    assert status == lib.LEMBED_ERROR_UNSUPPORTED
    assert licence in message
    assert "LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1" in message
    assert not any(tmp_path.iterdir())  # nothing downloaded


_CHILD = """
import json, sys
from libembedding._binding import ffi, lib
from libembedding.models import resolve_reranker_model
out = ffi.new("char **")
status = lib.lembed_ensure_reranker_model(
    resolve_reranker_model(sys.argv[1]), sys.argv[2].encode(), 0, 1, out)
print(json.dumps([status, ffi.string(lib.lembed_last_error()).decode("utf-8", "replace")]))
"""


@pytest.mark.skipif(not sys.executable, reason="needs a child Python process")
def test_opt_in_lets_it_through(tmp_path):
    # Child process: a Windows Debug DLL does not see os.environ changes made
    # after it loaded. Offline with an empty cache, the opt-in turns the
    # licence refusal into the ordinary "not in cache" download error.
    env = dict(os.environ, LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(libembedding.__file__).resolve().parent.parent)] + [p for p in sys.path if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, CASES[0][1], str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    status, message = json.loads(proc.stdout.strip().splitlines()[-1])
    assert status == lib.LEMBED_ERROR_DOWNLOAD
    assert "CC-BY-NC" not in message
