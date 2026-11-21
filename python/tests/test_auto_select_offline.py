# SPDX-License-Identifier: MIT
"""auto_select_model() on an empty model cache leaves nothing behind, no network.

Auteur: David Orel

The selection benchmarks its candidates with sessions created offline: an
uncached candidate is skipped, never downloaded. The downloader used to create
the candidate's ``models--<repo>`` directory before checking the offline flag,
leaving one empty directory per skipped candidate. It must now create none.

The selection runs in a child process whose model cache and autotune cache
both point into ``tmp_path``: no tuning cached on the developer's machine can
make a candidate win without being loaded, so every candidate is skipped and
the call ends in ``ModelNotFoundError`` whatever is installed locally.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import libembedding

PRE_EXISTING = "models--org-other"

_CHILD = """
import json
from libembedding import auto_select_model
from libembedding.exceptions import ModelNotFoundError
try:
    auto_select_model("balanced")
    print(json.dumps({"status": "ok"}))
except ModelNotFoundError:
    print(json.dumps({"status": "not_found"}))
"""


@pytest.mark.skipif(not sys.executable, reason="needs a child Python process")
def test_auto_select_on_empty_cache_creates_no_model_directory(tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    # A model already cached, an incomplete one: both must survive untouched.
    (cache / PRE_EXISTING).mkdir()
    (cache / PRE_EXISTING / "model.onnx").write_bytes(b"x" * 30)

    env = {k: v for k, v in os.environ.items() if k != "LIBEMBEDDING_VERBOSE"}
    env["LIBEMBEDDING_CACHE_DIR"] = str(cache)
    # Autotune cache: LOCALAPPDATA on Windows, HOME elsewhere.
    env["LOCALAPPDATA"] = str(tmp_path / "appdata")
    env["HOME"] = str(tmp_path / "home")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(libembedding.__file__).resolve().parent.parent)] + [p for p in sys.path if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["status"] == "not_found"
    assert {p.name for p in cache.iterdir()} == {PRE_EXISTING}
    assert (cache / PRE_EXISTING / "model.onnx").stat().st_size == 30
