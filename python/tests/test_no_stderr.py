# SPDX-License-Identifier: MIT
"""The library writes nothing on stderr unless LIBEMBEDDING_VERBOSE asks.

Diagnostic lines (autotune progress, auto model selection, cache cleanup in
dry-run mode) go through detail/log.hpp. The cache cleanup dry run exercises
that path without any network: its counts come back through the return value,
and the per-directory list appears only in verbose mode.

Both cases run in a child process whose stderr is a pipe from the start:
LIBEMBEDDING_VERBOSE is read once per process, and on Windows a library built
against another C runtime (a Debug build) writes to its own copy of fd 2,
which pytest's capfd redirection never sees.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import libembedding

pytestmark = pytest.mark.skipif(not sys.executable, reason="needs a child Python process")

_CHILD = """
import json, sys
from libembedding import cleanup_model_cache
result = cleanup_model_cache([], cache_dir=sys.argv[1], dry_run=True)
print(json.dumps(result["deleted_count"]))
"""


def _fake_cache(root: Path) -> Path:
    for name in ("models--org--a", "models--org--b"):
        (root / name).mkdir(parents=True)
        (root / name / "model.onnx").write_bytes(b"\0" * 16)
    return root


def _dry_run(cache: Path, **extra_env):
    env = {k: v for k, v in os.environ.items() if k != "LIBEMBEDDING_VERBOSE"}
    env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(libembedding.__file__).resolve().parent.parent)] + [p for p in sys.path if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(cache)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def test_dry_run_is_silent(tmp_path):
    cache = _fake_cache(tmp_path)
    deleted, err = _dry_run(cache)
    assert deleted == 2
    assert err == ""
    assert sorted(p.name for p in cache.iterdir()) == ["models--org--a", "models--org--b"]


def test_verbose_lists_what_would_be_deleted(tmp_path):
    cache = _fake_cache(tmp_path)
    deleted, err = _dry_run(cache, LIBEMBEDDING_VERBOSE="1")
    assert deleted == 2
    assert "[cleanup] Would delete:" in err
    assert "models--org--a" in err
    assert "[cleanup] Total: 2 directories" in err
