# SPDX-License-Identifier: MIT
"""ONNX Runtime telemetry stays off unless the user asks for it.

Since 1.31 the official Linux and macOS builds of ONNX Runtime report to
Microsoft by default, and store a device id under
Microsoft/DeveloperTools/.onnxruntime in the user's cache directory as soon as
their telemetry starts. libembedding sets ORT_DISABLE_TELEMETRY=1 before its
first ORT call (detail/onnx_session_impl.hpp), so that directory must never
appear.

Each case runs in a fresh process with a scratch HOME: ORT reads the variable
once per process. ORT also mutes its telemetry on CI machines, which would make
the check pass vacuously, so the CI variables are removed from the child's
environment. The control case sets ORT_DISABLE_TELEMETRY=0, the documented way
to keep ORT's telemetry, and expects the directory: it proves the check sees
telemetry when there is some. Its uploads go to a closed local proxy port.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import libembedding

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="ORT's Windows telemetry is ETW and stores nothing to check",
    ),
    pytest.mark.skipif(not sys.executable, reason="needs a child Python process"),
]

# ORT 1.31.0, onnxruntime/core/platform/telemetry_environment.h.
_CI_VARIABLES = (
    "CI", "TF_BUILD", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "TRAVIS",
    "JENKINS_URL", "CODEBUILD_BUILD_ID", "BUILDKITE", "TEAMCITY_VERSION",
    "APPVEYOR", "BITBUCKET_BUILD_NUMBER", "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
    "ORT_RUNNING_UNIT_TESTS", "ORT_DISABLE_TELEMETRY",
)

_CHILD = """
import sys
sys.path.insert(0, sys.argv[1])
from libembedding import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=sys.argv[2])
model.embed(["hello"])
model.close()
"""


def _model_cache():
    for name in ("LIBEMBEDDING_CACHE_DIR", "FASTEMBED_CACHE_DIR"):
        if os.environ.get(name):
            return os.environ[name]
    return str(Path.home() / ".cache" / "libembedding")


def _load_model(home, **extra_env):
    env = {k: v for k, v in os.environ.items() if k not in _CI_VARIABLES}
    env.update(HOME=str(home), XDG_CACHE_HOME=str(home / "cache"), **extra_env)
    package_root = str(Path(libembedding.__file__).resolve().parent.parent)
    subprocess.run(
        [sys.executable, "-c", _CHILD, package_root, _model_cache()],
        env=env,
        check=True,
        timeout=300,
    )
    return sorted(str(p) for p in home.rglob(".onnxruntime"))


def test_telemetry_is_off_by_default(tmp_path):
    assert _load_model(tmp_path) == []


def test_user_can_keep_ort_telemetry(tmp_path):
    closed = "http://127.0.0.1:9"
    found = _load_model(
        tmp_path,
        ORT_DISABLE_TELEMETRY="0",
        https_proxy=closed,
        HTTPS_PROXY=closed,
        no_proxy="",
        NO_PROXY="",
    )
    if not found:
        pytest.skip("this ONNX Runtime build has no telemetry")
    assert found
