# SPDX-License-Identifier: MIT
"""Download resilience: a stalled transfer is aborted and retried.

A local HTTP server stands in for HuggingFace (HF_ENDPOINT), so these tests
need no network. LIBEMBEDDING_DOWNLOAD_STALL_TIMEOUT shortens the stall
detection to one second.
"""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import libembedding

PAYLOAD = b"GGUF" + bytes(range(256)) * 16


class _Server:
    """HTTP server answering each request with the next scripted behaviour."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.requests = 0
        self.release = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                index = min(server.requests, len(server.behaviours) - 1)
                server.requests += 1
                behaviour = server.behaviours[index]
                if behaviour == "stall":
                    # Announce the body, then send nothing: the connection
                    # stays open and silent, as on the stuck CI runners.
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(PAYLOAD)))
                    self.end_headers()
                    self.wfile.flush()
                    server.release.wait(30)
                    return
                if isinstance(behaviour, int):
                    self.send_error(behaviour)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(PAYLOAD)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


# Runs in a child process: the library reads HF_ENDPOINT with getenv(), and a
# Debug build on Windows links the debug CRT, whose environment is a copy made
# when the DLL loads, so os.environ changes in this process never reach it.
_CHILD = """
import json, sys
sys.path.insert(0, sys.argv[1])
from libembedding._binding import ffi, lib
out = ffi.new("char **")
status = lib.lembed_ensure_gguf_model(
    b"org/repo", b"model.gguf", sys.argv[2].encode(), 0, 0, out)
if status == 0:
    text = ffi.string(out[0]).decode("utf-8")
    lib.lembed_free_string(out[0])
else:
    text = ffi.string(lib.lembed_last_error()).decode("utf-8", "replace")
print(json.dumps([status, text]))
"""


def _ensure(server, cache_dir):
    """Download org/repo/model.gguf from server through lembed_ensure_gguf_model."""
    env = dict(os.environ)
    env.update(
        HF_ENDPOINT=server.url,
        LIBEMBEDDING_DOWNLOAD_STALL_TIMEOUT="1",
        NO_PROXY="127.0.0.1",
    )
    package_root = str(Path(libembedding.__file__).resolve().parent.parent)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, package_root, str(cache_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    status, text = json.loads(proc.stdout.strip().splitlines()[-1])
    return status, text


def test_stalled_transfer_is_retried(tmp_path):
    with _Server(["stall", "ok"]) as server:
        started = time.monotonic()
        status, result = _ensure(server, tmp_path)
        elapsed = time.monotonic() - started
    assert status == 0, result
    with open(result, "rb") as f:
        assert f.read() == PAYLOAD
    assert server.requests == 2
    assert elapsed < 20


def test_server_error_is_retried_then_reported(tmp_path):
    with _Server([503]) as server:
        status, message = _ensure(server, tmp_path)
    assert status != 0
    assert server.requests == 3
    assert "after 3 attempts" in message
    assert not list(tmp_path.rglob("model.gguf*"))


def test_missing_file_is_not_retried(tmp_path):
    with _Server([404]) as server:
        status, message = _ensure(server, tmp_path)
    assert status != 0
    assert server.requests == 1
    assert "404" in message
