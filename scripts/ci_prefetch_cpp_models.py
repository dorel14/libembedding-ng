#!/usr/bin/env python3
# Auteur: David Orel
# SPDX-License-Identifier: MIT
"""Best-effort prefetch of the models the C++ unit tests load from the cache.

The sanitizer jobs build libembedding header-only, so they have no Python
binding of their own. This script runs on the released wheel
(pip install libembedding-ng) and calls the library's own downloader, which
writes the exact cache layout the C++ tests look up offline:

- test_text_max_length, parts B and C: all-MiniLM-L6-v2 (ONNX);
- test_backend_introspection, cache part: bge-small-en-v1.5 (ONNX, also the
  default text model), Snowflake-XS-Q4 (GGUF) and the Jina reranker v1 turbo
  (GGUF);
- test_multivector_fidelity: answerai-colbert-small-v1. The 1.16 wheel has no
  multivector downloader, so its three files are fetched straight from
  Hugging Face into the directory the test reads.

A model already in the cache is a fast no-op. The script never fails the job:
a failed model is a warning. When every model is in place it writes
<cache>/.cpp-prefetch-ok, which the workflow uses to decide whether a skipped
test is an error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

ATTEMPTS = 3
DELAY_SECONDS = 10

_COLBERT_REPO = "answerdotai/answerai-colbert-small-v1"
_COLBERT_FILES = ("vespa_colbert.onnx", "tokenizer.json", "onnx_config.json")

# (repo, filename) as in the GGUF registry (gguf_registry_impl.hpp).
_GGUF = (
    (
        "ChristianAzinn/snowflake-arctic-embed-xs-gguf",
        "snowflake-arctic-embed-xs--Q4_K_M.GGUF",
    ),
    (
        "sinjab/jina-reranker-v1-turbo-en-Q8_0-GGUF",
        "jina-reranker-v1-turbo-en-Q8_0.gguf",
    ),
)


def _check(lib, ffi, status: int, out) -> None:
    if status != lib.LEMBED_OK:
        raise RuntimeError(
            ffi.string(lib.lembed_last_error()).decode("utf-8", "replace")
        )
    if out[0] != ffi.NULL:
        lib.lembed_free_string(out[0])


def _text(model_attr: str):
    def run(lib, ffi, cache: bytes) -> None:
        model = getattr(lib, model_attr)
        out = ffi.new("char**")
        _check(lib, ffi, lib.lembed_ensure_text_model(model, cache, 0, 0, out), out)

    return run


def _gguf(repo: str, filename: str):
    def run(lib, ffi, cache: bytes) -> None:
        out = ffi.new("char**")
        status = lib.lembed_ensure_gguf_model(
            repo.encode("utf-8"), filename.encode("utf-8"), cache, 0, 0, out
        )
        _check(lib, ffi, status, out)

    return run


def _colbert(cache_dir: Path):
    def run(lib, ffi, cache: bytes) -> None:
        del lib, ffi, cache
        dest = cache_dir / ("models--" + _COLBERT_REPO.replace("/", "-"))
        dest.mkdir(parents=True, exist_ok=True)
        for name in _COLBERT_FILES:
            target = dest / name
            if target.is_file() and target.stat().st_size > 0:
                continue
            part = target.with_name(name + ".part")
            url = f"https://huggingface.co/{_COLBERT_REPO}/resolve/main/{name}"
            with (
                urllib.request.urlopen(url, timeout=60) as src,
                open(part, "wb") as dst,
            ):
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    dst.write(block)
            part.replace(target)

    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("LIBEMBEDDING_CACHE_DIR")
        or str(Path.home() / ".cache" / "libembedding"),
        help="model cache root (default: $LIBEMBEDDING_CACHE_DIR or ~/.cache/libembedding)",
    )
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / ".cpp-prefetch-ok"
    if marker.exists():
        marker.unlink()

    from libembedding import ffi, lib

    cache = str(cache_dir).encode("utf-8")
    models = [
        ("all-MiniLM-L6-v2", _text("LEMBED_TEXT_ALL_MINILM_L6_V2")),
        (
            "bge-small-en-v1.5 (default text model)",
            _text("LEMBED_TEXT_BGE_SMALL_EN_V15"),
        ),
        *((f"GGUF {name}", _gguf(repo, name)) for repo, name in _GGUF),
        ("ColBERT answerai-colbert-small-v1", _colbert(cache_dir)),
    ]
    failed = 0
    for label, run in models:
        for attempt in range(1, ATTEMPTS + 1):
            start = time.time()
            try:
                run(lib, ffi, cache)
            except Exception as exc:  # noqa: BLE001 - best effort, report and go on
                print(
                    f"prefetch {label}: attempt {attempt}/{ATTEMPTS} failed "
                    f"after {time.time() - start:.0f}s: {exc}",
                    flush=True,
                )
                if attempt < ATTEMPTS:
                    time.sleep(DELAY_SECONDS)
                continue
            print(f"prefetch {label}: ok in {time.time() - start:.0f}s", flush=True)
            break
        else:
            failed += 1
            print(
                f"::warning::prefetch {label} failed, its tests will skip", flush=True
            )
    if failed == 0:
        marker.write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
