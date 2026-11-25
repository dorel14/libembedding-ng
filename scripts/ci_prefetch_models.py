#!/usr/bin/env python3
# Auteur: David Orel
# SPDX-License-Identifier: MIT
"""Best-effort prefetch of the models the Python tests download.

CI runs this before pytest so that a slow or flaky Hugging Face download is
retried here, with a clear log line, instead of timing a test out. A model
already in the cache (LIBEMBEDDING_CACHE_DIR) is a fast no-op. The script never
fails the job: the tests skip on DownloadError, and a missing model is only a
warning.
"""

from __future__ import annotations

import sys
import time

ATTEMPTS = 3
DELAY_SECONDS = 10


def _text(name: str):
    from libembedding import TextEmbedding

    return lambda: TextEmbedding(name, show_download_progress=False)


def _sparse(name: str):
    from libembedding import SparseTextEmbedding

    return lambda: SparseTextEmbedding(name, show_download_progress=False)


def _reranker(name: str):
    from libembedding import Reranker

    return lambda: Reranker(name, show_download_progress=False)


def _colbert(name: str):
    from libembedding import MultiVectorEmbedding

    return lambda: MultiVectorEmbedding(name, show_download_progress=False)


_GGUF_BASE = (
    "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/"
)


def _gguf(filename: str):
    """Plain download into <cache>/gguf, where test_gguf_hangul.py looks."""

    def build():
        import os
        import urllib.request
        from pathlib import Path

        root = os.environ.get("LIBEMBEDDING_CACHE_DIR") or str(
            Path.home() / ".cache" / "libembedding"
        )
        dest = Path(root) / "gguf" / filename
        if dest.is_file():
            return _Closable()
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(_GGUF_BASE + filename, timeout=60) as src:
            with open(part, "wb") as out:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
        part.replace(dest)
        return _Closable()

    return build


class _Closable:
    def close(self) -> None:
        pass


def main() -> int:
    models = [
        ("MiniLM GGUF Q8_0", _gguf("all-MiniLM-L6-v2-Q8_0.gguf")),
        ("MiniLM GGUF Q4_K_M", _gguf("all-MiniLM-L6-v2-Q4_K_M.gguf")),
        ("MiniLM-L6", _text("sentence-transformers/all-MiniLM-L6-v2")),
        ("bge-small", _text("BAAI/bge-small-en-v1.5")),
        ("SPLADE", _sparse("prithivida/Splade_PP_en_v1")),
        ("reranker", _reranker("jinaai/jina-reranker-v1-turbo-en-quantized")),
        ("ColBERT", _colbert("answerdotai/answerai-colbert-small-v1")),
    ]
    for label, build in models:
        for attempt in range(1, ATTEMPTS + 1):
            start = time.time()
            try:
                build().close()
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
            print(f"::warning::prefetch {label} failed, its tests may skip", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
