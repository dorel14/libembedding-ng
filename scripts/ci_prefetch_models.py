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


def main() -> int:
    models = [
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
