"""Competitive dense benchmark: libembedding vs SentenceTransformers vs FastEmbed (T15).

Auteur: David Orel
SPDX-License-Identifier: MIT

One measurement per process (peak RSS is a per-process high-water mark): run
this script once per (library, model) pair, alternating the order between
rounds, and keep the median. Install the three libraries in a dedicated
virtual environment; the models must already be in each library's cache so
that no download is timed.

Usage::

    python benchmarks/competitive/bench_competitive.py <lib> <model> <corpus>

lib: libembedding | sentence-transformers | fastembed
Prints: docs_per_s load_ms peak_rss_mb
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _benchmeta as bm

N_TEXTS = 1000
BATCH = 32
WARMUP_TEXTS = 50
ROUNDS = 3


def load(lib: str, model: str):
    """Return a callable embedding a list of texts with batch size BATCH."""
    if lib == "libembedding":
        from libembedding import TextEmbedding

        m = TextEmbedding(model, show_download_progress=False)
        return lambda xs: m.embed(xs, batch_size=BATCH)
    if lib == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model, device="cpu")
        return lambda xs: st.encode(xs, batch_size=BATCH)
    if lib == "fastembed":
        from fastembed import TextEmbedding as FastEmbedText

        fe = FastEmbedText(model)
        return lambda xs: list(fe.embed(xs, batch_size=BATCH))
    raise SystemExit(f"unknown library: {lib}")


def main() -> int:
    lib, model, corpus = sys.argv[1], sys.argv[2], sys.argv[3]
    text = Path(corpus).read_text(encoding="utf-8")
    texts = [t for t in text.splitlines() if t.strip()][:N_TEXTS]

    run, load_ms = bm.timed_load(lambda: load(lib, model))
    run(texts[:WARMUP_TEXTS])
    samples = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        run(texts)
        samples.append(len(texts) / (time.perf_counter() - start))
    print(f"{statistics.median(samples):.2f} {load_ms:.1f} {bm.peak_rss_mb():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
