"""Benchmark the three embedding presets (fast / balanced / quality) on the
customers-2000000 corpus.

The presets are model selectors, not ONNX-config selectors:
  fast      -> Xenova/paraphrase-multilingual-MiniLM-L12-v2
  balanced  -> BAAI/bge-small-en-v1.5
  quality   -> BAAI/bge-base-en-v1.5

This script measures what the preset *actually* selects, so the naming
question (is "fast" really the fastest?) gets an answer instead of an
assumption.

Usage:
    python benchmarks/datas/bench_modes.py [--rows 5000] [--batch 32] [--threads 4]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "python" / "src"))

from libembedding import TextEmbedding  # noqa: E402

DATA = _REPO / "benchmarks" / "datas" / "customers-2000000.csv"

# Columns that carry real text worth embedding. Index/Customer Id are noise.
TEXT_COLUMNS = ["First Name", "Last Name", "Company", "City", "Country"]


def load_rows(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parts = [str(row.get(c, "")).strip() for c in TEXT_COLUMNS]
            text = " ".join(p for p in parts if p)
            if text:
                texts.append(text)
            if len(texts) >= limit:
                break
    return texts


def bench(mode: str, texts: list[str], batch: int, threads: int, offline: bool) -> dict:
    t0 = time.perf_counter()
    model = TextEmbedding.from_mode(
        mode,
        threads=threads,
        batch_size=batch,
        offline=offline,
        show_download_progress=False,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup (also forces the graph to compile).
    model.embed(texts[: min(64, len(texts))])

    runs = 3
    docs_total = 0
    wall_total = 0.0
    for _ in range(runs):
        t1 = time.perf_counter()
        model.embed(texts, batch_size=batch)
        t2 = time.perf_counter()
        wall_total += t2 - t1
        docs_total += len(texts)

    avg_s = wall_total / runs
    docs_per_sec = docs_total / avg_s if avg_s > 0 else 0.0
    ms_per_doc = avg_s * 1000.0 / docs_total if docs_total else 0.0

    info = model.info()
    model.close()
    return {
        "mode": mode,
        "model": info.name,
        "dim": info.dimension,
        "max_length": info.max_length,
        "quantization": info.quantization,
        "load_ms": load_ms,
        "n_docs": len(texts),
        "batch": batch,
        "threads": threads,
        "ms_per_doc": ms_per_doc,
        "docs_per_sec": docs_per_sec,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--offline", action="store_true", default=True)
    ap.add_argument("--all", action="store_true", help="run all three presets")
    ap.add_argument(
        "modes", nargs="*", choices=["fast", "balanced", "quality"], default=["balanced"]
    )
    args = ap.parse_args()

    if not DATA.exists():
        print(f"missing {DATA}", file=sys.stderr)
        return 2
    if not args.modes and not args.all:
        args.modes = ["balanced"]

    texts = load_rows(DATA, args.rows)
    print(f"loaded {len(texts)} texts from {DATA.name}")
    print(f"avg text length: {sum(len(t) for t in texts) / len(texts):.1f} chars")

    modes = ["fast", "balanced", "quality"] if args.all else args.modes
    results = [bench(m, texts, args.batch, args.threads, args.offline) for m in modes]

    print()
    print(
        f"{'mode':<10} {'model':<52} {'dim':>4} {'ctx':>5} {'q':<9} "
        f"{'load ms':>9} {'ms/doc':>8} {'docs/s':>9}"
    )
    for r in results:
        print(
            f"{r['mode']:<10} {r['model']:<52} {r['dim']:>4} {r['max_length']:>5} "
            f"{r['quantization']:<9} {r['load_ms']:>9.1f} {r['ms_per_doc']:>8.2f} "
            f"{r['docs_per_sec']:>9.1f}"
        )

    if len(results) >= 2:
        base = results[0]["docs_per_sec"]
        print()
        for r in results:
            ratio = r["docs_per_sec"] / base if base else 0.0
            print(f"{r['mode']} vs {results[0]['mode']}: {ratio:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())