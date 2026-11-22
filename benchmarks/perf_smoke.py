"""Performance smoke test for shared CI runners (T16, decision D7).

Auteur: David Orel
SPDX-License-Identifier: MIT

GitHub-hosted runners are shared machines: their absolute speed moves by tens
of percent from one run to the next, so an absolute docs/s figure cannot catch
a regression there. This script only reports quantities that cancel most of
the machine out:

* ``dense_b32_per_gflop`` / ``dense_b1_per_gflop``: embedding throughput
  divided by a reference float32 matrix product measured in the same process,
  just before. A slower runner slows both down.
* ``batch_gain``: throughput at batch 32 over throughput at batch 1. Pure
  ratio, no machine term at all. A drop means batching stopped paying.

Each figure is compared with ``perf_smoke_baseline.json`` (key = platform
name). A change beyond ``--threshold`` percent (default 25, the noise seen on
shared runners) is reported as a GitHub ``::warning::``. The script never
fails on a slowdown: a controlled runner is deferred (D7), so this job warns
and leaves the judgement to a human. It fails only when it cannot measure.

Usage::

    python benchmarks/perf_smoke.py --platform linux-x86_64 --out perf.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _benchmeta as bm

MODEL = "BAAI/bge-small-en-v1.5"
BASELINE_FILE = Path(__file__).resolve().parent / "perf_smoke_baseline.json"
# Higher is better for every metric: a regression is a drop.
METRICS = ("dense_b32_per_gflop", "dense_b1_per_gflop", "batch_gain")


def reference_gflops(rounds: int) -> float:
    """Median GFLOP/s of a 512x512 float32 matrix product (numpy BLAS)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((512, 512), dtype=np.float32)
    b = rng.standard_normal((512, 512), dtype=np.float32)
    c = np.empty_like(a)
    np.matmul(a, b, out=c)  # warmup
    flops = 2.0 * 512**3
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(20):
            np.matmul(a, b, out=c)
        samples.append(20 * flops / (time.perf_counter() - start) / 1e9)
    return statistics.median(samples)


def corpus(n: int) -> list[str]:
    """Deterministic sentences of 12 to 40 words."""
    words = ["vector", "embedding", "model", "runtime", "token", "batch", "latency",
             "cache", "query", "document", "search", "sparse", "dense", "ranking",
             "score", "memory", "thread"]
    out = []
    for i in range(n):
        length = 12 + (i * 7) % 29
        out.append(" ".join(words[(i + k) % len(words)] for k in range(length)))
    return out


def throughput(model, texts: list[str], batch: int, rounds: int) -> float:
    """Median docs/s over ``rounds`` timed passes, after one warmup pass."""
    model.embed(texts, batch_size=batch)
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        model.embed(texts, batch_size=batch)
        samples.append(len(texts) / (time.perf_counter() - start))
    return statistics.median(samples)


def compare(platform: str, results: dict[str, float], threshold: float) -> list[str]:
    """Warning lines for every metric that moved beyond the threshold."""
    try:
        baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8")).get(platform, {})
    except (OSError, ValueError):
        baseline = {}
    if not baseline:
        print(f"No baseline for {platform}: record these values in {BASELINE_FILE.name}.")
        return []
    warnings = []
    for name in METRICS:
        ref = baseline.get(name)
        if not ref:
            continue
        delta = bm.relative_delta_pct(ref, results[name])
        print(f"{name}: {results[name]:.4g} (baseline {ref:.4g}, {delta:+.1f} %)")
        if delta < -threshold:
            warnings.append(f"{name} dropped {delta:.1f} % against the baseline "
                            f"({results[name]:.4g} vs {ref:.4g}, threshold {threshold:g} %)")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", required=True, help="baseline key, e.g. linux-x86_64")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=25.0)
    args = parser.parse_args()

    from libembedding import TextEmbedding

    model, load_ms = bm.timed_load(
        lambda: TextEmbedding(MODEL, threads=0, show_download_progress=False))
    texts = corpus(128)
    gflops_before = reference_gflops(args.rounds)
    b32 = throughput(model, texts, 32, args.rounds)
    b1 = throughput(model, texts[:32], 1, args.rounds)
    gflops_after = reference_gflops(args.rounds)
    # The reference brackets the embedding runs: averaging both sides absorbs
    # a runner whose speed drifts during the job.
    gflops = (gflops_before + gflops_after) / 2.0

    results = {
        "dense_b32_per_gflop": b32 / gflops,
        "dense_b1_per_gflop": b1 / gflops,
        "batch_gain": b32 / b1,
        "raw": {"docs_per_s_b32": b32, "docs_per_s_b1": b1, "ref_gflops": gflops,
                "ref_gflops_spread_pct": bm.relative_delta_pct(gflops_before, gflops_after),
                "load_ms": load_ms, "peak_rss_mb": bm.peak_rss_mb()},
    }
    run = bm.RunParams(threads=0, batch=[1, 32], warmup=1, iterations=args.rounds,
                       format="onnx", quantization="fp32", dataset="synthetic-128")
    report = bm.build_report(bm.collect_machine_metadata(), run, results,
                             title=f"perf smoke {args.platform} {MODEL}")
    if args.out:
        bm.write_json(report, args.out)
    print(json.dumps({k: results[k] for k in METRICS}, indent=2))

    for line in compare(args.platform, results, args.threshold):
        print(f"::warning title=perf smoke ({args.platform})::{line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
