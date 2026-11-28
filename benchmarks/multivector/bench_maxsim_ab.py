"""A/B benchmark of lembed maxsim against a numpy loop, one DLL per process.

Usage (PowerShell, repo-root .venv):
    python bench_maxsim_ab.py OLD_DLL NEW_DLL [rounds]

Each round runs the old then the new DLL in its own process (the library is
chosen with LIBEMBEDDING_LIBRARY); inside a process every metric is the best of
5 timed runs and numpy is measured in the same process, so the ratio to numpy
cancels most of the machine load. Prints per-metric medians over the rounds.

Workload: one query of 32x96 against 1000 documents of 200-270 vectors.

Auteur: David Orel
Version: 1.16.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

CHILD = "--child"
NQ, DIM, NDOCS = 32, 96, 1000


def child(has_ragged: bool) -> None:
    import numpy as np

    from libembedding.similarity import maxsim, maxsim_batch

    rng = np.random.default_rng(1)
    q = rng.standard_normal((NQ, DIM)).astype(np.float32)
    docs = [
        rng.standard_normal((int(rng.integers(200, 270)), DIM)).astype(np.float32)
        for _ in range(NDOCS)
    ]
    data = np.concatenate(docs)
    offs = np.zeros(NDOCS + 1, dtype=np.int64)
    np.cumsum([len(d) for d in docs], out=offs[1:])

    def rate(fn) -> float:
        fn()
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return NDOCS / best

    res = {
        "per_doc": rate(lambda: [maxsim(q, d) for d in docs]),
        "batch_list": rate(lambda: maxsim_batch(q, docs)),
        "numpy": rate(lambda: [(q @ d.T).max(1).sum() for d in docs]),
    }
    if has_ragged:
        res["batch_ragged"] = rate(lambda: maxsim_batch(q, (data, offs)))
    print(json.dumps(res))


def run(dll: str, has_ragged: bool) -> dict:
    env = dict(os.environ, LIBEMBEDDING_LIBRARY=dll)
    out = subprocess.run(
        [sys.executable, __file__, CHILD, "ragged" if has_ragged else "plain"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == CHILD:
        child(sys.argv[2] == "ragged")
        return
    old_dll, new_dll = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    old_runs, new_runs = [], []
    for _ in range(rounds):
        old_runs.append(run(old_dll, False))
        new_runs.append(run(new_dll, True))

    def med(runs: list[dict], key: str) -> float:
        return statistics.median(r[key] for r in runs if key in r)

    print(f"{'metric':<14}{'old docs/s':>12}{'new docs/s':>12}{'new/old':>9}{'old/numpy':>11}{'new/numpy':>11}")
    for key in ("per_doc", "batch_list", "batch_ragged"):
        o = med(old_runs, key) if key != "batch_ragged" else float("nan")
        n = med(new_runs, key)
        ro = statistics.median(r[key] / r["numpy"] for r in old_runs if key in r) if key != "batch_ragged" else float("nan")
        rn = statistics.median(r[key] / r["numpy"] for r in new_runs)
        print(f"{key:<14}{o:>12.0f}{n:>12.0f}{n / o:>9.2f}{ro:>11.2f}{rn:>11.2f}")
    print(f"numpy docs/s: old-process {med(old_runs, 'numpy'):.0f}, new-process {med(new_runs, 'numpy'):.0f}")


if __name__ == "__main__":
    main()
