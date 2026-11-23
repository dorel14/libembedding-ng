"""Multi-vector (ColBERT, late interaction) benchmark: libembedding vs pylate.

Auteur: David Orel
SPDX-License-Identifier: MIT

Compares ``libembedding.MultiVectorEmbedding`` (ONNX Runtime) with pylate (PyTorch,
the reference implementation) on the same model, answerai-colbert-small-v1, with the
same lengths (query 32, document 300) and the same corpora, plus the MaxSim kernel
(``libembedding.maxsim`` / ``maxsim_batch``) against a numpy reference.

One measurement per process (peak RSS is a per-process high-water mark): warmup in
the process, then ``--runs`` timed passes (median). ``full`` runs the campaign with
one subprocess per cell, rotates the library order between rounds, rewrites the JSON
after every cell and skips cells already present when restarted.

To benchmark a source checkout set ``LIBEMBEDDING_LIBRARY`` to the built library and
add ``python/src`` to ``PYTHONPATH`` (MultiVectorEmbedding is not in the PyPI wheel
yet)::

    python bench_multivector.py full   [--out results.json] [options]
    python bench_multivector.py report [--json results.json] [--md results_tables.md]
    python bench_multivector.py fidelity
    python bench_multivector.py worker --kind embed --lib libembedding --corpus long --batch 32
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _benchmeta as bm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SHORT_CORPUS = REPO / "benchmarks" / "datas" / "customers_sample.txt"
LE_CACHE = Path.home() / ".cache" / "libembedding"
SCIFACT = LE_CACHE / "datasets" / "beir-scifact" / "scifact"
ONNX_DIR = LE_CACHE / "models--answerdotai-answerai-colbert-small-v1"
PYLATE_MODEL = "lightonai/answerai-colbert-small-v1"

N_TEXTS = 1000
WARMUP_TEXTS = 20
QUERY_LEN = 32
DOC_LEN = 300
LIBS = ["libembedding", "pylate"]
MAXSIM_IMPLS = ["lib_batch", "lib_single", "numpy"]
BATCHES = [1, 32]
# corpus name -> mode. "queries" are SciFact queries embedded in query mode.
CORPORA = {"short": "document", "long": "document", "queries": "query"}


def read_corpus(name: str, n: int) -> list[str]:
    if name == "short":
        lines = SHORT_CORPUS.read_text(encoding="utf-8").splitlines()
        return [t for t in lines if t.strip()][:n]
    if name == "long":
        out = []
        with open(SCIFACT / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out.append(f"{row.get('title') or ''} {row['text']}".strip())
                if len(out) >= n:
                    break
        return out
    if name == "queries":
        out = []
        with open(SCIFACT / "queries.jsonl", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line)["text"])
                if len(out) >= n:
                    break
        return out
    raise SystemExit(f"unknown corpus {name}")


def token_count(texts: list[str], mode: str) -> int:
    """Tokens fed to the model (same tokenizer.json for both libraries).

    Document: truncated at DOC_LEN. Query with expansion: always QUERY_LEN.
    The prefix token is counted. Same figure for both libraries by construction.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ONNX_DIR / "tokenizer.json"))
    limit = DOC_LEN if mode == "document" else QUERY_LEN
    total = 0
    for t in texts:
        n = len(tok.encode(t).ids) + 1  # +1: query/document marker
        total += QUERY_LEN if mode == "query" else min(n, limit)
    return total


# --------------------------------------------------------------------------
# Loaders: callable texts -> list of [n_i, dim] float32 arrays
# --------------------------------------------------------------------------


def load_embedder(lib: str, mode: str, batch: int):
    if lib == "libembedding":
        from libembedding import MultiVectorEmbedding

        m = MultiVectorEmbedding(
            str(ONNX_DIR), batch_size=batch, query_maxlen=QUERY_LEN, doc_maxlen=DOC_LEN,
            query_expansion=True,
        )
        return lambda xs: m.embed(xs, mode=mode)
    if lib == "pylate":
        from pylate import models

        m = models.ColBERT(
            PYLATE_MODEL, device="cpu", query_length=QUERY_LEN, document_length=DOC_LEN,
            do_query_expansion=True, local_files_only=True,
        )
        return lambda xs: m.encode(
            xs, batch_size=batch, is_query=(mode == "query"), show_progress_bar=False
        )
    raise SystemExit(f"unknown library {lib}")


def thread_facts() -> dict:
    facts: dict = {}
    with contextlib.suppress(Exception):
        import psutil

        facts["process_threads"] = psutil.Process().num_threads()
    if "torch" in sys.modules:
        facts["torch_threads"] = sys.modules["torch"].get_num_threads()
    return facts


def worker_embed(args) -> int:
    mode = CORPORA[args.corpus]
    texts = read_corpus(args.corpus, args.n_texts)
    run, load_ms = bm.timed_load(lambda: load_embedder(args.lib, mode, args.batch))
    run(texts[: args.warmup])
    times = []
    vectors = 0
    bytes_total = 0
    for _ in range(args.runs):
        start = time.perf_counter()
        out = run(texts)
        times.append(time.perf_counter() - start)
        vectors = sum(int(o.shape[0]) for o in out)
        dim = int(out[0].shape[1])
        bytes_total = vectors * dim * 4
    tokens = token_count(texts, mode)
    med = statistics.median(times)
    res = {
        "texts_per_s": len(texts) / med,
        "tokens_per_s": tokens / med,
        "vectors_per_s": vectors / med,
        "samples_s": times,
        "n_texts": len(texts),
        "tokens": tokens,
        "vectors": vectors,
        "dim": dim,
        "bytes_per_text": bytes_total / len(texts),
        "load_ms": load_ms,
        "peak_rss_mb": bm.peak_rss_mb(),
        **thread_facts(),
    }
    print("RESULT " + json.dumps(res))
    return 0


def worker_maxsim(args) -> int:
    """One query against N documents: pairs/s of one MaxSim implementation."""
    import numpy as np

    from libembedding import MultiVectorEmbedding, maxsim, maxsim_batch

    texts = read_corpus(args.corpus, args.n_texts)
    query_text = read_corpus("queries", 1)[0]
    with MultiVectorEmbedding(
        str(ONNX_DIR), query_maxlen=QUERY_LEN, doc_maxlen=DOC_LEN, query_expansion=True
    ) as mv:
        docs = mv.embed(texts, mode="document")
        query = mv.embed([query_text], mode="query")[0]

    def np_one(q, d):
        return float((q @ d.T).max(axis=1).sum())

    impls = {
        "lib_batch": lambda: maxsim_batch(query, docs),
        "lib_single": lambda: np.array([maxsim(query, d) for d in docs]),
        "numpy": lambda: np.array([np_one(query, d) for d in docs]),
    }
    run = impls[args.impl]
    run()  # warmup
    times = []
    scores = None
    for _ in range(args.runs):
        start = time.perf_counter()
        scores = run()
        times.append(time.perf_counter() - start)
    ref = np.array([np_one(query, d) for d in docs])
    res = {
        "pairs_per_s": len(docs) / statistics.median(times),
        "samples_s": times,
        "n_docs": len(docs),
        "mean_doc_vectors": float(np.mean([d.shape[0] for d in docs])),
        "max_abs_diff_vs_numpy": float(np.max(np.abs(np.asarray(scores) - ref))),
        "peak_rss_mb": bm.peak_rss_mb(),
    }
    print("RESULT " + json.dumps(res))
    return 0


def cmd_fidelity(args) -> int:
    """Not timed. Rank and value agreement of libembedding vs pylate on MaxSim scores."""
    import numpy as np
    from scipy.stats import spearmanr

    qs = read_corpus("queries", args.n_queries)
    ds = read_corpus("long", args.n_docs)
    le_q = load_embedder("libembedding", "query", 32)(qs)
    le_d = load_embedder("libembedding", "document", 32)(ds)
    py_q = load_embedder("pylate", "query", 32)(qs)
    py_d = load_embedder("pylate", "document", 32)(ds)

    def score(q, d):
        return np.array([[float((a @ b.T).max(axis=1).sum()) for b in d] for a in q])

    s_le, s_py = score(le_q, le_d), score(py_q, py_d)
    rho = [spearmanr(s_le[i], s_py[i]).statistic for i in range(len(qs))]
    out = {
        "n_queries": len(qs), "n_docs": len(ds),
        "mean_spearman_per_query": float(np.nanmean(rho)),
        "max_abs_score_diff": float(np.max(np.abs(s_le - s_py))),
        "mean_rel_score_diff": float(np.mean(np.abs(s_le - s_py) / np.maximum(np.abs(s_py), 1e-9))),
        "vectors_per_doc_equal": float(np.mean([a.shape[0] == b.shape[0] for a, b in zip(le_d, py_d)])),
    }
    print(json.dumps(out, indent=1))
    path = Path(args.out)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["fidelity"] = out
    _save(data, path)
    return 0


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def _save(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def run_worker(argv: list[str], timeout: int) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", *argv]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"timeout {timeout}s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return {"status": "ok", **json.loads(line[7:])}
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"status": "error", "error": " / ".join(tail)[:400], "rc": proc.returncode}


def plan(args) -> list[dict]:
    cells = []
    for rnd in range(args.rounds):
        for corpus in CORPORA:
            for batch in args.batches:
                libs = LIBS[rnd % len(LIBS):] + LIBS[: rnd % len(LIBS)]
                for lib in libs:
                    cells.append({"kind": "embed", "corpus": corpus, "batch": batch, "lib": lib, "round": rnd})
        for corpus in ("short", "long"):
            impls = MAXSIM_IMPLS[rnd % 3:] + MAXSIM_IMPLS[: rnd % 3]
            for impl in impls:
                cells.append({"kind": "maxsim", "corpus": corpus, "impl": impl, "round": rnd})
    return cells


def cid(c: dict) -> str:
    if c["kind"] == "embed":
        return f"embed|{c['corpus']}|b{c['batch']}|{c['lib']}|r{c['round']}"
    return f"maxsim|{c['corpus']}|{c['impl']}|r{c['round']}"


def cmd_full(args) -> int:
    out = Path(args.out)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    data.setdefault("cells", {})
    data["machine"] = bm.collect_machine_metadata()
    data["protocol"] = {"warmup_texts": args.warmup, "runs": args.runs, "rounds": args.rounds,
                        "n_texts": args.n_texts, "query_len": QUERY_LEN, "doc_len": DOC_LEN}
    versions = {}
    for name in ("libembedding", "pylate", "torch", "onnxruntime", "numpy"):
        with contextlib.suppress(Exception):
            versions[name] = getattr(__import__(name), "__version__", "?")
    data["versions"] = versions
    cells = plan(args)
    todo = [c for c in cells if cid(c) not in data["cells"]
            or (args.retry_errors and data["cells"][cid(c)]["status"] != "ok")]
    print(f"{len(cells)} cells planned, {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        if c["kind"] == "embed":
            argv = ["--kind", "embed", "--lib", c["lib"], "--corpus", c["corpus"], "--batch", str(c["batch"])]
        else:
            argv = ["--kind", "maxsim", "--impl", c["impl"], "--corpus", c["corpus"]]
        argv += ["--n-texts", str(args.n_texts), "--warmup", str(args.warmup), "--runs", str(args.runs)]
        res = run_worker(argv, args.timeout)
        data["cells"][cid(c)] = {**c, **res}
        _save(data, out)
        key = "texts_per_s" if c["kind"] == "embed" else "pairs_per_s"
        msg = f"{res[key]:.1f} {key}" if res["status"] == "ok" else res["error"][:80]
        print(f"[{i}/{len(todo)} {time.time() - t0:.0f}s] {cid(c)} -> {msg}", flush=True)
        time.sleep(args.pause)
    return 0


def _med(cells, key):
    return statistics.median([c[key] for c in cells])


def cmd_report(args) -> int:
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    ok = [c for c in data["cells"].values() if c["status"] == "ok"]
    w_lines: list[str] = []
    w = w_lines.append
    w("<!-- generated by `bench_multivector.py report`; do not edit this block by hand -->\n")
    w("### Embedding throughput (median of rounds; min-max of texts/s in brackets)\n")
    w("| Corpus | Batch | Library | texts/s | tokens/s | vectors/s | KB / text | Load (ms) | Peak RSS (MB) | Rounds |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    ratios = []
    for corpus, corpus_mode in CORPORA.items():
        for batch in BATCHES:
            row = {}
            for lib in LIBS:
                cs = [c for c in ok if c["kind"] == "embed" and c["corpus"] == corpus
                      and c["batch"] == batch and c["lib"] == lib]
                if not cs:
                    continue
                row[lib] = cs
                tp = [c["texts_per_s"] for c in cs]
                w(f"| {corpus} ({corpus_mode}) | {batch} | {lib} | {statistics.median(tp):.1f} "
                  f"({min(tp):.1f}-{max(tp):.1f}) | {_med(cs, 'tokens_per_s'):.0f} | "
                  f"{_med(cs, 'vectors_per_s'):.0f} | {_med(cs, 'bytes_per_text') / 1024:.1f} | "
                  f"{_med(cs, 'load_ms'):.0f} | {_med(cs, 'peak_rss_mb'):.0f} | {len(cs)} |")
            if len(row) == 2:
                r = _med(row["libembedding"], "texts_per_s") / _med(row["pylate"], "texts_per_s")
                ratios.append(f"| {corpus} | {batch} | {r:.2f}x | {(r - 1) * 100:+.0f} % |")
    w("")
    w("### libembedding / pylate throughput ratio\n")
    w("| Corpus | Batch | Ratio | Gap |")
    w("|---|---|---|---|")
    lines_ratio = ratios or ["| - | - | - | - |"]
    w_lines.extend(lines_ratio)
    w("")
    w("### MaxSim, one query against 1000 documents (pairs/s)\n")
    w("| Corpus | Implementation | pairs/s (min-max) | max abs diff vs numpy | mean doc vectors | Rounds |")
    w("|---|---|---|---|---|---|")
    names = {"lib_batch": "libembedding maxsim_batch", "lib_single": "libembedding maxsim (loop)",
             "numpy": "numpy (loop)"}
    for corpus in ("short", "long"):
        for impl in MAXSIM_IMPLS:
            cs = [c for c in ok if c["kind"] == "maxsim" and c["corpus"] == corpus and c["impl"] == impl]
            if cs:
                pp = [c["pairs_per_s"] for c in cs]
                w(f"| {corpus} | {names[impl]} | {statistics.median(pp):.0f} ({min(pp):.0f}-{max(pp):.0f}) | "
                  f"{max(c['max_abs_diff_vs_numpy'] for c in cs):.2e} | "
                  f"{_med(cs, 'mean_doc_vectors'):.0f} | {len(cs)} |")
    w("")
    fid = data.get("fidelity")
    if fid:
        w("### Fidelity of libembedding against pylate (not timed)\n")
        w(f"{fid['n_queries']} queries x {fid['n_docs']} SciFact documents: mean per-query Spearman of the "
          f"MaxSim scores {fid['mean_spearman_per_query']:.4f}, max absolute score difference "
          f"{fid['max_abs_score_diff']:.4f}, mean relative difference {fid['mean_rel_score_diff']:.4%}, "
          f"documents with the same vector count {fid['vectors_per_doc_equal']:.1%}.\n")
    errs = [c for c in data["cells"].values() if c["status"] != "ok"]
    if errs:
        w("### Failed cells\n")
        for c in errs:
            w(f"- {cid(c)}: {c['error']}")
        w("")
    Path(args.md).write_text("\n".join(w_lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.md}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    full = sub.add_parser("full", help="run the campaign (resumable)")
    full.add_argument("--out", default=str(HERE / "results.json"))
    full.add_argument("--rounds", type=int, default=3)
    full.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")], default=BATCHES)
    full.add_argument("--n-texts", type=int, default=N_TEXTS)
    full.add_argument("--warmup", type=int, default=WARMUP_TEXTS)
    full.add_argument("--runs", type=int, default=3)
    full.add_argument("--pause", type=float, default=3.0)
    full.add_argument("--timeout", type=int, default=3600)
    full.add_argument("--retry-errors", action="store_true")

    rep = sub.add_parser("report", help="render results.json as markdown tables")
    rep.add_argument("--json", default=str(HERE / "results.json"))
    rep.add_argument("--md", default=str(HERE / "results_tables.md"))

    fid = sub.add_parser("fidelity", help="score agreement with pylate (not timed)")
    fid.add_argument("--out", default=str(HERE / "results.json"))
    fid.add_argument("--n-queries", type=int, default=50)
    fid.add_argument("--n-docs", type=int, default=100)

    wk = sub.add_parser("worker", help="one measurement in this process")
    wk.add_argument("--kind", choices=["embed", "maxsim"], required=True)
    wk.add_argument("--lib", choices=LIBS)
    wk.add_argument("--impl", choices=MAXSIM_IMPLS)
    wk.add_argument("--corpus", choices=list(CORPORA), default="short")
    wk.add_argument("--batch", type=int, default=32)
    wk.add_argument("--n-texts", type=int, default=N_TEXTS)
    wk.add_argument("--warmup", type=int, default=WARMUP_TEXTS)
    wk.add_argument("--runs", type=int, default=3)

    args = ap.parse_args()
    if args.cmd == "full":
        return cmd_full(args)
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "fidelity":
        return cmd_fidelity(args)
    return worker_embed(args) if args.kind == "embed" else worker_maxsim(args)


if __name__ == "__main__":
    sys.exit(main())
