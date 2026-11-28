"""Competitive benchmark: libembedding vs SentenceTransformers vs FastEmbed (T15 / LE-11).

Auteur: David Orel
SPDX-License-Identifier: MIT

One measurement per process (peak RSS is a per-process high-water mark). Install
the three libraries in a dedicated virtual environment; the models must already
be in each library's cache so that no download is timed. To benchmark a source
checkout, set ``LIBEMBEDDING_LIBRARY`` to the built library and add
``python/src`` to ``PYTHONPATH``.

Reduced protocol (T15, kept unchanged)::

    python bench_competitive.py <lib> <model> <corpus>
    # lib: libembedding | sentence-transformers | fastembed
    # prints: docs_per_s load_ms peak_rss_mb

Full LE-11 protocol::

    python bench_competitive.py full  [--out results.json] [options]   # campaign, resumable
    python bench_competitive.py report [--json results.json] [--md results_tables.md]
    python bench_competitive.py worker --kind perf --lib L --model M --batch B --threads T

``full`` spawns one ``worker`` subprocess per cell (model, library, batch, threads,
round) and rewrites the JSON file after each cell. Re-running the same command skips
the cells already present.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _benchmeta as bm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CORPUS = REPO / "benchmarks" / "datas" / "customers_sample.txt"
LE_CACHE = Path.home() / ".cache" / "libembedding"
SCIFACT = LE_CACHE / "datasets" / "beir-scifact" / "scifact"

N_TEXTS = 1000
BATCH = 32
WARMUP_TEXTS = 50
ROUNDS = 3

ST_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE_SMALL = "BAAI/bge-small-en-v1.5"

# Logical models. Per library the value is what that library loads. A missing key
# means the library is not benchmarked on that model (results.md says why).
# n_texts/batches/threads restrict slow models (documented in results.md).
MODELS: dict[str, dict] = {
    "minilm": {
        "kind": "dense",
        "label": "all-MiniLM-L6-v2",
        "libembedding": ST_MINILM,
        "libembedding-gguf": str(LE_CACHE / "gguf" / "all-MiniLM-L6-v2-Q8_0.gguf"),
        "sentence-transformers": ST_MINILM,
        "fastembed": ST_MINILM,
    },
    "bge-small": {
        "kind": "dense",
        "label": "bge-small-en-v1.5",
        "libembedding": BGE_SMALL,
        # FastEmbed's default file is Qdrant/bge-small-en-v1.5-onnx-Q (model_optimized.onnx,
        # 66 MB, FP16 weights, ORT-optimized graph): libembedding-fp16 loads the same file.
        # libembedding-int8 is a genuine INT8 export (onnx-community), an extra row.
        "libembedding-fp16": BGE_SMALL,
        "libembedding-int8": BGE_SMALL,
        "libembedding-gguf": str(
            LE_CACHE / "models--BAAI-bge-small-en-v1.5" / "bge-small-en-v1.5-Q8_0.gguf"
        ),
        "sentence-transformers": BGE_SMALL,
        "fastembed": BGE_SMALL,
    },
    "qwen3-0.6b": {
        "kind": "dense",
        "label": "Qwen3-Embedding-0.6B",
        "libembedding-gguf": str(LE_CACHE / "gguf" / "qwen3-emb" / "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        "sentence-transformers": "Qwen/Qwen3-Embedding-0.6B",
        "n_texts": 100,
        "batches": [1, 32],
        "threads": [0],
    },
    "splade-pp": {
        "kind": "sparse",
        "label": "SPLADE++ en v1",
        "libembedding": "prithivida/Splade_PP_en_v1",
        "fastembed": "prithivida/Splade_PP_en_v1",
        "batches": [1, 32],
        "threads": [0],
    },
    "bge-m3-sparse": {
        "kind": "sparse",
        "label": "bge-m3 sparse",
        "libembedding": "BAAI/bge-m3",
        "n_texts": 200,
        "batches": [1, 32],
        "threads": [0],
    },
}
LIB_ORDER = [
    "libembedding", "libembedding-fp16", "libembedding-int8", "libembedding-gguf", "sentence-transformers", "fastembed",
]
BATCHES = [1, 8, 32, 64, 128]
REFERENCE_LIB = "sentence-transformers"
QUALITY_VARIANTS = {"cut": 100, "full": 0}  # variant -> document word limit (0 = none)


def read_texts(n: int) -> list[str]:
    text = CORPUS.read_text(encoding="utf-8")
    return [t for t in text.splitlines() if t.strip()][:n]


# --------------------------------------------------------------------------
# Loaders (one callable "list[str] -> embeddings" per library)
# --------------------------------------------------------------------------


def load_dense(lib: str, spec: str, batch: int, threads: int):
    if lib.startswith("libembedding"):
        from libembedding import TextEmbedding

        m = TextEmbedding(
            spec, show_download_progress=False, threads=threads, batch_size=batch,
            quantization={"libembedding-int8": "dynamic", "libembedding-fp16": "fp16"}.get(lib),
        )
        return lambda xs: m.embed(xs, batch_size=batch)
    if lib == "sentence-transformers":
        import torch
        from sentence_transformers import SentenceTransformer

        if threads > 0:
            torch.set_num_threads(threads)
        st = SentenceTransformer(spec, device="cpu")
        return lambda xs: st.encode(xs, batch_size=batch)
    if lib == "fastembed":
        from fastembed import TextEmbedding as FastEmbedText

        fe = FastEmbedText(spec, threads=threads if threads > 0 else None)
        return lambda xs: list(fe.embed(xs, batch_size=batch))
    raise SystemExit(f"unknown library: {lib}")


def load_sparse(lib: str, spec: str, batch: int, threads: int):
    if lib.startswith("libembedding"):
        from libembedding import SparseTextEmbedding

        m = SparseTextEmbedding(
            spec, show_download_progress=False, threads=threads, batch_size=batch
        )
        return lambda xs: m.embed(xs, batch_size=batch)
    if lib == "fastembed":
        from fastembed import SparseTextEmbedding as FastEmbedSparse

        fe = FastEmbedSparse(spec, threads=threads if threads > 0 else None)
        return lambda xs: list(fe.embed(xs, batch_size=batch))
    raise SystemExit(f"unsupported sparse library: {lib}")


def load(lib: str, kind: str, spec: str, batch: int, threads: int):
    return (load_sparse if kind == "sparse" else load_dense)(lib, spec, batch, threads)


def thread_facts() -> dict:
    """What the process really runs with, to verify the thread setting took effect."""
    facts: dict = {}
    with contextlib.suppress(Exception):
        import psutil

        facts["process_threads"] = psutil.Process().num_threads()
    if "torch" in sys.modules:
        facts["torch_threads"] = sys.modules["torch"].get_num_threads()
    return facts


# --------------------------------------------------------------------------
# Worker: one measurement in this process
# --------------------------------------------------------------------------


def worker_perf(args) -> int:
    spec = MODELS[args.model][args.lib]
    kind = MODELS[args.model]["kind"]
    texts = read_texts(args.n_texts)
    run, load_ms = bm.timed_load(lambda: load(args.lib, kind, spec, args.batch, args.threads))
    run(texts[: args.warmup])
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        run(texts)
        samples.append(len(texts) / (time.perf_counter() - start))
    out = {
        "docs_per_s": statistics.median(samples),
        "samples": samples,
        "load_ms": load_ms,
        "peak_rss_mb": bm.peak_rss_mb(),
        "n_texts": len(texts),
        **thread_facts(),
    }
    print("RESULT " + json.dumps(out))
    return 0


def _natural_key(s: str):
    return (len(s), s)


def load_pairs(n_pairs: int = 100, words: int = 0):
    """100 BEIR SciFact (query, document, label) pairs: half relevant, half not.

    Relevant: the first qrels (test) pair of the first n/2 queries. Not relevant:
    the same query with the relevant document of the next query (deterministic).
    ``words`` > 0 cuts every document to its first ``words`` words so that no
    library truncates (the GGUF dense backend aborts above 512 tokens).
    """
    qrels: dict[str, str] = {}
    with open(SCIFACT / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, did)
    queries = {}
    with open(SCIFACT / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    qids = sorted((q for q in qrels if q in queries), key=_natural_key)[: n_pairs // 2 + 1]
    needed = {qrels[q] for q in qids}
    corpus = {}
    with open(SCIFACT / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row["_id"]) in needed:
                text = f"{row.get('title') or ''} {row['text']}".strip()
                corpus[str(row["_id"])] = " ".join(text.split()[:words]) if words else text
    half = n_pairs // 2
    pairs = []
    for i in range(half):
        q = qids[i]
        pairs.append((queries[q], corpus[qrels[q]], 1))
        pairs.append((queries[q], corpus[qrels[qids[i + 1]]], 0))
    return pairs


def worker_quality(args) -> int:
    """Embed the texts of the 100 pairs; save the vectors for the aggregator."""
    import numpy as np

    kind = MODELS[args.model]["kind"]
    spec = MODELS[args.model][args.lib]
    pairs = load_pairs(words=args.words)
    texts = sorted({t for q, d, _ in pairs for t in (q, d)})
    run = load(args.lib, kind, spec, BATCH, 0)
    vec = np.asarray(run(texts), dtype=np.float32)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    np.save(args.out, vec)
    print("RESULT " + json.dumps({"n_texts": len(texts), "dim": int(vec.shape[1])}))
    return 0


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def cell_id(kind, model, lib, batch, threads, rnd) -> str:
    return f"{kind}|{model}|{lib}|b{batch}|t{threads}|r{rnd}"


def save_json(data: dict, path: Path) -> None:
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


def plan_cells(args) -> list[dict]:
    """Headline cells first (all rounds), then the grid (grid_rounds)."""
    ncpu = os.cpu_count() or 8
    default_threads = [0, ncpu, max(1, ncpu // 2)]
    wanted_models = args.models.split(",") if args.models else list(MODELS)
    headline, grid = [], []
    for rnd in range(args.rounds):
        for mname in wanted_models:
            m = MODELS[mname]
            libs = [lib for lib in LIB_ORDER if lib in m]
            if args.libs:
                libs = [lib for lib in libs if lib in args.libs.split(",")]
            shift = rnd % len(libs) if libs else 0
            libs = libs[shift:] + libs[:shift]  # rotate the order between rounds
            batches = args.batches or m.get("batches", BATCHES)
            threads_list = (
                [int(t) for t in args.threads.split(",")] if args.threads
                else m.get("threads", default_threads)
            )
            for b in batches:
                for t in threads_list:
                    is_headline = b == BATCH and t == 0
                    if rnd >= args.grid_rounds and not is_headline:
                        continue
                    for lib in libs:
                        cell = {
                            "kind": m["kind"], "model": mname, "lib": lib,
                            "batch": b, "threads": t, "round": rnd,
                            "n_texts": args.n_texts or m.get("n_texts", N_TEXTS),
                        }
                        (headline if is_headline else grid).append(cell)
    return headline + grid


def _cid(c: dict) -> str:
    return cell_id(c["kind"], c["model"], c["lib"], c["batch"], c["threads"], c["round"])


def cmd_full(args) -> int:
    out = Path(args.out)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    data.setdefault("cells", {})
    data["machine"] = bm.collect_machine_metadata()
    data["protocol"] = {
        "warmup_texts": args.warmup, "runs": args.runs, "rounds": args.rounds,
        "grid_rounds": args.grid_rounds, "batch_default": BATCH,
    }
    data["versions"] = lib_versions()
    cells = plan_cells(args)

    def pending(c: dict) -> bool:
        prev = data["cells"].get(_cid(c))
        return prev is None or (args.retry_errors and prev["status"] != "ok")

    todo = [] if args.quality_only else [c for c in cells if pending(c)]
    print(f"{len(cells)} cells planned, {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        argv = ["--kind", "perf", "--lib", c["lib"], "--model", c["model"], "--batch", str(c["batch"]),
                "--threads", str(c["threads"]), "--n-texts", str(c["n_texts"]),
                "--warmup", str(args.warmup), "--runs", str(args.runs)]
        res = run_worker(argv, args.timeout)
        data["cells"][_cid(c)] = {**c, **res}
        save_json(data, out)
        msg = f"{res['docs_per_s']:.1f} docs/s" if res["status"] == "ok" else res["error"][:80]
        print(f"[{i}/{len(todo)} {time.time() - t0:.0f}s] {_cid(c)} -> {msg}", flush=True)
        time.sleep(args.pause)
    if not args.skip_quality:
        quality(args, data, out)
    return 0


def lib_versions() -> dict:
    v = {}
    for name in ("libembedding", "sentence_transformers", "fastembed", "torch", "onnxruntime", "numpy"):
        with contextlib.suppress(Exception):
            if name == "libembedding":
                import libembedding

                v[name] = getattr(libembedding, "__version__", "?")
            else:
                v[name] = __import__(name).__version__
    return v


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    r = spearmanr(a, b).statistic
    return 0.0 if math.isnan(r) else float(r)


def quality(args, data: dict, out: Path) -> None:
    """Agreement with the reference library and Spearman of pair scores vs labels."""
    import numpy as np

    wanted_models = [m for m in (args.models.split(",") if args.models else list(MODELS))
                     if MODELS[m]["kind"] == "dense"]
    qual = data.setdefault("quality", {})
    scratch = Path(tempfile.mkdtemp(prefix="lembed_quality_"))
    # "cut": documents cut to 100 words, so no library truncates and the GGUF
    # backend stays under its 512-token limit. "full": whole abstracts; the GGUF
    # dense backend is skipped there (it aborts above 512 tokens).
    for variant, words in QUALITY_VARIANTS.items():
        pairs = load_pairs(words=words)
        texts = sorted({t for q, d, _ in pairs for t in (q, d)})
        index = {t: i for i, t in enumerate(texts)}
        labels = np.array([p[2] for p in pairs], dtype=np.float64)
        for mname in wanted_models:
            if mname in qual.get(variant, {}) and not args.retry_errors:
                continue
            vecs = {}
            result = qual.setdefault(variant, {}).setdefault(mname, {})
            for lib in LIB_ORDER:
                if lib not in MODELS[mname] or (args.libs and lib not in args.libs.split(",")):
                    continue
                if variant == "full" and lib == "libembedding-gguf":
                    continue
                path = scratch / f"{variant}_{mname}_{lib}.npy"
                res = run_worker(["--kind", "quality", "--lib", lib, "--model", mname,
                                  "--out", str(path), "--words", str(words)], args.timeout)
                if res["status"] == "ok":
                    vecs[lib] = np.load(path)
                else:
                    result[lib] = res
            scores = {
                lib: np.array([float(v[index[q]] @ v[index[d]]) for q, d, _ in pairs])
                for lib, v in vecs.items()
            }
            for lib, v in vecs.items():
                entry = {"status": "ok", "spearman_vs_labels": _spearman(scores[lib], labels)}
                if REFERENCE_LIB in vecs and lib != REFERENCE_LIB:
                    cos = (v * vecs[REFERENCE_LIB]).sum(axis=1)
                    entry.update(
                        ref=REFERENCE_LIB, cos_mean=float(cos.mean()), cos_min=float(cos.min()),
                        pair_score_spearman_vs_ref=_spearman(scores[lib], scores[REFERENCE_LIB]),
                    )
                result[lib] = entry
            save_json(data, out)
            print(f"quality {variant} {mname}: {json.dumps(result)}", flush=True)
        data.setdefault("quality_meta", {})[variant] = {
            "n_pairs": len(pairs), "n_texts": len(texts), "words": words,
            "reference": REFERENCE_LIB,
            "dataset": "BEIR SciFact test, 50 relevant + 50 non-relevant pairs",
        }
    save_json(data, out)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def aggregate(data: dict) -> dict:
    """(model, lib, batch, threads) -> medians over rounds."""
    groups: dict = {}
    for c in data["cells"].values():
        if c["status"] == "ok":
            groups.setdefault((c["model"], c["lib"], c["batch"], c["threads"]), []).append(c)
    agg = {}
    for key, cs in groups.items():
        d = [c["docs_per_s"] for c in cs]
        agg[key] = {
            "docs": statistics.median(d), "min": min(d), "max": max(d), "n": len(cs),
            "load_ms": statistics.median([c["load_ms"] for c in cs]),
            "rss": statistics.median([c["peak_rss_mb"] for c in cs]),
            "threads_seen": statistics.median([c.get("process_threads", math.nan) for c in cs]),
        }
    return agg


LIB_LABEL = {
    "libembedding": "libembedding (ONNX FP32)",
    "libembedding-fp16": "libembedding (ONNX FP16 file, = FastEmbed's)",
    "libembedding-int8": "libembedding (ONNX INT8 dynamic)",
    "libembedding-gguf": "libembedding (GGUF Q8_0)",
    "sentence-transformers": "SentenceTransformers (torch FP32)",
    "fastembed": "FastEmbed (ONNX, own default file)",
}


def cmd_report(args) -> int:
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    agg = aggregate(data)
    lines: list[str] = []
    w = lines.append
    models = [m for m in MODELS if any(k[0] == m for k in agg)]
    ncpu = os.cpu_count() or 8
    tname = {0: "default", ncpu: str(ncpu), max(1, ncpu // 2): str(max(1, ncpu // 2))}
    gaps: list[str] = []
    w("<!-- generated by `bench_competitive.py report`; do not edit this block by hand -->\n")
    for m in models:
        info = MODELS[m]
        libs = [lib for lib in LIB_ORDER if any(k[0] == m and k[1] == lib for k in agg)]
        w(f"### {info['label']} ({info['kind']})\n")
        keys = [k for k in agg if k[0] == m]
        batches = sorted({k[2] for k in keys})
        threads_set = sorted({k[3] for k in keys})
        w("Headline (batch 32, default threads, median of rounds; range in brackets):\n")
        w("| Library | docs/s | Load (ms) | Peak RSS (MB) | Rounds |")
        w("|---|---|---|---|---|")
        for lib in libs:
            a = agg.get((m, lib, BATCH, 0))
            if a:
                w(f"| {LIB_LABEL[lib]} | {a['docs']:.1f} ({a['min']:.1f}-{a['max']:.1f}) "
                  f"| {a['load_ms']:.0f} | {a['rss']:.0f} | {a['n']} |")
        w("")
        w("Throughput by batch size, default threads (docs/s):\n")
        w("| Library | " + " | ".join(f"b{b}" for b in batches) + " |")
        w("|---|" + "---|" * len(batches))
        for lib in libs:
            row = [f"{agg[(m, lib, b, 0)]['docs']:.1f}" if (m, lib, b, 0) in agg else "-" for b in batches]
            w(f"| {LIB_LABEL[lib]} | " + " | ".join(row) + " |")
        w("")
        if len(threads_set) > 1:
            w("Throughput by thread count, batch 32 (docs/s; process threads seen in brackets):\n")
            w("| Library | " + " | ".join(tname.get(t, str(t)) for t in threads_set) + " |")
            w("|---|" + "---|" * len(threads_set))
            for lib in libs:
                row = []
                for t in threads_set:
                    a = agg.get((m, lib, BATCH, t))
                    row.append(f"{a['docs']:.1f} [{a['threads_seen']:.0f}]" if a else "-")
                w(f"| {LIB_LABEL[lib]} | " + " | ".join(row) + " |")
            w("")
        for b in batches:
            for t in threads_set:
                mine = [(lib, agg[(m, lib, b, t)]["docs"])
                        for lib in ("libembedding", "libembedding-fp16", "libembedding-int8",
                                    "libembedding-gguf")
                        if (m, lib, b, t) in agg]
                others = [(lib, agg[(m, lib, b, t)]["docs"]) for lib in ("sentence-transformers", "fastembed")
                          if (m, lib, b, t) in agg]
                for lib, v in mine:
                    for olib, ov in others:
                        gap = (v / ov - 1.0) * 100
                        # listed for the headline settings; results.json has every cell
                        if abs(gap) > 20 and t == 0 and b in (1, 32):
                            gaps.append(f"| {info['label']} | b{b} / {tname.get(t, t)} | {lib} vs {olib} | "
                                        f"{v:.1f} vs {ov:.1f} | {gap:+.0f} % |")
    w("### Gaps above 20 % (libembedding vs another library, same cell)\n")
    w("| Model | Cell | Pair | docs/s | Gap |")
    w("|---|---|---|---|---|")
    lines.extend(gaps or ["| - | - | none | - | - |"])
    w("")
    qual = data.get("quality", {})
    if qual:
        w("### Quality: agreement and pair ranking (100 BEIR SciFact pairs)\n")
        w(f"Reference library: {REFERENCE_LIB}. cos = cosine between a library's vector and the "
          "reference's for the same text (mean / min over the pairs' texts); Spearman = rank "
          "correlation of the 100 pair cosines with the 0/1 relevance labels; pair-vs-ref = "
          "Spearman between a library's pair scores and the reference's. Variant `cut`: "
          "documents cut to 100 words (no library truncates). Variant `full`: whole abstracts "
          "(each library applies its own default maximum length; GGUF dense skipped, see Limits).\n")
        w("| Variant | Model | Library | cos mean | cos min | Spearman vs labels | Spearman pair scores vs ref |")
        w("|---|---|---|---|---|---|---|")
        for variant, per_model in qual.items():
            for m, per in per_model.items():
                for lib, e in per.items():
                    if e.get("status") != "ok":
                        w(f"| {variant} | {MODELS[m]['label']} | {LIB_LABEL[lib]} | error | | | |")
                        continue
                    w(f"| {variant} | {MODELS[m]['label']} | {LIB_LABEL[lib]} | "
                      f"{e.get('cos_mean', 1.0):.4f} | {e.get('cos_min', 1.0):.4f} | "
                      f"{e['spearman_vs_labels']:.3f} | "
                      f"{e.get('pair_score_spearman_vs_ref', 1.0):.3f} |")
        w("")
    errs = [c for c in data["cells"].values() if c["status"] != "ok"]
    if errs:
        w("### Failed cells\n")
        for c in errs:
            w(f"- {c['model']} / {c['lib']} / b{c['batch']} / t{c['threads']} / r{c['round']}: {c['error']}")
        w("")
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.md}")
    return 0


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def legacy(argv: list[str]) -> int:
    """Reduced T15 protocol: one (library, model, corpus) measurement."""
    lib, model, corpus = argv[0], argv[1], argv[2]
    text = Path(corpus).read_text(encoding="utf-8")
    texts = [t for t in text.splitlines() if t.strip()][:N_TEXTS]
    run, load_ms = bm.timed_load(lambda: load_dense(lib, model, BATCH, 0))
    run(texts[:WARMUP_TEXTS])
    samples = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        run(texts)
        samples.append(len(texts) / (time.perf_counter() - start))
    print(f"{statistics.median(samples):.2f} {load_ms:.1f} {bm.peak_rss_mb():.1f}")
    return 0


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] not in ("full", "report", "worker"):
        return legacy(sys.argv[1:4])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    full = sub.add_parser("full", help="run the full LE-11 campaign (resumable)")
    full.add_argument("--out", default=str(HERE / "results.json"))
    full.add_argument("--models", help="comma list of logical models (default: all)")
    full.add_argument("--libs", help="comma list of libraries (default: all available per model)")
    full.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")])
    full.add_argument("--threads", help="comma list, 0 = library default")
    full.add_argument("--rounds", type=int, default=3, help="rounds for the headline cell (b32, default threads)")
    full.add_argument("--grid-rounds", type=int, default=1, help="rounds for the other cells")
    full.add_argument("--n-texts", type=int, default=0, help="override the corpus size (smoke test)")
    full.add_argument("--warmup", type=int, default=WARMUP_TEXTS)
    full.add_argument("--runs", type=int, default=3)
    full.add_argument("--pause", type=float, default=3.0, help="seconds between cells (thermal)")
    full.add_argument("--timeout", type=int, default=3600)
    full.add_argument("--retry-errors", action="store_true")
    full.add_argument("--skip-quality", action="store_true")
    full.add_argument("--quality-only", action="store_true")

    rep = sub.add_parser("report", help="render results.json as markdown tables")
    rep.add_argument("--json", default=str(HERE / "results.json"))
    rep.add_argument("--md", default=str(HERE / "results_tables.md"))

    wk = sub.add_parser("worker", help="one measurement in this process (used by full)")
    wk.add_argument("--kind", choices=["perf", "quality"], required=True)
    wk.add_argument("--lib", required=True)
    wk.add_argument("--model", required=True)
    wk.add_argument("--batch", type=int, default=BATCH)
    wk.add_argument("--threads", type=int, default=0)
    wk.add_argument("--n-texts", type=int, default=N_TEXTS)
    wk.add_argument("--warmup", type=int, default=WARMUP_TEXTS)
    wk.add_argument("--runs", type=int, default=3)
    wk.add_argument("--out")
    wk.add_argument("--words", type=int, default=0, help="quality: cut documents to N words")

    args = ap.parse_args()
    if args.cmd == "full":
        return cmd_full(args)
    if args.cmd == "report":
        return cmd_report(args)
    return worker_quality(args) if args.kind == "quality" else worker_perf(args)


if __name__ == "__main__":
    sys.exit(main())
