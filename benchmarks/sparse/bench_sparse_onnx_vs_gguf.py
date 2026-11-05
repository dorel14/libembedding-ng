"""Sparse ONNX vs sparse GGUF: vector fidelity, retrieval quality and cost.

Both backends run the same SPLADE++ checkpoint (prithivida/Splade_PP_en_v1):
ONNX Runtime on the FP32 export, libembedding's own ggml runtime on the Q8_0
GGUF. The question is not "which is faster" alone but "what does the GGUF file
cost in quality for what it saves", so one run answers three things on the same
vectors:

1. **Fidelity** -- per text, the cosine between the two sparse vectors, the
   largest weight difference, the overlap of their ten heaviest terms and the
   ratio of their sizes. Vectors are kept whole (``top_k = 0``): a cosine between
   two truncated vectors measures the truncation, not the model.
2. **Retrieval** -- nDCG@10, Recall@100 and MRR@10 against human relevance
   judgements (BEIR format), plus how much the two rankings agree with each
   other (overlap of the top 10). A fidelity number nobody can relate to search
   quality is not worth publishing on its own.
3. **Cost** -- corpus throughput, single-query latency and peak RSS, one
   backend per subprocess (a peak is a per-process high-water mark) and
   alternated across rounds, so thermal drift on a laptop hits both sides.

Dataset: any BEIR folder (``corpus.jsonl``, ``queries.jsonl``,
``qrels/<split>.tsv``). SciFact is small enough for a laptop::

    https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

``--max-docs`` subsamples the corpus deterministically, keeping every document
judged relevant for the selected queries, so a quick run still has answers to
find. Absolute scores on a subsample are higher than on the full corpus; the
ONNX/GGUF comparison stays fair because both see the same documents.

Set ``LIBEMBEDDING_LIBRARY`` to the freshly built DLL when running from a source
checkout, or the package copy -- possibly stale -- is what gets measured.

Auteur: David Orel
Version: 1.11.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_sparse_backend_comparison import (
    collect_environment,
    median,
    process_peak_mb,
)

_RESULT_PREFIX = "__RESULT__ "
_LATENCY_QUERIES = 50


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    doc_ids: list[str]
    docs: list[str]
    query_ids: list[str]
    queries: list[str]
    qrels: dict[str, dict[str, int]]


def load_beir(root: Path, split: str, max_docs: int, max_queries: int) -> Dataset:
    """Reads a BEIR folder; the files are data and are only ever parsed."""
    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)

    query_text: dict[str, str] = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query_text[str(row["_id"])] = row["text"]

    query_ids = sorted((q for q in qrels if q in query_text), key=_natural_key)
    if max_queries > 0:
        query_ids = query_ids[:max_queries]
    qrels = {q: qrels[q] for q in query_ids}

    corpus: dict[str, str] = {}
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            title = row.get("title") or ""
            corpus[str(row["_id"])] = f"{title} {row['text']}".strip()

    doc_ids = list(corpus)
    if max_docs > 0 and max_docs < len(doc_ids):
        relevant = {d for rel in qrels.values() for d in rel if d in corpus}
        keep = [d for d in doc_ids if d in relevant]
        for d in doc_ids:
            if len(keep) >= max_docs:
                break
            if d not in relevant:
                keep.append(d)
        doc_ids = keep

    return Dataset(
        doc_ids=doc_ids,
        docs=[corpus[d] for d in doc_ids],
        query_ids=query_ids,
        queries=[query_text[q] for q in query_ids],
        qrels=qrels,
    )


def _natural_key(s: str):
    return (0, int(s)) if s.isdigit() else (1, s)


# --------------------------------------------------------------------------- #
# Worker: one backend, one process
# --------------------------------------------------------------------------- #


@dataclass
class Spec:
    backend: str  # "onnx" | "gguf"
    onnx_model: str
    gguf_repo: str
    gguf_file: str
    threads: int
    batch_size: int  # 0 = library default for the backend
    max_length: int


@dataclass
class WorkerResult:
    backend: str
    ok: bool = True
    error: str = ""
    load_s: float = 0.0
    corpus_s: float = 0.0
    docs_per_s: float = 0.0
    query_p50_ms: float = 0.0
    peak_rss_mb: float = 0.0
    library: str = ""


def load_model(spec: Spec):
    from libembedding import SparseTextEmbedding

    common = {
        "threads": spec.threads,
        "batch_size": spec.batch_size or None,
        "max_length": spec.max_length,
        "top_terms": 0,
    }
    if spec.backend == "onnx":
        return SparseTextEmbedding(
            spec.onnx_model, offline=True, show_download_progress=False, **common
        )
    return SparseTextEmbedding.from_gguf_model(
        spec.gguf_repo, spec.gguf_file, offline=True,
        show_download_progress=False, **common,
    )


def save_vectors(path: Path, embeddings) -> None:
    lengths = np.array([len(e.indices) for e in embeddings], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    indices = (np.concatenate([e.indices for e in embeddings])
               if len(embeddings) else np.zeros(0, np.int32))
    values = (np.concatenate([e.values for e in embeddings])
              if len(embeddings) else np.zeros(0, np.float32))
    np.savez(path, offsets=offsets, indices=indices, values=values)


def run_worker(payload: dict) -> WorkerResult:
    spec = Spec(**payload["spec"])
    result = WorkerResult(backend=spec.backend)
    try:
        from libembedding._binding import _find_library

        result.library = _find_library()
        docs = payload["docs"]
        queries = payload["queries"]

        t0 = time.perf_counter()
        model = load_model(spec)
        result.load_s = time.perf_counter() - t0

        model.embed(docs[: min(16, len(docs))])  # warm the allocators

        t0 = time.perf_counter()
        doc_vecs = model.embed(docs)
        result.corpus_s = time.perf_counter() - t0
        result.docs_per_s = len(docs) / result.corpus_s if result.corpus_s else 0.0

        query_vecs = model.embed(queries)
        latencies = []
        for q in queries[:_LATENCY_QUERIES]:
            t0 = time.perf_counter()
            model.embed([q], batch_size=1)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        result.query_p50_ms = median(latencies) if latencies else 0.0

        if payload.get("vectors_dir"):
            out = Path(payload["vectors_dir"])
            save_vectors(out / f"{spec.backend}_docs.npz", doc_vecs)
            save_vectors(out / f"{spec.backend}_queries.npz", query_vecs)
        result.peak_rss_mb = process_peak_mb()
    except Exception as exc:  # noqa: BLE001 -- reported, the parent compares sides
        # The frame and the state of the output folder travel with the message:
        # a bare str(exc) from a subprocess says what failed, never where.
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        vectors_dir = payload.get("vectors_dir") or ""
        result.ok = False
        result.error = (
            f"{type(exc).__name__}: {exc} [at {Path(frame.filename).name}:"
            f"{frame.lineno} in {frame.name}; vectors_dir exists="
            f"{os.path.isdir(vectors_dir) if vectors_dir else 'n/a'}]"
        )
    return result


def spawn_worker(payload: dict) -> WorkerResult:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(payload, f)
        payload_path = f.name
    try:
        proc = subprocess.run(  # fixed argv, no shell
            [sys.executable, __file__, "--worker", payload_path],
            capture_output=True, text=True, check=False,
        )
    finally:
        os.unlink(payload_path)
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return WorkerResult(**json.loads(line[len(_RESULT_PREFIX):]))
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
    return WorkerResult(backend=payload["spec"]["backend"], ok=False,
                        error=f"worker exited {proc.returncode}: {' | '.join(tail)}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class SparseMatrix:
    """Ragged sparse vectors as written by save_vectors()."""

    def __init__(self, path: Path):
        data = np.load(path)
        self.offsets = data["offsets"]
        self.indices = data["indices"]
        self.values = data["values"]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.indices[lo:hi], self.values[lo:hi]


def fidelity(a: SparseMatrix, b: SparseMatrix, top: int = 10) -> dict[str, float]:
    """Per-text agreement between two backends, summarised over the texts."""
    cosines, max_diffs, overlaps, size_ratios = [], [], [], []
    for i in range(len(a)):
        ia, va = a.row(i)
        ib, vb = b.row(i)
        common, pa, pb = np.intersect1d(ia, ib, assume_unique=True,
                                        return_indices=True)
        dot = float(np.dot(va[pa], vb[pb])) if len(common) else 0.0
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        cosines.append(dot / (na * nb) if na > 0 and nb > 0 else float(na == nb))

        da = dict(zip(ia.tolist(), va.tolist()))
        db = dict(zip(ib.tolist(), vb.tolist()))
        keys = set(da) | set(db)
        max_diffs.append(max((abs(da.get(k, 0.0) - db.get(k, 0.0)) for k in keys),
                             default=0.0))

        ta = set(ia[np.argsort(-va)[:top]].tolist())
        tb = set(ib[np.argsort(-vb)[:top]].tolist())
        overlaps.append(len(ta & tb) / max(1, len(ta), len(tb)))
        size_ratios.append(len(ib) / len(ia) if len(ia) else 1.0)

    cos = np.array(cosines)
    return {
        "texts": len(cosines),
        "cosine_mean": float(cos.mean()) if len(cos) else 0.0,
        "cosine_p5": float(np.percentile(cos, 5)) if len(cos) else 0.0,
        "cosine_min": float(cos.min()) if len(cos) else 0.0,
        "max_weight_diff": float(max(max_diffs, default=0.0)),
        f"top{top}_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "nnz_ratio_mean": float(np.mean(size_ratios)) if size_ratios else 0.0,
    }


def rank(docs: SparseMatrix, queries: SparseMatrix, depth: int) -> list[np.ndarray]:
    """Dot-product retrieval through an inverted index: [query] -> doc rows."""
    n_docs = len(docs)
    rows = np.repeat(np.arange(n_docs), np.diff(docs.offsets))
    order = np.argsort(docs.indices, kind="stable")
    terms, postings, weights = docs.indices[order], rows[order], docs.values[order]

    ranked = []
    for qi in range(len(queries)):
        qt, qw = queries.row(qi)
        scores = np.zeros(n_docs, dtype=np.float32)
        lo = np.searchsorted(terms, qt, side="left")
        hi = np.searchsorted(terms, qt, side="right")
        for l, h, w in zip(lo, hi, qw):
            scores[postings[l:h]] += w * weights[l:h]
        k = min(depth, n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        ranked.append(top[np.argsort(-scores[top], kind="stable")])
    return ranked


def retrieval_metrics(ranked: list[np.ndarray], ds: Dataset) -> dict[str, float]:
    ndcg, recall, mrr = [], [], []
    for qi, qid in enumerate(ds.query_ids):
        rel = ds.qrels[qid]
        got = [ds.doc_ids[d] for d in ranked[qi]]

        dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(r + 2)
                  for r, d in enumerate(got[:10]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg else 0.0)

        recall.append(len(set(got[:100]) & set(rel)) / len(rel))
        mrr.append(next((1.0 / (r + 1) for r, d in enumerate(got[:10]) if d in rel),
                        0.0))
    return {
        "ndcg@10": float(np.mean(ndcg)),
        "recall@100": float(np.mean(recall)),
        "mrr@10": float(np.mean(mrr)),
    }


def ranking_agreement(a: list[np.ndarray], b: list[np.ndarray], k: int = 10) -> float:
    return float(np.mean([len(set(x[:k].tolist()) & set(y[:k].tolist())) / k
                          for x, y in zip(a, b)]))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    environment: dict
    dataset: dict
    settings: dict
    cost: dict = field(default_factory=dict)
    fidelity_docs: dict = field(default_factory=dict)
    fidelity_queries: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    top10_agreement: float = 0.0
    errors: list = field(default_factory=list)


def write_markdown(r: Report, path: Path) -> None:
    lines = ["# Sparse ONNX vs sparse GGUF", ""]
    env = r.environment
    lines += [
        (f"- Date: {env.get('timestamp')} — git `{env.get('git')}` — "
         f"libembedding {env.get('libembedding')}"),
        f"- Machine: {env.get('platform')}, {env.get('cpu_count')} logical CPUs",
        (f"- Dataset: {r.dataset['name']} ({r.dataset['docs']} documents, "
         f"{r.dataset['queries']} queries, split `{r.dataset['split']}`)"),
        (f"- Settings: threads={r.settings['threads']}, "
         f"batch={r.settings['batch_size'] or 'library default'}, "
         f"max_length={r.settings['max_length']}, top_k=0, "
         f"rounds={r.settings['rounds']}"),
        "",
    ]

    if r.errors:
        lines += ["## Errors", ""] + [f"- {e}" for e in r.errors] + [""]

    lines += ["## Retrieval quality", "",
          "| Backend | nDCG@10 | Recall@100 | MRR@10 |", "|---|---|---|---|"]
    for name, m in r.retrieval.items():
        lines.append(f"| {name} | {m['ndcg@10']:.4f} | {m['recall@100']:.4f} | "
                 f"{m['mrr@10']:.4f} |")
    lines += ["", f"Top-10 ranking overlap ONNX vs GGUF: **{r.top10_agreement:.3f}**", ""]

    lines += ["## Vector fidelity (GGUF against ONNX)", "",
          ("| Texts | n | cosine mean | cosine p5 | cosine min | max Δweight | "
           "top-10 term overlap | nnz GGUF/ONNX |"),
          "|---|---|---|---|---|---|---|---|"]
    for label, f in (("documents", r.fidelity_docs), ("queries", r.fidelity_queries)):
        if f:
            lines.append(f"| {label} | {f['texts']} | {f['cosine_mean']:.5f} | "
                     f"{f['cosine_p5']:.5f} | {f['cosine_min']:.5f} | "
                     f"{f['max_weight_diff']:.4f} | {f['top10_overlap_mean']:.3f} | "
                     f"{f['nnz_ratio_mean']:.3f} |")

    lines += ["", "## Cost", "",
          "| Backend | corpus docs/s (median) | query p50 ms | load s | peak RSS MB |",
          "|---|---|---|---|---|"]
    for name, c in r.cost.items():
        lines.append(f"| {name} | {c['docs_per_s']:.2f} | {c['query_p50_ms']:.1f} | "
                 f"{c['load_s']:.2f} | {c['peak_rss_mb']:.0f} |")
    lines += ["",
          ("Throughput is the median over rounds; backends alternate within "
           "each round. Peak RSS is per subprocess."),
          ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def owned_vectors_dir(report: Report, report_path: Path):
    """Where the workers leave their vectors: next to the report, not in %TEMP%.

    A run lasts up to hours, and %TEMP% is shared ground other tools may clean:
    on 2026-11-04 the GGUF worker of one run found its temporary folder gone
    when it came to save (FileNotFoundError), with no trace of what removed it.
    A folder this run owns cannot be swept from under it. It is removed when the
    run succeeds and kept when it does not, so a failure can be inspected.
    """
    path = report_path.with_name(report_path.stem + "_vectors")
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)
    if not report.errors:
        shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset", type=Path,
                   default=Path.home() / ".cache/libembedding/datasets/beir-scifact/scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--max-docs", type=int, default=0, help="0 = whole corpus")
    p.add_argument("--max-queries", type=int, default=0, help="0 = every query")
    p.add_argument("--onnx-model", default="Qdrant/Splade_PP_en_v1")
    p.add_argument("--gguf-repo", default="cstr/splade-pp-en-v1-GGUF")
    p.add_argument("--gguf-file", default="splade-pp-en-v1-q8_0.gguf")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=0,
                   help="0 = each backend's library default")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--rounds", type=int, default=1,
                   help="cost rounds; vectors come from the first")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent / "onnx_vs_gguf_results.md")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        payload = json.loads(Path(args.worker).read_text(encoding="utf-8"))
        print(_RESULT_PREFIX + json.dumps(asdict(run_worker(payload))), flush=True)
        return 0

    ds = load_beir(args.dataset, args.split, args.max_docs, args.max_queries)
    print(f"dataset: {len(ds.docs)} docs, {len(ds.queries)} queries", flush=True)

    report = Report(
        environment=collect_environment(),
        dataset={"name": args.dataset.name, "split": args.split,
                 "docs": len(ds.docs), "queries": len(ds.queries)},
        settings={"threads": args.threads, "batch_size": args.batch_size,
                  "max_length": args.max_length, "rounds": args.rounds},
    )

    with owned_vectors_dir(report, args.output) as vectors_dir:
        runs: dict[str, list[WorkerResult]] = {"onnx": [], "gguf": []}
        for rnd in range(args.rounds):
            order = ("onnx", "gguf") if rnd % 2 == 0 else ("gguf", "onnx")
            for backend in order:
                spec = Spec(backend, args.onnx_model, args.gguf_repo, args.gguf_file,
                            args.threads, args.batch_size, args.max_length)
                payload = {"spec": asdict(spec), "docs": ds.docs,
                           "queries": ds.queries,
                           "vectors_dir": vectors_dir if rnd == 0 else ""}
                res = spawn_worker(payload)
                print(f"round {rnd + 1} {backend}: "
                      + (f"{res.docs_per_s:.2f} docs/s, {res.peak_rss_mb:.0f} MB"
                         if res.ok else f"FAILED {res.error}"), flush=True)
                if res.ok:
                    runs[backend].append(res)
                else:
                    report.errors.append(f"{backend} round {rnd + 1}: {res.error}")

        for backend, rs in runs.items():
            if rs:
                report.cost[backend] = {
                    "docs_per_s": median([x.docs_per_s for x in rs]),
                    "query_p50_ms": median([x.query_p50_ms for x in rs]),
                    "load_s": median([x.load_s for x in rs]),
                    "peak_rss_mb": max(x.peak_rss_mb for x in rs),
                    "library": rs[0].library,
                }

        vec = Path(vectors_dir)
        ranked = {}
        for backend in ("onnx", "gguf"):
            docs_path = vec / f"{backend}_docs.npz"
            if docs_path.exists():
                ranked[backend] = rank(SparseMatrix(docs_path),
                                       SparseMatrix(vec / f"{backend}_queries.npz"), 100)
                report.retrieval[backend] = retrieval_metrics(ranked[backend], ds)
        if len(ranked) == 2:
            report.top10_agreement = ranking_agreement(ranked["onnx"], ranked["gguf"])
            report.fidelity_docs = fidelity(SparseMatrix(vec / "onnx_docs.npz"),
                                            SparseMatrix(vec / "gguf_docs.npz"))
            report.fidelity_queries = fidelity(SparseMatrix(vec / "onnx_queries.npz"),
                                               SparseMatrix(vec / "gguf_queries.npz"))

    write_markdown(report, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2),
                                                encoding="utf-8")
    print(f"report: {args.output}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
