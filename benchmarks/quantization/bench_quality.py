"""Retrieval quality by quantization variant (BEIR SciFact).

Speed and size of a quantized variant are measured by ``bench_quantization.py``;
this script answers the other half of the question: what does the variant cost
in retrieval quality?

For each dense text model that ships several registry entries (FP32 and its
FP16 / dynamic / static siblings), every variant embeds the SciFact corpus and
queries, documents are ranked by brute-force cosine similarity (numpy: this is
a benchmark, not library code), and the script reports nDCG@10, MRR@10 and
Recall@100. It also reports, per variant, the mean cosine between each document
embedding and the same document embedded by the FP32 reference (fidelity).

No timing is reported here: the variants run in one process, so nothing in this
report is a speed or memory claim.

A registry sibling is a different repository, hence possibly a different
export of the model (tokenizer, pooling, max length). The table therefore
measures "the registry entry a user gets", not "this quantizer alone".

Dataset: a BEIR folder (``corpus.jsonl``, ``queries.jsonl``, ``qrels/<split>.tsv``),
loaded with the same reader as ``benchmarks/sparse/bench_sparse_onnx_vs_gguf.py``.
Set ``LIBEMBEDDING_LIBRARY`` to the freshly built DLL when running from a source
checkout.

Usage::

    python benchmarks/quantization/bench_quality.py \\
        --models BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 \\
        --out benchmarks/quantization/quality_results

Auteur: David Orel
Version: 1.0.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))  # _benchmeta
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "sparse"))  # dataset loader

import _benchmeta as bm
from bench_sparse_onnx_vs_gguf import Dataset, load_beir
from libembedding import TextEmbedding
from libembedding.models import list_text_models

DEFAULT_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]
DEFAULT_DATASET = Path.home() / ".cache/libembedding/datasets/beir-scifact/scifact"

# Query instruction recommended by the model authors (applied to queries only).
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def registry_variants(base_name: str) -> list:
    """Registry entries for ``base_name``, FP32 ('none') first."""
    infos = [i for i in list_text_models() if i.model_name == base_name]
    infos.sort(key=lambda i: (i.quantization != "none", i.quantization, i.model_code))
    return infos


def embed_normalized(model: TextEmbedding, texts: list[str]) -> np.ndarray:
    """L2-normalized float32 embeddings, so a dot product is a cosine."""
    emb = np.asarray(model.embed(texts, normalized=True), dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def rank(docs: np.ndarray, queries: np.ndarray, depth: int) -> list[np.ndarray]:
    """Top ``depth`` document rows per query, by cosine."""
    scores = queries @ docs.T
    k = min(depth, docs.shape[0])
    out = []
    for row in scores:
        top = np.argpartition(-row, k - 1)[:k]
        out.append(top[np.argsort(-row[top], kind="stable")])
    return out


def retrieval_metrics(ranked: list[np.ndarray], ds: Dataset) -> dict[str, float]:
    ndcg, recall, mrr = [], [], []
    for qi, qid in enumerate(ds.query_ids):
        rel = ds.qrels[qid]
        got = [ds.doc_ids[d] for d in ranked[qi]]
        dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(r + 2) for r, d in enumerate(got[:10]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum((2**g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg else 0.0)
        recall.append(len(set(got[:100]) & set(rel)) / len(rel))
        mrr.append(next((1.0 / (r + 1) for r, d in enumerate(got[:10]) if d in rel), 0.0))
    return {
        "ndcg@10": float(np.mean(ndcg)),
        "mrr@10": float(np.mean(mrr)),
        "recall@100": float(np.mean(recall)),
    }


def evaluate_model(base_name: str, ds: Dataset, batch_size: int) -> list[dict]:
    """Evaluate every registry variant of ``base_name``; FP32 is the reference."""
    prefix = QUERY_PREFIX.get(base_name, "")
    queries = [prefix + q for q in ds.queries]
    rows: list[dict] = []
    ref_docs: np.ndarray | None = None
    for info in registry_variants(base_name):
        row = {
            "model": base_name,
            "model_code": info.model_code,
            "quantization": info.quantization,
        }
        try:
            model = TextEmbedding(
                info.model_code,
                quantization=info.quantization,
                provider="cpu",
                batch_size=batch_size,
                show_download_progress=False,
            )
            docs = embed_normalized(model, ds.docs)
            qs = embed_normalized(model, queries)
            del model
        except Exception as exc:  # noqa: BLE001 - a broken variant must not kill the report
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"  {info.model_code}: FAILED {row['error']}", flush=True)
            continue
        row.update(retrieval_metrics(rank(docs, qs, 100), ds))
        if ref_docs is None and info.quantization == "none":
            ref_docs = docs
        if ref_docs is not None and ref_docs.shape == docs.shape:
            row["cosine_vs_fp32"] = float(np.mean(np.sum(ref_docs * docs, axis=1)))
        else:
            row["cosine_vs_fp32"] = None
        rows.append(row)
        print(
            f"  {info.model_code} [{info.quantization}] ndcg@10={row['ndcg@10']:.4f} "
            f"mrr@10={row['mrr@10']:.4f} recall@100={row['recall@100']:.4f} "
            f"cos={row['cosine_vs_fp32']}",
            flush=True,
        )
    return rows


def results_table(rows: list[dict]) -> str:
    def fmt(v, digits=4):
        return "n/a" if v is None else f"{v:.{digits}f}"

    lines = [
        "| Model | Registry entry | Quantization | nDCG@10 | MRR@10 | Recall@100 | Cosine vs FP32 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(
                f"| {r['model']} | {r['model_code']} | {r['quantization']} | error | | | |"
            )
            continue
        lines.append(
            f"| {r['model']} | {r['model_code']} | {r['quantization']} "
            f"| {fmt(r['ndcg@10'])} | {fmt(r['mrr@10'])} | {fmt(r['recall@100'])} "
            f"| {fmt(r['cosine_vs_fp32'], 5)} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="registry model names")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="BEIR folder")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all judged queries")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "quality_results",
        help="output path without extension (writes .json and .md)",
    )
    args = ap.parse_args()

    ds = load_beir(args.dataset, args.split, args.max_docs, args.max_queries)
    dataset_label = (
        f"BEIR SciFact {args.split} ({len(ds.docs)} docs, {len(ds.queries)} queries)"
    )
    print(dataset_label, flush=True)

    rows: list[dict] = []
    for name in args.models:
        print(name, flush=True)
        rows += evaluate_model(name, ds, args.batch_size)

    run = bm.RunParams(
        threads=0,
        batch=args.batch_size,
        warmup=0,
        iterations=1,
        format="onnx",
        quantization="all registry variants",
        dataset=dataset_label,
    )
    report = bm.build_report(
        bm.collect_machine_metadata(),
        run,
        results={"variants": rows},
        title="Retrieval quality by quantization variant",
    )
    json_path = args.out.with_suffix(".json")
    md_path = args.out.with_suffix(".md")
    bm.write_json(report, json_path)
    md = [
        "# Retrieval quality by quantization variant",
        "",
        f"Dataset: {dataset_label}. Brute-force cosine, no timing claim.",
        (
            "Cosine vs FP32: mean cosine between a document embedding and the FP32 "
            "entry's embedding of the same document."
        ),
        "",
        results_table(rows),
        "",
        bm.to_markdown(report),
    ]
    md_path.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {json_path} and {md_path}")
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
