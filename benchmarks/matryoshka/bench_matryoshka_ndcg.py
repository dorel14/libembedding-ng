"""Matryoshka truncation: retrieval quality (nDCG@10) per dimension.

Embeds a BEIR dataset once at the model's native dimension, then scores the
same vectors truncated to each requested size (prefix + L2 renormalisation,
exactly what ``TextEmbedding(truncate_dim=k)`` and
``lembed_embeddings_truncate(emb, k, 1)`` return; the identity is pinned by
``python/tests/test_matryoshka.py``). Embedding once per model keeps the run
short and guarantees every dimension sees the same encoder output.

Quality only: nothing here is timed.

Dataset: any BEIR folder (``corpus.jsonl``, ``queries.jsonl``,
``qrels/<split>.tsv``), read with the loader of the sparse benchmark. SciFact::

    https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

Retrieval is exhaustive (one matrix product per dimension): an evaluation
detail, not a library feature.

Usage::

    python benchmarks/matryoshka/bench_matryoshka_ndcg.py \
        --beir ~/.cache/libembedding/datasets/beir-scifact/scifact \
        --model nomic-ai/nomic-embed-text-v1.5 \
        --model mixedbread-ai/mxbai-embed-large-v1

Auteur: David Orel
Version: 1.14.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sparse"))
from bench_sparse_onnx_vs_gguf import load_beir

# Prompts from each model card. Both models are trained with asymmetric
# query/document prefixes; leaving them out understates every dimension.
PROMPTS = {
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "mixedbread-ai/mxbai-embed-large-v1": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
}

DEFAULT_DIMS = (1024, 768, 512, 256, 128, 64)


def truncate(arr: np.ndarray, k: int) -> np.ndarray:
    out = arr[:, :k]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


def ndcg_at_k(
    scores: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> float:
    """Mean nDCG@k, linear gain (trec_eval / BEIR convention)."""
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    total = 0.0
    for qi, qid in enumerate(query_ids):
        rel = qrels[qid]
        dcg = sum(
            rel.get(doc_ids[d], 0) / math.log2(rank + 2)
            for rank, d in enumerate(top[qi])
        )
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
        total += dcg / idcg if idcg > 0 else 0.0
    return total / len(query_ids)


def embed_model(model_name: str, data, args) -> tuple[np.ndarray, np.ndarray, int]:
    from libembedding import TextEmbedding

    q_prefix, d_prefix = PROMPTS.get(model_name, ("", ""))
    model = TextEmbedding(
        model_name,
        offline=args.offline,
        show_download_progress=False,
        max_length=args.max_length,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    try:
        native = model.dim
        docs = model.embed([d_prefix + t for t in data.docs])
        queries = model.embed([q_prefix + t for t in data.queries])
    finally:
        model.close()
    return docs, queries, native


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beir", type=Path, required=True, help="BEIR dataset folder")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    ap.add_argument("--max-docs", type=int, default=0, help="0 = full corpus")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all queries")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--cache", type=Path, default=None,
                    help="folder to keep the full-size vectors between runs")
    ap.add_argument("--json", type=Path, default=None, help="write results here")
    args = ap.parse_args()

    data = load_beir(args.beir.expanduser(), args.split, args.max_docs, args.max_queries)
    print(f"{args.beir.name}: {len(data.docs)} docs, {len(data.queries)} queries "
          f"(split {args.split}, max_length {args.max_length})")

    results: dict[str, dict[str, float]] = {}
    for name in args.model:
        cached = None
        if args.cache:
            args.cache.mkdir(parents=True, exist_ok=True)
            cached = args.cache / (name.replace("/", "__") + ".npz")
        if cached and cached.is_file():
            z = np.load(cached)
            docs, queries, native = z["docs"], z["queries"], int(z["native"])
        else:
            docs, queries, native = embed_model(name, data, args)
            if cached:
                np.savez(cached, docs=docs, queries=queries, native=native)

        row: dict[str, float] = {}
        for k in sorted({d for d in args.dims if d <= native} | {native}, reverse=True):
            scores = truncate(queries, k) @ truncate(docs, k).T
            row[str(k)] = ndcg_at_k(scores, data.doc_ids, data.query_ids, data.qrels)
            print(f"  {name:<40} dim {k:>5}: nDCG@10 = {row[str(k)]:.4f}")
        results[name] = row

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
