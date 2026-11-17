"""Reference pipeline for per-model validation (T25).

Auteur: David Orel
Version: 1.14.0

SPDX-License-Identifier: MIT

The reference is the SAME ONNX file run with Python ``onnxruntime``, fed with
ids, attention mask and token types from HuggingFace ``tokenizers``, then pooled
and L2-normalised the way the registry entry says. Weights and runtime are
shared with libembedding, so the cosine against libembedding's output isolates
the tokenizer, the input packing and the pooling.

Run as a script to print the full report (all cached models)::

    python python/tests/model_reference.py [--json out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "tests" / "data" / "tokenizer" / "corpus.json"
MAX_LEN_CAP = 512  # bounds the cost of the 8192-token models on both sides

# Tokenizer family labels
WORDPIECE = "wordpiece"
UNIGRAM = "unigram"
BPE_BYTE = "bpe-bytelevel"
BPE_OTHER = "bpe-other"

# Verdict thresholds on the minimum cosine
OK_MIN = 0.999
DEGRADED_MIN = 0.99


def load_corpus() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["texts"]


def english_subset(texts: list[dict]) -> list[dict]:
    return [t for t in texts if t["script"] == "latin-ascii"]


def long_text() -> str:
    """A text well over 512 tokens, to exercise truncation."""
    base = (
        "The quick brown fox jumps over the lazy dog while the transformer "
        "model reads a long document about retrieval and ranking. "
    )
    return base * 60


def tokenizer_family(tokenizer_json: Path) -> str:
    cfg = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    mtype = cfg.get("model", {}).get("type", "")
    if mtype == "WordPiece":
        return WORDPIECE
    if mtype == "Unigram":
        return UNIGRAM
    if mtype == "BPE":
        pre = json.dumps(cfg.get("pre_tokenizer") or {})
        dec = json.dumps(cfg.get("decoder") or {})
        return BPE_BYTE if ("ByteLevel" in pre or "ByteLevel" in dec) else BPE_OTHER
    return mtype.lower() or "unknown"


def verdict(min_cos: float) -> str:
    if min_cos >= OK_MIN:
        return "validated"
    if min_cos >= DEGRADED_MIN:
        return "degraded"
    return "broken"


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def have_reference_deps() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------- registry


def registry_entries(kind: str):
    """Yield (index, ModelInfo) of a registry kind: text, sparse, reranker."""
    from libembedding import models

    lister = {
        "text": models.list_text_models,
        "sparse": models.list_sparse_models,
        "reranker": models.list_reranker_models,
    }[kind]
    return list(enumerate(lister()))


def model_dir(kind: str, index: int) -> Path | None:
    """Cached model directory of a registry entry, or None (never downloads)."""
    from libembedding._binding import ffi, lib

    ensure = {
        "text": lib.lembed_ensure_text_model,
        "sparse": getattr(lib, "lembed_ensure_sparse_model", None),
        "reranker": getattr(lib, "lembed_ensure_reranker_model", None),
    }[kind]
    if ensure is None:
        return None
    out = ffi.new("char **")
    status = ensure(index, ffi.NULL, 0, 1, out)
    if out[0] == ffi.NULL:
        return None
    path = ffi.string(out[0]).decode("utf-8")
    lib.lembed_free_string(out[0])
    return Path(path) if status == lib.LEMBED_OK else None


# --------------------------------------------------------------- reference


class Reference:
    """One ONNX file + HF tokenizer, run one text at a time (never padded)."""

    def __init__(self, directory: Path, model_file: str, max_len: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.no_padding()
        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        self.sess = ort.InferenceSession(
            str(directory / model_file), opts, providers=["CPUExecutionProvider"]
        )
        self.in_names = {i.name for i in self.sess.get_inputs()}
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def _feed(self, enc) -> dict:
        ids = np.array([enc.ids], dtype=np.int64)
        feed = {"input_ids": ids}
        if "attention_mask" in self.in_names:
            feed["attention_mask"] = np.array([enc.attention_mask], dtype=np.int64)
        if "token_type_ids" in self.in_names:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        return feed

    def _output_index(self) -> int:
        # Same precedence as select_output() in onnx_session_impl.hpp.
        if len(self.out_names) == 1:
            return 0
        for want in ("text_embeds", "last_hidden_state", "sentence_embedding"):
            if want in self.out_names:
                return self.out_names.index(want)
        return 0

    def dense(self, text: str, pooling: str) -> np.ndarray:
        enc = self.tok.encode(text)
        out = self.sess.run(
            [self.out_names[self._output_index()]], self._feed(enc)
        )[0]
        vec = out[0, 0] if (out.ndim == 3 and pooling == "cls") else None
        if vec is None:
            vec = out[0].mean(axis=0) if out.ndim == 3 else out[0]
        vec = vec.astype(np.float64)
        n = np.linalg.norm(vec)
        return (vec / n) if n > 0 else vec

    def sparse(self, text: str) -> dict[int, float]:
        enc = self.tok.encode(text)
        out = self.sess.run([self.out_names[0]], self._feed(enc))[0][0]
        scores = np.log1p(np.maximum(out, 0.0)).max(axis=0)
        return {int(i): float(scores[i]) for i in np.nonzero(scores)[0]}

    def pair_logit(self, query: str, doc: str) -> float:
        enc = self.tok.encode(query, doc)
        out = self.sess.run([self.out_names[0]], self._feed(enc))[0]
        return float(np.asarray(out).reshape(-1)[0])


def sparse_cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    return cosine(va, vb)


# -------------------------------------------------------------- measurement


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "mean": None}
    return {"min": float(min(values)), "mean": float(np.mean(values))}


def measure_dense(index: int, info, directory: Path, texts: list[dict]) -> dict:
    from libembedding import TextEmbedding

    max_len = min(info.max_tokens, MAX_LEN_CAP)
    ref = Reference(directory, info.model_file, max_len)
    items = [(t["id"], t["script"], t["text"]) for t in texts]
    items.append(("long", "latin-ascii-long", long_text()))
    emb = TextEmbedding(
        info.model_name,
        quantization=info.quantization,
        max_length=max_len,
        show_download_progress=False,
    )
    try:
        ours = emb.embed([x[2] for x in items])
    finally:
        emb.close()
    all_cos, en_cos, per = [], [], {}
    for (tid, script, text), vec in zip(items, ours):
        c = cosine(ref.dense(text, info.pooling), vec)
        per[tid] = c
        all_cos.append(c)
        if script.startswith("latin-ascii"):
            en_cos.append(c)
    return {"all": _stats(all_cos), "en": _stats(en_cos), "per_text": per}


def measure_sparse(index: int, info, directory: Path, texts: list[dict]) -> dict:
    from libembedding import SparseTextEmbedding

    ref = Reference(directory, info.model_file, min(info.max_tokens, MAX_LEN_CAP))
    items = [(t["id"], t["script"], t["text"]) for t in texts]
    emb = SparseTextEmbedding(
        info.model_name, max_length=min(info.max_tokens, MAX_LEN_CAP),
        show_download_progress=False,
    )
    try:
        ours = emb.embed([x[2] for x in items])
    finally:
        emb.close()
    all_cos, en_cos, per = [], [], {}
    for (tid, script, text), sv in zip(items, ours):
        mine = {int(i): float(v) for i, v in zip(sv.indices, sv.values)}
        c = sparse_cosine(ref.sparse(text), mine)
        per[tid] = c
        all_cos.append(c)
        if script.startswith("latin-ascii"):
            en_cos.append(c)
    return {"all": _stats(all_cos), "en": _stats(en_cos), "per_text": per}


RERANK_PAIRS = [
    ("what is the capital of france", "Paris is the capital of France."),
    ("what is the capital of france", "Berlin is a city in Germany."),
    ("comment fonctionne un transformeur", "Un transformeur utilise l'attention."),
    ("wie funktioniert ein Motor", "Ein Motor wandelt Energie in Bewegung um."),
    ("什么是机器学习", "机器学习是人工智能的一个分支。"),
    ("что такое Москва", "Москва является столицей России."),
    ("best pizza", "Pizza margherita is a classic neapolitan dish."),
    ("quantum computing", "The recipe needs two eggs and flour."),
]


def measure_reranker(index: int, info, directory: Path) -> dict:
    from libembedding import Reranker

    ref = Reference(directory, info.model_file, min(info.max_tokens, MAX_LEN_CAP))
    rr = Reranker(info.model_name, show_download_progress=False)
    try:
        ours_all, ref_all = [], []
        for q in sorted({p[0] for p in RERANK_PAIRS}):
            docs = [p[1] for p in RERANK_PAIRS if p[0] == q]
            res = rr.rerank(q, docs)
            by_doc = {docs[r.index]: r.score for r in res}
            for d in docs:
                ours_all.append(by_doc[d])
                ref_all.append(ref.pair_logit(q, d))
    finally:
        rr.close()
    o, r = np.array(ours_all), np.array(ref_all)
    # libembedding returns the raw logit, so compare in linear space.
    corr = float(np.corrcoef(o, r)[0, 1]) if len(o) > 2 else 1.0
    return {
        "pearson_vs_logit": corr,
        "max_abs_diff_vs_logit": float(np.max(np.abs(o - r))),
        "ours": o.tolist(),
        "ref_logit": r.tolist(),
    }


FIXTURE_FILE = Path(__file__).resolve().parent / "data" / "reference_embeddings.json"
# Registry entries whose ONNX reference is committed (fast, English-exact).
FIXTURE_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "none"),
    ("BAAI/bge-small-en-v1.5", "none"),
    ("snowflake/snowflake-arctic-embed-xs", "none"),
]
# Plain English texts: WordPiece ids are exact on them, so the committed vectors
# test the whole pipeline (tokenizer, packing, pooling, normalisation).
FIXTURE_TEXT_IDS = [
    "en-basic", "en-mixed-case", "en-sentence", "en-punct", "en-numbers", "en-compound",
]


def write_fixtures() -> None:
    """Regenerate reference_embeddings.json from the ONNX reference."""
    entries = {}
    texts = {t["id"]: t["text"] for t in load_corpus()}
    for idx, info in registry_entries("text"):
        if (info.model_name, info.quantization) not in FIXTURE_MODELS:
            continue
        d = model_dir("text", idx)
        if d is None:
            raise SystemExit(f"{info.model_name} is not cached")
        ref = Reference(d, info.model_file, min(info.max_tokens, MAX_LEN_CAP))
        entries[f"{info.model_name}|{info.quantization}"] = {
            "file": info.model_file,
            "pooling": info.pooling,
            "vectors": {
                i: [round(float(x), 5) for x in ref.dense(texts[i], info.pooling)]
                for i in FIXTURE_TEXT_IDS
            },
        }
    FIXTURE_FILE.parent.mkdir(exist_ok=True)
    FIXTURE_FILE.write_text(
        json.dumps({"_comment": "Reference embeddings: same ONNX file run with "
                    "onnxruntime and HF tokenizers (python/tests/model_reference.py "
                    "--fixtures). SPDX-License-Identifier: MIT. Auteur: David Orel.",
                    "models": entries}, indent=0) + "\n",
        encoding="utf-8", newline="\n",
    )


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT / "python" / "src"))
    if not have_reference_deps():
        print("onnxruntime / tokenizers missing")
        return 1
    if "--fixtures" in argv:
        write_fixtures()
        return 0
    if "--baselines" in argv:
        write_baselines(Path(argv[argv.index("--baselines") + 1]))
        return 0
    max_mb = 560  # 500 plus the cached SPLADE++ (532 MB)
    texts = load_corpus()
    report = []
    for kind in ("text", "sparse", "reranker"):
        for idx, info in registry_entries(kind):
            if info.model_file.endswith(".gguf"):
                continue
            d = model_dir(kind, idx)
            row = {"kind": kind, "name": info.model_name, "repo": info.model_code,
                   "file": info.model_file, "pooling": info.pooling,
                   "size_mb": info.download_size_mb}
            if d is None or not (d / info.model_file).is_file():
                row["status"] = "not-cached"
                report.append(row)
                continue
            if info.download_size_mb > max_mb:
                row["status"] = "too-large"
                report.append(row)
                continue
            try:
                row["family"] = tokenizer_family(d / "tokenizer.json")
                if kind == "text":
                    row.update(measure_dense(idx, info, d, texts))
                elif kind == "sparse":
                    row.update(measure_sparse(idx, info, d, texts))
                else:
                    row.update(measure_reranker(idx, info, d))
                row["status"] = "measured"
            except Exception as exc:  # noqa: BLE001 - report, keep going
                row["status"] = f"error: {type(exc).__name__}: {str(exc)[:160]}"
            report.append(row)
            print(row["kind"], row["name"], row["file"], row["status"],
                  row.get("family"), row.get("all"), flush=True)
    if "--json" in argv:
        out = Path(argv[argv.index("--json") + 1])
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


BASELINE_FILE = Path(__file__).resolve().parent / "data" / "model_baselines.json"
# English texts the tokenizer gets wrong on every WordPiece model (a word longer
# than 100 characters: HF emits [UNK], libembedding splits it). Tracked apart so
# that "English works" stays a testable claim. See the matrix in docs/models.md.
KNOWN_ENGLISH_GAP = "en-long-word"


def english_core(per_text: dict[str, float]) -> float:
    vals = [
        v for k, v in per_text.items()
        if (k.startswith("en-") and k != KNOWN_ENGLISH_GAP) or k == "long"
    ]
    return min(vals)


def write_baselines(report_path: Path) -> None:
    """Freeze a measured report as the regression baseline (rounded down)."""
    rows = {}
    for r in json.loads(report_path.read_text(encoding="utf-8")):
        if r["status"] != "measured":
            continue
        key = f"{r['kind']}|{r['name']}|{r['file']}"
        if r["kind"] == "reranker":
            rows[key] = {"family": r["family"],
                         "pearson": round(r["pearson_vs_logit"], 3)}
        else:
            rows[key] = {
                "family": r["family"],
                "min": round(r["all"]["min"], 4),
                "mean": round(r["all"]["mean"], 4),
                "en_core_min": round(english_core(r["per_text"]), 4),
            }
    BASELINE_FILE.parent.mkdir(exist_ok=True)
    BASELINE_FILE.write_text(
        json.dumps({"_comment": "Measured baseline of libembedding vs ONNX+HF "
                    "tokenizers (python/tests/model_reference.py). Regenerate with "
                    "--json then --baselines. SPDX-License-Identifier: MIT. "
                    "Auteur: David Orel.", "models": rows}, indent=1) + "\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
