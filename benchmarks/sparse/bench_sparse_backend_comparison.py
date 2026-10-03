"""
LE-9.6 — Four-backend comparison: Dense ONNX / Dense GGUF / Sparse ONNX / Sparse GGUF.

Measures the same corpus across every backend/type combination that libembedding
actually offers, and reports the ones it does not — with the reason — instead of
crashing.

Three rules drive the implementation, all inherited from
``benchmarks/quantization/bench_quantization.py`` and its documented lessons:

1. **Peak memory is a per-process high-water mark.** A process that already ran
   a previous backend can never report a lower peak for the next one, so the
   value cannot be attributed to a backend. One subprocess per backend is the
   only way to make the number mean anything. It also keeps a crashing or
   OOM-killed backend from taking the whole benchmark down.

2. **A missing backend is a result, not an error.** ``Sparse GGUF`` is reported
   as ``unavailable`` with the reason recorded in
   ``benchmarks/sparse/LE-9.4-validation.md``: llama.cpp v0.3.0 loads a BERT GGUF
   but never loads or executes the ``mlm_*`` projection head that SPLADE needs,
   so no ``[seq_len, vocab_size]`` logits tensor exists to post-process. The
   GGUF files themselves *do* carry the head, so a different runtime may well
   work — which is why the reason points at the P1.5a/P1.5b spike instead of
   claiming the concept is dead. Failing the whole run because one column is
   empty would hide the three columns that do work.

3. **Recall@K is computed where the vectors live.** Vectors cannot cross the
   subprocess boundary as JSON without dominating the benchmark's own memory, so
   the validation set travels *into* the worker and the recall numbers come back
   inside the result. Everything a row reports is therefore measured on the same
   vectors that were timed.

Two measurement caveats, stated here because they change how the numbers read:

- **Sparsity is capped by ``--top-k``.** With ``top_k=50`` a sparse row's ``nnz``
  cannot exceed 50, so it measures the truncation policy, not the model's natural
  density. The report states the active ``top_k`` next to every sparse row; run
  with ``--top-k 0`` to observe the untruncated density.
- **Sparsity is meaningless without a vocabulary size.** "0.8% of terms" says
  nothing about what it is a percentage *of*, so ``vocab_size`` is always reported
  alongside ``nnz``. Dense rows report 100% by convention.

Recall@K is only computed when a validation set is supplied
(``--recall-corpus``); without one those columns read ``n/a`` rather than being
filled with a fabricated number.

Author: David Orel
Version: 1.10.1

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

_RESULT_PREFIX = "__RESULT__ "
_REPORT_PATH = Path(__file__).resolve().parent / "LE-9.6-results.md"

# SPLADE++ and OpenSearch neural sparse both project onto the 30522-token BERT
# vocabulary. Dense vectors are fully populated, hence 100% by convention.
_VOCAB_SPARSE = 30522

UNAVAILABLE_SPARSE_GGUF = (
    "blocked, not measured: llama.cpp v0.3.0 loads a BERT GGUF but neither loads "
    "nor builds the mlm_* projection head SPLADE requires "
    "(third_party/llama.cpp/src/models/bert.cpp:23-74 and 227-232), so it never "
    "produces the [seq_len, vocab_size] logits. The GGUF files themselves do carry "
    "the head, so a different runtime may work -- see the P1.5a/P1.5b spike in "
    "benchmarks/sparse/LE-9.4-validation.md before concluding anything about "
    "sparse GGUF"
)


@dataclass(frozen=True)
class Backend:
    """One cell of the comparison matrix."""

    key: str
    kind: str  # "dense" | "sparse"
    backend: str  # "onnx" | "gguf"
    model: str
    quantization: str
    unavailable_reason: str | None = None


@dataclass
class BackendResult:
    """Outcome of measuring one Backend."""

    key: str
    kind: str
    backend: str
    model: str
    quantization: str
    status: str = "pending"  # pending | ok | unavailable | failed
    reason: str = ""
    load_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    ms_per_doc: float = 0.0
    docs_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    avg_nnz: float = 0.0
    vocab_size: int = 0
    sparsity_pct: float = 0.0
    dim: int = 0
    top_k: int = 0
    recall_at_k: dict[str, float] = field(default_factory=dict)
    warmup_docs: int = 0
    timed_docs: int = 0

    @property
    def label(self) -> str:
        return f"{self.kind.title()} {self.backend.upper()}"


def process_peak_mb() -> float:
    """Peak resident memory of *this* process, in MB.

    One backend per subprocess is required for this to mean anything: the
    high-water mark can only rise, so a process that already ran a previous
    backend can never report a lower peak for the next one.
    """
    if os.name == "nt":
        with contextlib.suppress(Exception):
            import ctypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            # GetProcessMemoryInfo lives in psapi.dll; kernel32 only forwards it.
            psapi = ctypes.windll.psapi
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400 | 0x0010, False, os.getpid())
            if handle:
                try:
                    counters = PROCESS_MEMORY_COUNTERS()
                    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    if psapi.GetProcessMemoryInfo(
                        handle, ctypes.byref(counters), counters.cb
                    ):
                        return counters.PeakWorkingSetSize / (1024 * 1024)
                finally:
                    kernel32.CloseHandle(handle)

    # ru_maxrss is a high-water mark on Linux (KiB) and on macOS (bytes).
    with contextlib.suppress(Exception):
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024

    return 0.0


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
    return s[n // 2]


def p95(values: list[float]) -> float:
    s = sorted(values)
    idx = int(0.95 * len(s)) - 1
    return s[max(0, min(idx, len(s) - 1))]


def generate_corpus(n: int, seed: int = 42) -> list[str]:
    """Generate n synthetic documents of varying length.

    Same shape as benchmarks/reranking/bench_common.py: a small fixed set of
    sentences repeated to the requested count, so every backend sees
    byte-identical input and the same corpus order.
    """
    import random

    random.seed(seed)
    base_docs = [
        "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Pizza is a traditional Italian dish made with dough, tomato sauce, and cheese.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "The Python programming language was created by Guido van Rossum in 1991.",
        "Quantum computing leverages quantum mechanical phenomena to perform computation.",
        "The Great Wall of China is a series of fortifications built over centuries.",
        "Natural language processing enables computers to understand human language.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
        "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
        "Shakespeare wrote 37 plays during his lifetime in Elizabethan England.",
        "CRISPR gene editing allows scientists to modify DNA sequences with precision.",
        "Electric vehicles are becoming more popular as battery technology improves.",
        "The theory of relativity changed our understanding of space, time, and gravity.",
        "Blockchain technology enables decentralized and transparent digital transactions.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The Internet of Things connects billions of devices to exchange data.",
        "Antibiotics revolutionized medicine by treating bacterial infections effectively.",
        "Virtual reality creates immersive digital environments for users to explore.",
    ]
    docs: list[str] = []
    for i in range(n):
        doc = base_docs[i % len(base_docs)]
        repeat = (i % 3) + 1
        docs.append((doc + " ") * repeat)
    return docs


def dense_scores(query_vec, doc_vecs) -> list[float]:
    """Cosine similarity between one query vector and every document vector."""
    import numpy as np

    q = np.asarray(query_vec, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    out = []
    for d in doc_vecs:
        dv = np.asarray(d, dtype=np.float32)
        denom = qn * float(np.linalg.norm(dv))
        out.append(0.0 if denom == 0.0 else float(np.dot(q, dv)) / denom)
    return out


def sparse_scores(query_emb, doc_embs) -> list[float]:
    """SPLADE dot product over the real vocabulary, not an approximation.

    Both operands are {token_id: weight} maps, so the dot product is the sum of
    the products over the intersection of their indices — exact, and therefore
    comparable across backends that emit different index sets.
    """
    out = []
    for d in doc_embs:
        q_idx = query_emb.indices
        q_val = query_emb.values
        d_idx = d.indices
        d_val = d.values
        total = 0.0
        i = j = 0
        while i < len(q_idx) and j < len(d_idx):
            a = int(q_idx[i])
            b = int(d_idx[j])
            if a == b:
                total += float(q_val[i]) * float(d_val[j])
                i += 1
                j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        out.append(total)
    return out


def recall_at_k(scores: list[float], relevant: set[int], k: int) -> float:
    """Percentage of relevant documents retrieved in the top-k."""
    if not relevant:
        return 0.0
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = sum(1 for idx in ranked[:k] if idx in relevant)
    return 100.0 * hits / len(relevant)


def resolve_model_path(
    model_name: str, offline: bool = False, cache_dir: str = ""
) -> str:
    """Resolve a registry name to a local file path.

    ONNX models are identified by name; GGUF models are identified by *file*, and
    ``TextEmbedding`` routes on the ``.gguf`` extension. So a GGUF registry name
    has to be turned into a path first — otherwise the run fails with a
    confusing "No text model matching 'MiniLM-L6-Q4'" instead of downloading
    the file the registry points at.
    """
    if model_name.lower().endswith(".gguf") or os.path.exists(model_name):
        return model_name

    from libembedding._binding import ffi, lib
    from libembedding._status import check_status

    info = lib.lembed_find_gguf_model(model_name.encode("utf-8"))
    if info == ffi.NULL:
        raise LookupError(f"No GGUF model matching {model_name!r} in the registry")

    url = ffi.string(info.gguf_url).decode("utf-8")
    marker = "/resolve/main/"
    if marker not in url:
        raise LookupError(f"Unsupported GGUF URL for {model_name!r}: {url}")

    repo, filename = url.split("huggingface.co/", 1)[1].split(marker, 1)
    cache_buf = ffi.new("char[]", cache_dir.encode("utf-8")) if cache_dir else ffi.NULL
    out = ffi.new("char **")
    check_status(
        lib.lembed_ensure_gguf_model(
            repo.encode("utf-8"),
            filename.encode("utf-8"),
            cache_buf,
            0,
            1 if offline else 0,
            out,
        )
    )
    try:
        return ffi.string(out[0]).decode("utf-8")
    finally:
        lib.lembed_free_string(out[0])


def run_worker(payload: dict) -> BackendResult:
    """Measure one backend in this process. Invoked as a subprocess by the driver."""
    from libembedding import SparseTextEmbedding, TextEmbedding

    kind = payload["kind"]
    model_name = payload["model"]
    batch_size = payload["batch_size"]
    iterations = payload["iterations"]
    warmup_docs = payload["warmup_docs"]
    top_k = payload["top_k"]
    queries = payload.get("queries") or []
    recall_k = payload.get("recall_k") or []

    timed_texts = payload["texts"][warmup_docs:]

    result = BackendResult(
        key=payload["key"],
        kind=kind,
        backend=payload["backend"],
        model=model_name,
        quantization=payload["quantization"],
        status="pending",
        warmup_docs=warmup_docs,
        timed_docs=len(timed_texts),
        top_k=top_k if kind == "sparse" else 0,
    )

    model = None
    try:
        t0 = time.perf_counter()
        load_target = model_name
        if payload["backend"] == "gguf":
            load_target = resolve_model_path(model_name, offline=payload.get("offline"))
        if kind == "dense":
            model = TextEmbedding(
                load_target,
                batch_size=batch_size,
                show_download_progress=False,
            )
        else:
            model = SparseTextEmbedding(
                load_target,
                batch_size=batch_size,
                show_download_progress=False,
                top_terms=top_k,
            )
        result.load_ms = (time.perf_counter() - t0) * 1000

        # Warmup runs on its own slice, never on the timed one.
        warm_slice = timed_texts[: min(8, len(timed_texts))]
        for _ in range(2):
            if kind == "dense":
                model.embed(warm_slice, batch_size=batch_size)
            else:
                model.embed(warm_slice, batch_size=batch_size, top_terms=top_k)

        timings = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            if kind == "dense":
                vectors = model.embed(timed_texts, batch_size=batch_size)
            else:
                embeddings = model.embed(
                    timed_texts, batch_size=batch_size, top_terms=top_k
                )
            timings.append((time.perf_counter() - t0) * 1000)

        result.median_ms = median(timings)
        result.p95_ms = p95(timings)
        result.ms_per_doc = result.median_ms / len(timed_texts)
        result.docs_per_sec = 1000.0 / result.ms_per_doc if result.ms_per_doc else 0.0
        result.peak_memory_mb = process_peak_mb()

        if kind == "dense":
            result.dim = int(vectors.shape[1]) if len(vectors) else 0
            result.vocab_size = 0
            result.avg_nnz = result.dim
            result.sparsity_pct = 100.0  # by convention: a dense vector is full
        else:
            result.vocab_size = _VOCAB_SPARSE
            result.avg_nnz = sum(len(e.indices) for e in embeddings) / max(
                1, len(embeddings)
            )
            result.sparsity_pct = 100.0 * result.avg_nnz / result.vocab_size

        if queries:
            for q in queries:
                if kind == "dense":
                    q_vecs = model.embed([q["text"]], batch_size=1)
                    scores = dense_scores(q_vecs[0], vectors)
                else:
                    q_embs = model.embed([q["text"]], batch_size=1, top_terms=top_k)
                    scores = sparse_scores(q_embs[0], embeddings)
                relevant = set(q.get("relevant", ()))
                for k in recall_k:
                    key = str(k)
                    prev = result.recall_at_k.get(key, 0.0)
                    result.recall_at_k[key] = prev + recall_at_k(scores, relevant, k)

            n_queries = len(queries)
            result.recall_at_k = {
                k: v / n_queries for k, v in result.recall_at_k.items()
            }

        result.status = "ok"
    except Exception as exc:  # noqa: BLE001 - cffi/ORT/llama raise many types
        result.status = "failed"
        result.reason = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            with contextlib.suppress(Exception):
                model.close()

    return result


def run_backend(
    backend_spec: Backend,
    texts: list[str],
    batch_size: int,
    iterations: int,
    warmup_docs: int,
    top_k: int,
    queries: list[dict],
    recall_k: list[int],
    offline: bool,
) -> BackendResult:
    """Dispatch one backend to a fresh subprocess, or short-circuit unavailable ones."""
    if backend_spec.unavailable_reason:
        return BackendResult(
            key=backend_spec.key,
            kind=backend_spec.kind,
            backend=backend_spec.backend,
            model=backend_spec.model,
            quantization=backend_spec.quantization,
            status="unavailable",
            reason=backend_spec.unavailable_reason,
            top_k=top_k if backend_spec.kind == "sparse" else 0,
        )

    payload = {
        "key": backend_spec.key,
        "kind": backend_spec.kind,
        "backend": backend_spec.backend,
        "model": backend_spec.model,
        "quantization": backend_spec.quantization,
        "batch_size": batch_size,
        "iterations": iterations,
        "warmup_docs": warmup_docs,
        "top_k": top_k,
        "texts": texts,
        "queries": queries,
        "recall_k": recall_k,
        "offline": offline,
    }
    proc = subprocess.run(  # fixed argv, no shell
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return BackendResult(**json.loads(line[len(_RESULT_PREFIX) :]))

    reason = f"worker exited with code {proc.returncode}"
    stderr = proc.stderr.strip()
    if stderr:
        reason += f": {stderr.splitlines()[-1]}"
    return BackendResult(
        key=backend_spec.key,
        kind=backend_spec.kind,
        backend=backend_spec.backend,
        model=backend_spec.model,
        quantization=backend_spec.quantization,
        status="failed",
        reason=reason,
        top_k=top_k if backend_spec.kind == "sparse" else 0,
    )


def load_recall_set(path: Path) -> list[dict]:
    """Load a validation set: {"queries": [{"text": str, "relevant": [int]}]}.

    ``relevant`` holds indices into the generated corpus, TREC style.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"queries": data}
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list) or not queries:
        raise ValueError(
            'recall corpus must be {"queries": [{"text": ..., "relevant": [...]}]}'
        )
    return queries


def source_version() -> str:
    """Version stamped in include/libembedding/config.h, or 'unknown'."""
    header = _REPO_ROOT / "include" / "libembedding" / "config.h"
    try:
        for line in header.read_text(encoding="utf-8").splitlines():
            if "LIBEMBEDDING_VERSION_STRING" in line and '"' in line:
                return line.split('"')[1]
    except OSError:
        pass
    return "unknown"


def git_revision() -> str:
    with contextlib.suppress(Exception):
        proc = subprocess.run(  # fixed argv, no shell
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    return "unknown"


def collect_environment() -> dict[str, object]:
    """Record the hardware and runtime identity alongside the numbers."""
    env: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "libembedding": source_version(),
        "git": git_revision(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return env


def write_report(
    results: list[BackendResult],
    env: dict[str, object],
    corpus_size: int,
    warmup_docs: int,
    top_k: int,
    batch_size: int,
    iterations: int,
    recall_k: list[int],
) -> None:
    """Produce LE-9.6-results.md in the same format as LE-8.10-results.md."""
    ok = [r for r in results if r.status == "ok"]
    unavailable = [r for r in results if r.status == "unavailable"]
    failed = [r for r in results if r.status == "failed"]

    lines: list[str] = []
    lines.append("# LE-9.6: Four-backend comparison — Dense/Sparse × ONNX/GGUF")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"- **{len(ok)}** backend(s) measured, **{len(unavailable)}** unavailable, "
        f"**{len(failed)}** failed."
    )
    for r in unavailable:
        lines.append(f"- **{r.label}** is unavailable — {r.reason}")
    if unavailable:
        lines.append(
            "- Consequently there is no sparse GGUF column: libembedding's sparse path "
            "is ONNX-only today. This is a *blocked* state, not a closed door — the "
            "P1.5a/P1.5b spike decides whether a different runtime changes it "
            "(`benchmarks/sparse/LE-9.4-validation.md`)."
        )
    lines.append("")

    lines.append("## 1. Objectif")
    lines.append("")
    lines.append(
        "Compare the four backend×type combinations of LE-9.6 on one corpus: "
        "throughput, latency, memory, sparsity and Recall@K."
    )
    lines.append("")

    lines.append("## 2. Environment")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    for k, v in env.items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"| corpus (timed) | {corpus_size} docs |")
    lines.append(f"| warmup (untimed) | {warmup_docs} docs |")
    lines.append(f"| batch_size | {batch_size} |")
    lines.append(f"| timed iterations | {iterations} (median reported) |")
    lines.append(f"| top_k | {top_k if top_k else 'untruncated'} |")
    lines.append("")

    lines.append("## 3. Results")
    lines.append("")
    header = (
        "| Backend | Model | Quant. | Top-K | Load (ms) | docs/s | ms/doc | "
        "Peak RAM (MB) | nnz | vocab | Sparsity (%) |"
    )
    if recall_k:
        header += " " + " | ".join(f"R@{k}" for k in recall_k) + " |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" * len(recall_k)))
    for r in results:
        if r.status == "ok":
            top_k_cell = str(r.top_k) if r.kind == "sparse" else "—"
            vocab_cell = str(r.vocab_size) if r.vocab_size else "—"
            cells = [
                r.label,
                f"`{r.model}`",
                r.quantization,
                top_k_cell,
                f"{r.load_ms:.0f}",
                f"{r.docs_per_sec:.1f}",
                f"{r.ms_per_doc:.2f}",
                f"{r.peak_memory_mb:.0f}",
                f"{r.avg_nnz:.0f}",
                vocab_cell,
                f"{r.sparsity_pct:.3f}",
            ]
            for k in recall_k:
                value = r.recall_at_k.get(str(k))
                cells.append("n/a" if value is None else f"{value:.1f}")
            lines.append("| " + " | ".join(cells) + " |")
        else:
            cells = [
                r.label,
                f"`{r.model}`",
                r.quantization,
                "—",
                "n/a" if r.status == "unavailable" else "ERROR",
            ] + ["n/a"] * (6 + len(recall_k))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 4. How to read these numbers")
    lines.append("")
    lines.append(
        "- **Peak RAM** is a per-backend high-water mark: each backend ran in its "
        "own subprocess, so the value is attributable to that backend alone."
    )
    lines.append(
        f"- **nnz** for a sparse row is capped by `--top-k` ({top_k}). It measures "
        "the truncation policy, not the model's natural density. Re-run with "
        "`--top-k 0` for the untruncated figure."
        if top_k
        else "- **nnz** is untruncated (`--top-k 0`), so it reflects the model's "
        "natural density."
    )
    lines.append(
        "- **Sparsity** is `nnz / vocab_size`. Dense rows are 100% by convention "
        "and carry no vocabulary."
    )
    if not recall_k:
        lines.append(
            "- **Recall@K** was not computed (no `--recall-corpus` supplied). The "
            "comparison then covers throughput, memory and sparsity only."
        )
    else:
        lines.append(
            "- **Recall@K** is computed on the same vectors that were timed, "
            "inside the worker subprocess. Sparse backends use the true SPLADE dot "
            "product over the vocabulary; dense backends use cosine similarity."
        )
    lines.append("")

    lines.append("## 5. Backends not measured")
    lines.append("")
    if not unavailable and not failed:
        lines.append("None — every backend in the matrix was measured.")
        lines.append("")
    for r in unavailable:
        lines.append(f"### {r.label} — `{r.model}`")
        lines.append("")
        lines.append(f"Reason: {r.reason}")
        lines.append("")
    for r in failed:
        lines.append(f"### {r.label} — `{r.model}`")
        lines.append("")
        lines.append(f"Error: `{r.reason}`")
        lines.append("")

    # Explicit newline: the repo is LF, and Path.write_text() would translate to
    # CRLF on Windows, churning the diff on every regeneration. Not using the
    # newline= argument because Path.write_text only grew it in Python 3.10 and the
    # project supports 3.9.
    with _REPORT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nReport written: {_REPORT_PATH}")


def build_matrix(args) -> list[Backend]:
    matrix = [
        Backend("dense_onnx", "dense", "onnx", args.dense_onnx, "fp32"),
        Backend("dense_gguf", "dense", "gguf", args.dense_gguf, "q4_k_m"),
        Backend("sparse_onnx", "sparse", "onnx", args.sparse_onnx, "fp32"),
        Backend(
            "sparse_gguf",
            "sparse",
            "gguf",
            args.sparse_gguf,
            "q4_k_m",
            unavailable_reason=UNAVAILABLE_SPARSE_GGUF,
        ),
    ]
    selected = {s.strip() for s in args.backends.split(",") if s.strip()}
    if selected:
        matrix = [b for b in matrix if b.key in selected]
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="LE-9.6 four-backend comparison")
    parser.add_argument("--dense-onnx", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-gguf", default="MiniLM-L6-Q4")
    parser.add_argument("--sparse-onnx", default="prithivida/Splade_PP_en_v1")
    parser.add_argument("--sparse-gguf", default="SPLADE-PP-En-v1")
    parser.add_argument(
        "--backends",
        default="",
        help="Comma-separated subset of dense_onnx,dense_gguf,sparse_onnx,sparse_gguf",
    )
    parser.add_argument("--num-texts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Sparse truncation (0 = untruncated). Caps the nnz column.",
    )
    parser.add_argument(
        "--recall-corpus",
        default="",
        help='JSON validation set: {"queries": [{"text": ..., "relevant": [ids]}]}',
    )
    parser.add_argument("--recall-k", default="20,50,100")
    parser.add_argument(
        "--offline", action="store_true", help="Use cached models only, never download"
    )
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )  # internal: one backend per subprocess

    args = parser.parse_args()

    if args.worker:
        result = run_worker(json.loads(sys.stdin.read()))
        print(_RESULT_PREFIX + json.dumps(asdict(result)))
        return 0

    if args.num_texts < 2:
        parser.error("--num-texts must be at least 2")

    corpus = generate_corpus(args.num_texts)
    warmup_docs = max(1, len(corpus) // 20)

    queries: list[dict] = []
    recall_k: list[int] = []
    if args.recall_corpus:
        queries = load_recall_set(Path(args.recall_corpus))
        recall_k = [int(x) for x in args.recall_k.split(",") if x.strip()]

    env = collect_environment()
    print("=" * 80)
    print("LE-9.6 Four-backend comparison")
    print("=" * 80)
    print(f"Corpus: {len(corpus) - warmup_docs} timed (+{warmup_docs} warmup)")
    print(f"Batch: {args.batch_size}  Iterations: {args.iterations}  Top-K: {args.top_k}")
    if recall_k:
        print(f"Recall@{recall_k} on {len(queries)} labelled queries")
    print(f"Environment: {env}")

    results: list[BackendResult] = []
    for backend_spec in build_matrix(args):
        print("-" * 80)
        if backend_spec.unavailable_reason:
            print(f"  SKIPPED   {backend_spec.key}: unavailable")
        else:
            print(f"  MEASURING {backend_spec.key} ({backend_spec.model})")
        result = run_backend(
            backend_spec,
            corpus,
            args.batch_size,
            args.iterations,
            warmup_docs,
            args.top_k,
            queries,
            recall_k,
            args.offline,
        )
        results.append(result)
        if result.status == "ok":
            if result.vocab_size:
                density = f"nnz {result.avg_nnz:.0f} ({result.sparsity_pct:.3f}% of {result.vocab_size})"
            else:
                density = f"dim {result.dim} (dense, no vocabulary)"
            print(
                f"    -> {result.docs_per_sec:.1f} docs/s, "
                f"{result.ms_per_doc:.2f} ms/doc, "
                f"peak {result.peak_memory_mb:.0f} MB, {density}"
            )
        else:
            print(f"    -> {result.status}: {result.reason}")

    write_report(
        results,
        env,
        len(corpus) - warmup_docs,
        warmup_docs,
        args.top_k,
        args.batch_size,
        args.iterations,
        recall_k,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())