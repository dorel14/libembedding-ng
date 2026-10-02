"""Benchmark quantization modes for ONNX embedding models.

Compares the registry variants of a model: the FP32 entry and its quantized
siblings (``_q`` / ``-Q`` model codes).

Two rules drive the implementation, both learned the hard way:

1. **The quantization mode is only real when the model *file* differs.** The
   ``quantization=`` argument of ``TextEmbedding`` does not change which file is
   loaded: it selects session options, and on an FP32 graph those options do
   nothing. Asking for ``quantization="static"`` on the FP32 entry benchmarks the
   same FP32 weights under a different label. This benchmark therefore selects
   the registry entry (hence the file) and passes the tag the registry records
   for it.

2. **Peak memory is a per-process high-water mark.** A process that has already
   run a previous configuration can never report a lower peak for the next one,
   so the value cannot be attributed to a configuration. One configuration per
   subprocess is the only way to make it meaningful. It also keeps a crashing or
   OOM-killed configuration from taking the whole benchmark down.

A third rule came out of the results themselves: **the registry tag is not proof
of the weight format.** The four ``Qdrant/*-onnx-Q`` entries were labelled
``static`` (INT8) and described as "Quantized", but every initializer in
``model_optimized.onnx`` is ``FLOAT16`` and the graph is an ORT-optimized one
(``Attention`` / ``SkipLayerNormalization`` / ``FastGelu`` fused). They are FP16,
not INT8, and on a CPU without native FP16 arithmetic they are slower than FP32
while being smaller. The registry now declares them ``fp16``; run
``python benchmarks/verify_quantization_modes.py`` to re-check every entry
against the dtypes actually present in its file.

Every configuration also records the ONNX Runtime version, the execution provider
and the initializer dtypes found in its weights file, and the run refuses to be
quiet about a mismatch: if two configurations ran on different runtimes or
providers, the report says so instead of publishing numbers nobody can compare.

Auteur: David Orel
Version: 1.8.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import html
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

# Add repo root to path for benchmarks
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

try:
    import psutil
except ImportError:
    psutil = None

from libembedding import TextEmbedding  # E402: needs the sys.path entry above
from libembedding.models import list_text_models  # E402: same

# Marker used to find the worker's result in its stdout.
_RESULT_PREFIX = "@@RESULT@@"


@dataclass
class ModelVariant:
    """One registry entry: a distinct set of weights."""

    base_name: str
    model_code: str
    quantization: str
    model_file: str
    note: str = ""


@dataclass
class QuantizationResult:
    model_name: str
    model_code: str
    quantization: str
    batch_size: int
    texts_per_second: float
    latency_ms_per_doc: float
    weights_mb: float
    process_peak_mb: float
    dim: int
    num_docs: int
    success: bool
    error: str = ""
    batched: bool = True
    # Runtime identity, recorded per configuration so a run that silently mixed
    # ONNX Runtime versions can be detected instead of published.
    onnxruntime: str = ""
    execution_provider: str = ""
    # Initializer dtypes found in the weights file, e.g. {"INT8": 144, ...}.
    # Empty when the optional 'onnx' package is unavailable.
    weight_types: dict[str, int] = field(default_factory=dict)


@dataclass
class SkippedVariant:
    model_name: str
    quantization: str
    reason: str


@dataclass
class Report:
    results: list[QuantizationResult] = field(default_factory=list)
    skipped: list[SkippedVariant] = field(default_factory=list)


def _model_dir(model_code: str) -> Path:
    """Directory the library caches `model_code` in.

    Mirrors get_cache_dir() / repo_to_dirname() in
    detail/downloader_impl.hpp: an explicit LIBEMBEDDING_CACHE_DIR or
    FASTEMBED_CACHE_DIR wins, otherwise ~/.cache/libembedding, and the repo
    slug replaces each '/' with a single '-'. Hard-coding a different layout
    here would silently report 0 MB for every model.
    """
    root = (
        os.environ.get("LIBEMBEDDING_CACHE_DIR")
        or os.environ.get("FASTEMBED_CACHE_DIR")
        or str(Path.home() / ".cache" / "libembedding")
    )
    return Path(root) / f"models--{model_code.replace('/', '-')}"


def cached_model_path(model_code: str, model_file: str = "") -> Path:
    """Locate a registry entry's graph file inside the local cache.

    Returns a path that may not exist; weight_types() reports an empty dict for
    an unreadable file rather than guessing.
    """
    model_dir = _model_dir(model_code)
    candidates = [model_dir / model_file] if model_file else []
    candidates += [model_dir / "onnx" / "model.onnx", model_dir / "model.onnx"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(model_dir.rglob("*.onnx")) if model_dir.exists() else []
    return matches[0] if matches else model_dir / "model.onnx"


def model_weights_mb(model_code: str, model_file: str) -> float:
    """Size on disk of the weights for a registry entry, in MB.

    This is what a download costs. It is not the same as resident memory: ONNX
    Runtime memory-maps the file and allocates its own arena on top, so
    process_peak_mb() is larger than the file and includes a fixed overhead.
    Both are reported because both matter. External-data models ship their
    weights in a sibling "<name>_data" file, so every sibling of the main file
    is counted.
    """
    model_dir = _model_dir(model_code)
    main = model_dir / model_file
    if not main.is_file():
        return 0.0
    total = main.stat().st_size
    stem = main.name.split(".")[0]
    for sibling in main.parent.glob(f"{stem}*"):
        if sibling.is_file() and sibling != main:
            total += sibling.stat().st_size
    return total / (1024 * 1024)


def registry_variants(base_name: str) -> list[ModelVariant]:
    """Return every registry entry that provides ``base_name``.

    The registry lists a quantized sibling as its own entry with its own model
    code, so the tags tell us which modes actually exist for this model.
    """
    out = []
    for info in list_text_models():
        if info.model_name == base_name:
            out.append(
                ModelVariant(
                    base_name=base_name,
                    model_code=info.model_code,
                    quantization=info.quantization,
                    model_file=info.model_file,
                )
            )
    return out


def process_peak_mb() -> float:
    """Peak resident memory of *this* process, in MB.

    One configuration per subprocess is required for this to mean anything: the
    high-water mark can only rise, so a process that already ran a previous
    configuration can never report a lower peak for the next one. It also keeps
    a crashing or OOM-killed configuration from taking the benchmark down.
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

            # GetProcessMemoryInfo lives in psapi.dll (kernel32 only forwards it).
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

    # Last resort: instantaneous RSS. Sampled after the run, so it is a lower
    # bound, not a peak. Still better than reporting 0.
    if psutil is not None:
        with contextlib.suppress(Exception):
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return 0.0


def run_single_configuration(
    variant: ModelVariant,
    texts: list[str],
    batch_size: int | None,
) -> QuantizationResult:
    """Measure one configuration. Runs in a dedicated process; see above."""
    num_docs = len(texts)
    dim = 0
    batched = batch_size is not None
    weights = model_weights_mb(variant.model_code, variant.model_file)

    def failure(message: str) -> QuantizationResult:
        return QuantizationResult(
            model_name=variant.base_name,
            model_code=variant.model_code,
            quantization=variant.quantization,
            batch_size=batch_size or 0,
            texts_per_second=0.0,
            latency_ms_per_doc=0.0,
            weights_mb=weights,
            process_peak_mb=process_peak_mb(),
            dim=dim,
            num_docs=0,
            success=False,
            error=message,
            batched=batched,
        )

    try:
        # Both are required: model_code selects the weights, quantization selects
        # the session options that keep the QDQ nodes in the graph.
        model = TextEmbedding(
            variant.model_code,
            quantization=variant.quantization,
            provider="cpu",
            batch_size=batch_size if batched else 8,
            show_download_progress=False,
        )
    except Exception as exc:  # noqa: BLE001 - a benchmark records any failure
        return failure(str(exc))

    try:
        dim = model.dim
        # One warmup pass on the same instance that is measured: warming a
        # throwaway instance would warm nothing, since model load, lazy init and
        # the thread pool are all per instance.
        model.embed(texts[: min(32, num_docs)])

        t0 = time.perf_counter()
        if batched:
            model.embed(texts, batch_size=batch_size)
        else:
            model.embed(texts)
        t1 = time.perf_counter()

        peak = process_peak_mb()
        elapsed_ms = (t1 - t0) * 1000
        docs_per_sec = num_docs / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
        latency_ms = elapsed_ms / num_docs if num_docs > 0 else 0.0
    except Exception as exc:  # noqa: BLE001 - a benchmark records any failure
        return failure(str(exc))
    finally:
        with contextlib.suppress(Exception):
            model.close()

    return QuantizationResult(
        model_name=variant.base_name,
        model_code=variant.model_code,
        quantization=variant.quantization,
        batch_size=batch_size or 0,
        texts_per_second=docs_per_sec,
        latency_ms_per_doc=latency_ms,
        weights_mb=weights,
        process_peak_mb=peak,
        dim=dim,
        num_docs=num_docs,
        success=True,
        batched=batched,
    )


def benchmark_configuration(
    variant: ModelVariant,
    texts: list[str],
    batch_size: int | None,
) -> QuantizationResult:
    """Run one configuration in a fresh process and return its result."""
    payload = {
        "model_code": variant.model_code,
        "quantization": variant.quantization,
        "model_file": variant.model_file,
        "base_name": variant.base_name,
        "texts": texts,
        "batch_size": batch_size,
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
            result = QuantizationResult(**json.loads(line[len(_RESULT_PREFIX) :]))
            result.model_name = variant.base_name
            result.batch_size = batch_size or 0
            result.batched = batch_size is not None
            return result

    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    reason = detail[-1] if detail else f"worker exited with code {proc.returncode}"
    return QuantizationResult(
        model_name=variant.base_name,
        model_code=variant.model_code,
        quantization=variant.quantization,
        batch_size=batch_size or 0,
        texts_per_second=0.0,
        latency_ms_per_doc=0.0,
        weights_mb=model_weights_mb(variant.model_code, variant.model_file),
        process_peak_mb=0.0,
        dim=0,
        num_docs=0,
        success=False,
        error=f"worker failed: {reason}",
        batched=batch_size is not None,
    )


def generate_corpus(n: int = 1000) -> list[str]:
    """Generate a representative corpus of varied text lengths."""
    templates = [
        "The quick brown fox jumps over the lazy dog.",
        (
            "Machine learning is a subset of artificial intelligence that focuses on "
            "building systems that learn from data."
        ),
        (
            "Embeddings are dense vector representations of text that capture semantic "
            "meaning and can be used for similarity search, classification, and "
            "clustering tasks in natural language processing applications."
        ),
        (
            "The transformer architecture, introduced in the paper 'Attention Is All You "
            "Need', revolutionized natural language processing by using self-attention "
            "mechanisms to process sequential data in parallel."
        ),
        (
            "Retrieval-augmented generation (RAG) combines the benefits of parametric and "
            "non-parametric memory to produce more accurate and factual responses by "
            "retrieving relevant information from a knowledge base before generating a "
            "response."
        ),
    ]
    return [f"[{i}] {templates[i % len(templates)]}" for i in range(n)]


def format_justification(
    baseline_docs_per_sec: float,
    variant_docs_per_sec: float,
    baseline_peak: float,
    variant_peak: float,
) -> str:
    """Compare a variant to its baseline on speed and resident memory."""
    if baseline_docs_per_sec <= 0:
        return "Baseline failed"

    parts = []
    if variant_docs_per_sec > 0:
        speed_ratio = variant_docs_per_sec / baseline_docs_per_sec
        if speed_ratio > 1.05:
            parts.append(f"{speed_ratio:.1f}x faster")
        elif speed_ratio < 0.95:
            parts.append(f"{1 / speed_ratio:.1f}x slower")
        else:
            parts.append("similar speed")
    else:
        parts.append("no throughput")

    # A missing measurement (0.0) must not be reported as a comparison.
    if baseline_peak > 0 and variant_peak > 0:
        mem_ratio = variant_peak / baseline_peak
        if mem_ratio < 0.95:
            parts.append(f"{1 / mem_ratio:.1f}x less RAM")
        elif mem_ratio > 1.05:
            parts.append(f"{mem_ratio:.1f}x more RAM")
        else:
            parts.append("similar RAM")
    else:
        parts.append("RAM not measured")

    return ", ".join(parts)


def baseline_by_batch(
    results: list[QuantizationResult],
) -> dict[tuple[str, int], QuantizationResult]:
    """Index the unquantized baseline *per batch size*.

    A single baseline is not enough: comparing a batch-32 variant against a
    batch-8 baseline measures the batch size, not the quantization.
    """
    return {
        (r.model_name, r.batch_size): r
        for r in sorted(results, key=lambda x: (x.batch_size, x.quantization))
        if r.quantization == "none" and r.success
    }


def _row(r: QuantizationResult, justification: str) -> str:
    batch = r.batch_size if r.batched else "n/a"
    return (
        f"    <tr><td>{batch}</td>"
        f"<td>{html.escape(r.quantization)}</td>"
        f"<td>{html.escape(r.model_code)}</td>"
        f"<td>{r.texts_per_second:.1f}</td>"
        f"<td>{r.latency_ms_per_doc:.2f}</td>"
        f"<td>{r.weights_mb:.1f}</td>"
        f"<td>{r.process_peak_mb:.0f}</td>"
        f"<td class=\"justification\">{html.escape(justification)}</td></tr>"
    )


def write_html_report(
    results: list[QuantizationResult],
    skipped: list[SkippedVariant],
    env: dict,
    output_path: Path,
) -> None:
    """Write an HTML report with per-batch-size comparison tables."""
    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Quantization Benchmark Results</title>",
        "    <style>",
        "        body { font-family: monospace; margin: 20px; }",
        "        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
        "        th { background-color: #f2f2f2; text-align: center; }",
        "        tr:nth-child(even) { background-color: #f9f9f9; }",
        "        .justification { font-size: 0.9em; color: #555; max-width: 300px; }",
        "        h1, h2 { color: #333; }",
        "        .env { font-size: 0.85em; color: #555; margin-bottom: 20px; }",
        "        .note { font-size: 0.85em; color: #a00; margin-bottom: 20px; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Quantization Benchmark Results</h1>",
        '    <div class="env">'
        + html.escape(", ".join(f"{k}: {v}" for k, v in env.items()))
        + "</div>",
        "    <div class=\"note\">"
        + html.escape(
            "Weights MB is the model file on disk (what the download costs). "
            "Peak MB is the resident high-water mark of a process that ran "
            "exactly this one configuration: it is larger than the file because "
            "ONNX Runtime maps the file and allocates its own arena on top."
        )
        + "</div>",
    ]

    for model_name in sorted({r.model_name for r in results if r.success}):
        model_results = [r for r in results if r.model_name == model_name and r.success]
        baselines = baseline_by_batch(model_results)

        parts.append(f"    <h2>{html.escape(model_name)}</h2>")
        parts.append("    <table>")
        parts.append(
            "    <tr><th>Batch Size</th><th>Quantization</th><th>Model code</th>"
            "<th>docs/s</th><th>ms/doc</th><th>Weights MB</th><th>Process peak MB</th>"
            "<th>Justification</th></tr>"
        )

        for r in sorted(model_results, key=lambda x: (x.batch_size, x.quantization)):
            just = ""
            baseline = baselines.get((r.model_name, r.batch_size))
            if baseline is not None and r.quantization != "none":
                just = format_justification(
                    baseline.texts_per_second,
                    r.texts_per_second,
baseline.process_peak_mb,
                r.process_peak_mb,
                )
            parts.append(_row(r, just))

        parts.append("    </table>")

    if skipped:
        parts.append("    <h2>Not measured</h2>")
        parts.append("    <table>")
        parts.append("    <tr><th>Model</th><th>Quantization</th><th>Reason</th></tr>")
        for s in skipped:
            parts.append(
                f"    <tr><td>{html.escape(s.model_name)}</td>"
                f"<td>{html.escape(s.quantization)}</td>"
                f"<td>{html.escape(s.reason)}</td></tr>"
            )
        parts.append("    </table>")

    failed = [r for r in results if not r.success]
    if failed:
        parts.append("    <h2>Failures</h2>")
        parts.append("    <table>")
        parts.append(
            "    <tr><th>Model</th><th>Code</th><th>Quantization</th><th>Batch</th>"
            "<th>Error</th></tr>"
        )
        for r in failed:
            parts.append(
                f"    <tr><td>{html.escape(r.model_name)}</td>"
                f"<td>{html.escape(r.model_code)}</td>"
                f"<td>{html.escape(r.quantization)}</td>"
                f"<td>{r.batch_size if r.batched else 'n/a'}</td>"
                f"<td>{html.escape(r.error)}</td></tr>"
            )
        parts.append("    </table>")

    parts.append("</body>")
    parts.append("</html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def collect_environment() -> dict:
    """Describe the conditions the results were produced under.

    Without this, two runs cannot be compared: a machine under load makes the
    baseline slower and every "faster" claim that follows is unfounded.
    """
    version = "unknown"
    with contextlib.suppress(Exception):
        import libembedding

        version = getattr(libembedding, "__version__", "unknown")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": _cpu_name(),
        "python": platform.python_version(),
        # The installed distribution's version. It can differ from the source
        # tree: this benchmark puts python/src ahead of the installed package,
        # so the code running is the tree, not the distribution.
        "libembedding_dist": version,
        "libembedding_source": _source_version(),
        # Quantized throughput depends on the runtime as much as on the weights:
        # the QDQ fusion that turns DequantizeLinear + MatMulInteger into
        # QLinearMatMul is a runtime feature, so two runs of the same model on
        # different ORT versions are not comparable. Recorded per configuration
        # and cross-checked at the end; see verify_runtime_consistency().
        "onnxruntime": _onnxruntime_version(),
        "execution_provider": _execution_provider(),
    }


def _cpu_name() -> str:
    """CPU model string, so 'measured on an X' claims can be checked.

    Reads the Windows registry rather than shelling out to wmic: wmic is absent
    from recent Windows 11 builds, and the fallback (platform.processor) yields
    "Intel64 Family 6 Model 126 Stepping 5", which is not the name a reader can
    look up.
    """
    if platform.system() == "Windows":
        try:
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                name, _ = winreg.QueryValueEx(handle, "ProcessorNameString")
                if name:
                    return str(name).strip()
        except (OSError, ImportError):
            pass
    return platform.processor() or "unknown"


def _onnxruntime_version() -> str:
    """ONNX Runtime version the library is linked against."""
    with contextlib.suppress(Exception):
        from libembedding import _binding

        return _binding.ffi.string(_binding.lib.lembed_onnxruntime_version()).decode("utf-8")
    return "unknown"


def _execution_provider() -> str:
    """Which execution provider actually served the embeddings."""
    with contextlib.suppress(Exception):
        from libembedding import _binding

        return _binding.ffi.string(_binding.lib.lembed_execution_provider_name(0)).decode("utf-8")
    return "unknown"


def weight_types(path) -> dict[str, int]:
    """Count initializer tensors per weight dtype in an ONNX file.

    This is the check that catches a file whose name says "quantized" and whose
    registry tag says INT8 while every initializer is FLOAT16. Returns an empty
    dict when the optional 'onnx' package is unavailable, so the benchmark still
    runs without it.
    """
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        return {}

    names = {v: k for k, v in TensorProto.DataType.items()}
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception:  # noqa: BLE001 - a unreadable file is reported as unknown
        return {}
    counts: dict[str, int] = {}
    for init in model.graph.initializer:
        name = names.get(init.data_type, f"?{init.data_type}")
        counts[name] = counts.get(name, 0) + 1
    return counts


def verify_runtime_consistency(results: list[QuantizationResult]) -> list[str]:
    """Every configuration must have run on the same runtime.

    A benchmark that silently mixes ONNX Runtime versions cannot be compared to
    anything, including itself. Returns a list of problems; empty means clean.
    """
    problems: list[str] = []
    versions = {r.onnxruntime for r in results if r.success and r.onnxruntime}
    if len(versions) > 1:
        problems.append(
            "ONNX Runtime differs between configurations: "
            + ", ".join(sorted(versions))
            + " -- these results are not comparable"
        )
    providers = {r.execution_provider for r in results if r.success and r.execution_provider}
    if len(providers) > 1:
        problems.append(
            "Execution provider differs between configurations: "
            + ", ".join(sorted(providers))
        )
    unknown = {
        r.onnxruntime
        for r in results
        if r.success and (not r.onnxruntime or r.onnxruntime == "unknown")
    }
    if unknown:
        problems.append(
            "ONNX Runtime version could not be read for some configurations; "
            "record it manually before publishing"
        )
    return problems


def _source_version() -> str:
    """Version stamped in include/libembedding/config.h, or 'unknown'."""
    header = _REPO_ROOT.parent / "include" / "libembedding" / "config.h"
    try:
        for line in header.read_text(encoding="utf-8").splitlines():
            if "LIBEMBEDDING_VERSION_STRING" in line and '"' in line:
                return line.split('"')[1]
    except OSError:
        pass
    return "unknown"


def _git_revision() -> str:
    with contextlib.suppress(Exception):
        proc = subprocess.run(  # fixed argv, no shell
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT.parent),
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    return "unknown"


def print_recommendations(results: list[QuantizationResult]) -> None:
    """Print the best configuration per model, per criterion."""
    print("\nRecommendations:")
    for model_name in sorted({r.model_name for r in results if r.success}):
        model_results = [r for r in results if r.model_name == model_name and r.success]
        if not model_results:
            continue
        baselines = baseline_by_batch(model_results)

        best_throughput = max(model_results, key=lambda r: r.texts_per_second)
        just = ""
        baseline = baselines.get((best_throughput.model_name, best_throughput.batch_size))
        if baseline is not None and best_throughput.quantization != "none":
            just = f" ({format_justification(baseline.texts_per_second, best_throughput.texts_per_second, baseline.process_peak_mb, best_throughput.process_peak_mb)})"
        batch = best_throughput.batch_size if best_throughput.batched else "n/a"
        print(
            f"  {model_name}: best throughput = {best_throughput.quantization} "
            f"@ batch {batch} ({best_throughput.model_code}, "
            f"{best_throughput.texts_per_second:.1f} docs/s){just}"
        )

        best_latency = min(model_results, key=lambda r: r.latency_ms_per_doc)
        batch = best_latency.batch_size if best_latency.batched else "n/a"
        print(
            f"  {model_name}: best latency = {best_latency.quantization} "
            f"@ batch {batch} ({best_latency.model_code}, "
            f"{best_latency.latency_ms_per_doc:.2f} ms/doc)"
        )

        measured = [r for r in model_results if r.process_peak_mb > 0]
        if measured:
            best_mem = min(measured, key=lambda r: r.process_peak_mb)
            batch = best_mem.batch_size if best_mem.batched else "n/a"
            print(
                f"  {model_name}: lowest peak RAM = {best_mem.quantization} "
                f"@ batch {batch} ({best_mem.model_code}, "
                f"{best_mem.process_peak_mb:.0f} MB)"
            )
        else:
            print(f"  {model_name}: lowest peak RAM = not measured")


def run_worker() -> int:
    """Measure a single configuration and emit its result on stdout."""
    payload = json.loads(sys.stdin.read())
    variant = ModelVariant(
        base_name=payload["base_name"],
        model_code=payload["model_code"],
        quantization=payload["quantization"],
        model_file=payload["model_file"],
    )
    batch_size = payload["batch_size"]
    result = run_single_configuration(
        variant,
        payload["texts"],
        None if batch_size is None else int(batch_size),
    )
    # Stamp the runtime identity from inside the process that ran the model, so
    # it cannot drift from the configuration it describes.
    result.onnxruntime = _onnxruntime_version()
    result.execution_provider = _execution_provider()
    result.weight_types = weight_types(cached_model_path(variant.model_code, variant.model_file))
    print(_RESULT_PREFIX + json.dumps(asdict(result)), flush=True)
    return 0


def plan_runs(
    base_models: list[str],
    quantizations: list[str],
    batch_sizes: list[int],
) -> tuple[list[tuple[ModelVariant, int | None]], list[SkippedVariant]]:
    """Build the run plan, and explain every mode that cannot be measured."""
    runs: list[tuple[ModelVariant, int | None]] = []
    skipped: list[SkippedVariant] = []

    for base in base_models:
        available = registry_variants(base)
        if not available:
            skipped.append(
                SkippedVariant(base, "-", "no registry entry for this model name")
            )
            continue
        by_quant = {v.quantization: v for v in available}
        for quant in quantizations:
            variant = by_quant.get(quant)
            if variant is None:
                have = ", ".join(sorted(by_quant))
                skipped.append(
                    SkippedVariant(
                        base,
                        quant,
                        f"no {quant} variant in the registry (available: {have})",
                    )
                )
                continue
            # Every mode is measured at every batch size. Dynamic quantization
            # used to be non-batched (it forced batch_size = num_texts and
            # rejected a smaller explicit batch); that guard was removed, and
            # dynamic now returns bit-identical embeddings batched or not. The
            # comparison is only fair if the variant and its baseline share a
            # batch size, so dynamic is benchmarked like every other mode.
            for bs in batch_sizes:
                runs.append((variant, bs))

    return runs, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the registry variants (FP32 and quantized) of a model"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
        help="Base model names to benchmark; every registry variant is compared",
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        choices=["none", "static", "dynamic", "fp16"],
        default=["none", "static", "dynamic", "fp16"],
        help="Modes to compare; modes absent from the registry are reported "
        "as not measured (default: none static dynamic fp16). fp16 covers the "
        "Qdrant/*-onnx-Q entries, whose weights are float16 behind a _Q suffix "
        "rather than INT8",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[8, 32, 64, 128],
        help="Batch sizes to test (default: 8 32 64 128). Every mode, "
        "dynamic included, is measured at every batch size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # deprecated: use --batch-sizes
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=1000,
        help=(
            "Total corpus size including the warmup slice "
            "(default: 1000). A 5% warmup slice is excluded from the timing."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quantization/results.json",
        help="Output JSON file for results (relative to benchmarks/)",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="quantization/results.html",
        help="Output HTML file for results (relative to benchmarks/)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: measure one configuration
    )
    args = parser.parse_args()

    if args.worker:
        return run_worker()

    if args.batch_size is not None:
        print(
            f"--batch-size is deprecated, use --batch-sizes {args.batch_size}",
            file=sys.stderr,
        )
        args.batch_sizes = [args.batch_size]

    if args.num_texts < 2:
        parser.error("--num-texts must be at least 2")

    runs, skipped = plan_runs(args.models, args.quantizations, args.batch_sizes)

    # Warmup slice is carved out of the corpus, so the total and the timed
    # count differ. Both are reported: the help text alone is misleading.
    warmup_size = max(1, args.num_texts // 20)
    timed_texts = generate_corpus(args.num_texts)[warmup_size:]

    env = collect_environment()
    env["git"] = _git_revision()

    print(f"{len(runs)} configuration(s) to measure, one subprocess each")
    print(f"Texts: {len(timed_texts)} timed (+{warmup_size} warmup)")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Conditions: {env}")
    if skipped:
        print("-" * 80)
        for s in skipped:
            print(f"  NOT MEASURED  {s.model_name} [{s.quantization}]: {s.reason}")
    print("-" * 80)

    report = Report()
    for variant, batch in runs:
        label = batch if batch is not None else "n/a"
        print(
            f"  {variant.base_name} [{variant.quantization}] "
            f"({variant.model_code}) batch={label} ...",
            end=" ",
            flush=True,
        )
        result = benchmark_configuration(variant, timed_texts, batch)
        report.results.append(result)
        if result.success:
            print(
                f"{result.texts_per_second:.1f} docs/s | "
                f"{result.latency_ms_per_doc:.2f} ms/doc | "
                f"{result.weights_mb:.1f} MB weights"
            )
        else:
            print(f"FAILED: {result.error}")

    report.skipped = skipped
    print("-" * 80)

    print("\nSummary:")
    print(
        f"{'Model':<38} {'Quant':<9} {'Code':<42} {'Batch':>6} "
        f"{'docs/s':>10} {'ms/doc':>9} {'Weights':>9} {'Peak':>6}"
    )
    print("-" * 136)
    for r in report.results:
        if r.success:
            batch = r.batch_size if r.batched else "n/a"
            print(
                f"{r.model_name:<38} {r.quantization:<9} {r.model_code:<42} {batch:>6} "
                f"{r.texts_per_second:>10.1f} {r.latency_ms_per_doc:>9.2f} "
                f"{r.weights_mb:>9.1f} {r.process_peak_mb:>6.0f}"
            )
    print("-" * 136)
    print("MB = model file on disk (download cost). Peak = resident high-water")
    print("mark of a process that ran only this configuration (includes the")
    print("ONNX Runtime arena, so it exceeds the file size).")

    # A run that mixed runtimes or providers is not publishable; say so loudly
    # rather than letting it become a footnote nobody reads.
    problems = verify_runtime_consistency(report.results)
    print()
    print(f"ONNX Runtime: {env.get('onnxruntime')}   Provider: {env.get('execution_provider')}")
    print(f"CPU: {env.get('cpu')}")
    if problems:
        for problem in problems:
            print(f"WARNING: {problem}")
    else:
        print("Runtime identical across all configurations: results are comparable.")

    print()
    print("Detected weight types (initializer tensors per dtype):")
    seen: set[str] = set()
    for r in report.results:
        if not r.success or r.model_code in seen:
            continue
        seen.add(r.model_code)
        if not r.weight_types:
            detail = "unavailable (pip install onnx)"
        else:
            detail = ", ".join(f"{k}: {v}" for k, v in sorted(r.weight_types.items()))
        print(f"  {r.model_name:<38} {r.quantization:<9} {r.model_code:<42} {detail}")

    output_path = _REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "environment": env,
                "runtime_consistency_problems": problems,
                "warmup_texts": warmup_size,
                "timed_texts": len(timed_texts),
                "skipped": [asdict(s) for s in report.skipped],
                "results": [asdict(r) for r in report.results],
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"\nResults saved to {output_path}")

    html_path = _REPO_ROOT / args.output_html
    write_html_report(report.results, report.skipped, env, html_path)
    print(f"HTML report saved to {html_path}")

    print_recommendations(report.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
