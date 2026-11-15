"""Shared run metadata for libembedding benchmarks.

A benchmark number without the conditions it was measured under cannot be
reproduced, compared or challenged. This module gathers those conditions once,
so every report carries the same fields:

* machine and software identity: git commit (and dirty flag), libembedding
  version, Python, OS, CPU model, physical/logical cores, RAM, ONNX Runtime and
  llama.cpp/ggml versions when the bindings expose them;
* per-run parameters: threads, batch, warmup, iterations, format,
  quantization, dataset;
* measurement helpers: p50/p95 of a sample list, model load time, peak RSS.

Peak RSS is a per-process high-water mark. It can only rise, so it is only
attributable to a backend when each backend runs in its own subprocess; see
docs/benchmark_methodology.md.

Usage::

    import _benchmeta as bm

    machine = bm.collect_machine_metadata()
    run = bm.RunParams(threads=0, batch=32, warmup=50, iterations=5,
                       format="onnx", quantization="fp32", dataset="synthetic-1000")
    model, load_ms = bm.timed_load(lambda: TextEmbedding(name))
    ...
    stats = bm.summarize_latencies(samples_ms)   # {"p50_ms": ..., "p95_ms": ...}
    report = bm.build_report(machine, run, results={...})
    bm.write_json(report, "out.json")
    print(bm.to_markdown(report))

Every function degrades to ``"unknown"`` / ``0`` instead of raising when a
source is unavailable (no DLL, no git, no psutil): a benchmark must not die
because its metadata could not be read, but a missing field is visible in the
report and flagged by :func:`missing_fields`.

Auteur: David Orel
Version: 1.0.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

SCHEMA_VERSION = 1
UNKNOWN = "unknown"

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Fields every report must carry (roadmap-2 section 11). `missing_fields()`
# checks them; a value of "unknown" or None counts as missing.
REQUIRED_MACHINE_FIELDS = (
    "git_commit",
    "libembedding_version",
    "os",
    "cpu_model",
    "physical_cores",
    "logical_cores",
    "ram_mb",
    "python",
)
REQUIRED_RUN_FIELDS = (
    "threads",
    "batch",
    "warmup",
    "iterations",
    "format",
    "quantization",
    "dataset",
)

T = TypeVar("T")


@dataclass
class RunParams:
    """Per-run parameters. ``threads=0`` means 'library default / auto'."""

    threads: int
    batch: int | Sequence[int]  # a sweep passes the list of batch sizes
    warmup: int
    iterations: int
    format: str  # "onnx" | "gguf" | ...
    quantization: str  # "fp32" | "q8_0" | "int8" | ...
    dataset: str  # name or description of the corpus

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def percentile(samples: Sequence[float], q: float) -> float:
    """Percentile ``q`` (0-100) with linear interpolation between ranks.

    Raises ValueError on an empty list or a ``q`` outside [0, 100]: a silent 0.0
    would look like a very fast run.
    """
    if not samples:
        raise ValueError("percentile of an empty sample list")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile must be within [0, 100], got {q}")
    s = sorted(float(x) for x in samples)
    if len(s) == 1:
        return s[0]
    rank = (len(s) - 1) * q / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def p50(samples: Sequence[float]) -> float:
    """Median (50th percentile)."""
    return percentile(samples, 50.0)


def p95(samples: Sequence[float]) -> float:
    """95th percentile."""
    return percentile(samples, 95.0)


def summarize_latencies(samples_ms: Sequence[float]) -> dict[str, float]:
    """n, min, p50, p95, max of a list of latencies in milliseconds."""
    return {
        "n": len(samples_ms),
        "min_ms": float(min(samples_ms)),
        "p50_ms": p50(samples_ms),
        "p95_ms": p95(samples_ms),
        "max_ms": float(max(samples_ms)),
    }


def relative_delta_pct(baseline: float, candidate: float) -> float:
    """``(candidate - baseline) / baseline`` in percent. Differences below 5 %
    are treated as noise by the project methodology."""
    if baseline == 0:
        raise ValueError("baseline is zero")
    return 100.0 * (candidate - baseline) / baseline


def is_noise(baseline: float, candidate: float, threshold_pct: float = 5.0) -> bool:
    """True when ``candidate`` is within ``threshold_pct`` of ``baseline``."""
    return abs(relative_delta_pct(baseline, candidate)) < threshold_pct


# --------------------------------------------------------------------------
# Timing and memory
# --------------------------------------------------------------------------


def timed_load(loader: Callable[[], T]) -> tuple[T, float]:
    """Call ``loader`` and return ``(result, load_time_ms)``.

    Measures the model load only: do the warmup separately, afterwards.
    """
    t0 = time.perf_counter()
    result = loader()
    return result, (time.perf_counter() - t0) * 1000.0


def peak_rss_mb() -> float:
    """Peak resident set size of *this* process, in MB (0.0 if unavailable).

    High-water mark: Windows ``PeakWorkingSetSize``, Linux/macOS ``ru_maxrss``
    (KiB on Linux, bytes on macOS). Use one subprocess per backend.
    """
    if os.name == "nt":
        with contextlib.suppress(Exception):
            import ctypes

            class _Counters(ctypes.Structure):
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

            windll = getattr(ctypes, "windll")  # noqa: B009 - absent off Windows
            handle = windll.kernel32.OpenProcess(0x0400 | 0x0010, False, os.getpid())
            if handle:
                try:
                    counters = _Counters()
                    counters.cb = ctypes.sizeof(_Counters)
                    if windll.psapi.GetProcessMemoryInfo(
                        handle, ctypes.byref(counters), counters.cb
                    ):
                        return counters.PeakWorkingSetSize / (1024 * 1024)
                finally:
                    windll.kernel32.CloseHandle(handle)
        return 0.0

    with contextlib.suppress(Exception):
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    return 0.0


# --------------------------------------------------------------------------
# Machine metadata
# --------------------------------------------------------------------------


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """``{"git_commit": <short sha>, "git_dirty": bool|None}``.

    ``git_dirty`` is None when git is unavailable, never a guessed False.
    """
    root = str(repo_root or _REPO_ROOT)

    def run(args: list[str]) -> str | None:
        with contextlib.suppress(Exception):
            proc = subprocess.run(  # fixed argv, no shell
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=root,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        return None

    commit = run(["rev-parse", "--short", "HEAD"])
    status = run(["status", "--porcelain"]) if commit else None
    return {
        "git_commit": commit or UNKNOWN,
        "git_dirty": None if status is None else bool(status),
    }


def _cstr(func_name: str) -> str:
    """Call a ``const char* f(void)`` of the C API, or return UNKNOWN."""
    with contextlib.suppress(Exception):
        from libembedding import _binding

        func = getattr(_binding.lib, func_name)
        return _binding.ffi.string(func()).decode("utf-8", errors="replace")
    return UNKNOWN


def _hardware() -> dict[str, Any]:
    """Hardware fingerprint from the native library, with a stdlib fallback."""
    with contextlib.suppress(Exception):
        from libembedding import detect_hardware

        hw = detect_hardware()
        return {
            "cpu_model": hw.cpu_name or UNKNOWN,
            "physical_cores": int(hw.physical_cores),
            "logical_cores": int(hw.logical_cores),
            "ram_mb": int(hw.ram_mb),
            "cpu_features": hw.features or "",
            # platform.platform() carries the release/build, hw.os_name only
            # the family ("Windows").
            "os": platform.platform(),
        }
    return {
        "cpu_model": platform.processor() or UNKNOWN,
        "physical_cores": 0,
        "logical_cores": os.cpu_count() or 0,
        "ram_mb": 0,
        "cpu_features": "",
        "os": platform.platform(),
    }


def collect_machine_metadata(repo_root: Path | None = None) -> dict[str, Any]:
    """Identity of the machine and software a benchmark ran on."""
    version = UNKNOWN
    with contextlib.suppress(Exception):
        import libembedding

        version = str(getattr(libembedding, "__version__", UNKNOWN))

    meta: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **git_info(repo_root),
        "libembedding_version": version,
        "python": platform.python_version(),
        "machine": platform.machine(),
        **_hardware(),
        "onnxruntime_version": _cstr("lembed_onnxruntime_version"),
        # lembed_llama_version() reports the build flavour ("llama.cpp enabled"),
        # not a version number: no ggml/llama.cpp version is exposed by the
        # bindings. The vendored snapshot is in third_party/README.md.
        "llama_cpp_build": _cstr("lembed_llama_version"),
        "ggml_cpu_features": _cstr("lembed_ggml_cpu_features"),
    }
    return meta


# --------------------------------------------------------------------------
# Report assembly and output
# --------------------------------------------------------------------------


def build_report(
    machine: dict[str, Any],
    run: RunParams,
    results: dict[str, Any] | None = None,
    title: str = "",
) -> dict[str, Any]:
    """Combine machine metadata, run parameters and results in one document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "title": title,
        "machine": dict(machine),
        "run": run.as_dict(),
        "results": results if results is not None else {},
    }


def missing_fields(report: dict[str, Any]) -> list[str]:
    """Mandatory metadata fields that are absent, None or 'unknown'.

    Names are prefixed ``machine.`` or ``run.``. An empty list means the report
    is complete. A 0 core count or RAM size also counts as missing.
    """
    missing: list[str] = []
    machine = report.get("machine", {})
    run = report.get("run", {})
    for name in REQUIRED_MACHINE_FIELDS:
        value = machine.get(name)
        if value in (None, "", UNKNOWN) or (
            name in ("physical_cores", "logical_cores", "ram_mb") and value == 0
        ):
            missing.append(f"machine.{name}")
    for name in REQUIRED_RUN_FIELDS:
        if run.get(name) is None or run.get(name) == "":
            missing.append(f"run.{name}")
    return missing


def to_json(report: dict[str, Any]) -> str:
    """Serialize with stable key order and a trailing newline."""
    return json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"


def write_json(report: dict[str, Any], path: Any) -> None:
    """Write the JSON report with LF line endings (repo rule, any platform)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(to_json(report))


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ") or "n/a"


def to_markdown(report: dict[str, Any], heading: str = "## Run metadata") -> str:
    """Markdown block with the machine and run parameter tables."""
    machine = report.get("machine", {})
    run = report.get("run", {})
    lines = [heading, "", "| Field | Value |", "|---|---|"]
    commit = machine.get("git_commit", UNKNOWN)
    dirty = machine.get("git_dirty")
    lines.append(
        f"| git commit | {_cell(commit)}{' (dirty)' if dirty else ''} |"
    )
    for key, value in machine.items():
        if key in ("git_commit", "git_dirty"):
            continue
        lines.append(f"| {key} | {_cell(value)} |")
    for f in fields(RunParams):
        lines.append(f"| {f.name} | {_cell(run.get(f.name))} |")
    absent = missing_fields(report)
    if absent:
        lines += ["", "Missing mandatory fields: " + ", ".join(absent) + "."]
    lines.append("")
    return "\n".join(lines)
