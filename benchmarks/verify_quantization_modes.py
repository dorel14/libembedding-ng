#!/usr/bin/env python3
"""Check each registry entry's declared quantization mode against its ONNX file.

Why this exists
---------------
A registry tag is metadata someone typed by hand. For the four
``Qdrant/*-onnx-Q`` entries it said ``static`` (INT8) and the description said
"Quantized", while every initializer in ``model_optimized.onnx`` was
``FLOAT16``. Selecting ``quantization="static"`` then returned float16 weights:
smaller on disk, slower than plain FP32 on any CPU without native FP16
arithmetic, and not what the caller asked for.

The benchmark noticed the symptom (a 9.5x slowdown) but could not name the
cause. This script names it, and can run in CI so the next mismatch is caught
where it is introduced rather than in a published benchmark.

What it checks
--------------
For every text / sparse / image registry entry, it reads the initializer dtypes
out of the cached ``.onnx`` file and compares them with the declared mode:

===================  ==========================================================
declared mode        initializer dtypes expected
===================  ==========================================================
``none``             ``FLOAT`` (float32)
``static``           ``INT8`` / ``UINT8``, or ``FLOAT16`` only when the graph
                     really is a QDQ graph -- otherwise ``FLOAT16`` is reported
``dynamic``          ``INT8`` / ``UINT8``
``fp16``             ``FLOAT16``
===================  ==========================================================

A ``FLOAT16`` file declared ``static`` or ``dynamic`` is the failure this script
exists for, and it exits non-zero.

Usage
-----
    python benchmarks/verify_quantization_modes.py            # report
    python benchmarks/verify_quantization_modes.py --strict   # non-zero on any mismatch

Only cached models are inspected (no network). Entries whose file is absent are
reported as SKIPPED, so a fresh checkout reports coverage rather than failing.

Auteur: David Orel
Version: 1.9.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "python" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import onnx
    from onnx import TensorProto
except ImportError:  # pragma: no cover - reported, not raised
    print(
        "ERROR: the 'onnx' package is required.\n"
        "       pip install onnx",
        file=sys.stderr,
    )
    raise SystemExit(2)

_DTYPE_NAMES = {v: k for k, v in TensorProto.DataType.items()}

# Dtypes that can carry a weight, and the dtype each declared mode promises.
# INT64 / BOOL / string initializers are shape and position buffers, not
# weights: bge-m3, e5-small and nomic all carry them in FP32 models, so they
# must not influence the verdict.
_INT8 = {"INT8", "UINT8", "INT32"}
_FP32 = {"FLOAT"}
_FP16 = {"FLOAT16"}
_WEIGHT_DTYPES = _INT8 | _FP32 | _FP16


def cache_dir() -> Path:
    """Where models are cached, same order the library uses.

    The first candidate that actually holds models wins: on this machine
    ``%LOCALAPPDATA%\\libembedding`` exists but is empty, while the models live
    under ``~/.cache/libembedding``, and picking the empty one would report
    "nothing to verify" instead of checking anything.
    """
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "libembedding")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(Path(xdg) / "libembedding")
    candidates.append(Path(os.path.expanduser("~")) / ".cache" / "libembedding")

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("models--*")):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def model_file(cache_root: Path, model_code: str, model_file: str) -> Path:
    """Resolve a registry entry to its cached file.

    Mirrors the on-disk layout: ``<cache>/models--<owner>-<repo>/<model_file>``,
    plus the ``onnx/`` subdirectory some entries live in.
    """
    root = cache_root / f"models--{model_code.replace('/', '-')}"
    for candidate in (root / model_file, root / "onnx" / model_file):
        if candidate.exists():
            return candidate
    # Fall back to any ONNX file in the repo directory.
    matches = sorted(root.rglob("*.onnx")) if root.exists() else []
    return matches[0] if matches else root / model_file


def initializer_dtypes(path: Path) -> dict[str, int]:
    """Count initializer dtypes in an ONNX file, by friendly name."""
    model = onnx.load(str(path), load_external_data=False)
    counter: collections.Counter[str] = collections.Counter()
    for init in model.graph.initializer:
        counter[_DTYPE_NAMES.get(init.data_type, f"?{init.data_type}")] += 1
    return dict(counter)


def classify(dtypes: dict[str, int]) -> str:
    """Reduce a dtype histogram to the weight format actually shipped.

    Returns one of ``fp32``, ``fp16``, ``int8`` or ``unknown``, and compares
    against the same vocabulary as :func:`expected`. Shapes and indices are
    ignored; partial or layer-wise quantization leaves some tensors FLOAT, but
    as soon as any INT8/UINT8 weight is present the file is an INT8 graph.
    """
    present = set(dtypes) & _WEIGHT_DTYPES
    if not present:
        return "unknown"
    if present & _INT8:
        return "int8"
    if present & _FP16:
        return "fp16"
    return "fp32"


def expected(declared: str) -> set[str]:
    """Weight formats acceptable for a declared mode, same vocabulary as
    :func:`classify`."""
    if declared == "none":
        return {"fp32"}
    if declared in ("static", "dynamic"):
        return {"int8"}
    if declared == "fp16":
        return {"fp16"}
    # Unknown / "auto": no registry entry should carry it.
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any mismatch (default: report only)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="only check these model_codes",
    )
    args = parser.parse_args()

    from libembedding.models import list_text_models

    root = cache_dir()
    print(f"cache: {root}")
    if not root.exists():
        print("no model cache: nothing to verify", file=sys.stderr)
        return 0

    checked = skipped = mismatched = 0

    for info in list_text_models():
        if args.models and info.model_code not in args.models:
            continue
        path = model_file(root, info.model_code, info.model_file)
        if not path.exists():
            skipped += 1
            continue

        try:
            dtypes = initializer_dtypes(path)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            print(f"  ERROR   {info.model_code}: {exc}")
            skipped += 1
            continue

        found = classify(dtypes)
        declared = info.quantization
        checked += 1

        ok = found in expected(declared)
        detail = ", ".join(f"{k}={v}" for k, v in sorted(dtypes.items()))
        if ok:
            print(f"  OK      {info.model_code:<44} declared={declared:<8} file={found}")
            continue

        mismatched += 1
        print(
            f"  MISMATCH {info.model_code:<44} declared={declared:<8} "
            f"file={found}  [{detail}]",
            file=sys.stderr,
        )

    print()
    print(f"checked={checked} skipped={skipped} mismatched={mismatched}")
    if mismatched:
        print(
            "\nA declared mode that does not match the file means "
            "quantization= selects weights the caller did not ask for.\n"
            "Fix the registry entry, or re-tag it to the format it really ships.",
            file=sys.stderr,
        )
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
