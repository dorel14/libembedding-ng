"""Shared pytest configuration for libembedding Python tests."""

import os
import platform
import sys
from pathlib import Path

# Ensure the src layout is importable when running tests from the repo root.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "python" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Locate the built shared library and ensure its directory is on PATH
# so Windows can resolve the runtime DLLs (onnxruntime, libcurl, MSVC runtime).
from libembedding._binding import _find_library

_lib_path = _find_library()
if os.path.exists(_lib_path):
    _lib_dir = str(Path(_lib_path).parent)
    if _lib_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _lib_dir + os.pathsep + os.environ.get("PATH", "")
else:
    # Fallback candidates (in case _binding.py doesn't know about them).
    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(
        platform.system(), ".so"
    )
    for candidate in [
        ROOT / "build" / "python" / f"libembedding{ext}",
        ROOT / "build" / f"libembedding{ext}",
        ROOT / "build" / "Release" / f"libembedding{ext}",
        ROOT / "build3" / "Release" / f"libembedding{ext}",
        ROOT / "build3" / f"libembedding{ext}",
    ]:
        if candidate.exists():
            _lib_dir = str(candidate.parent)
            if _lib_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _lib_dir + os.pathsep + os.environ.get("PATH", "")
            break
