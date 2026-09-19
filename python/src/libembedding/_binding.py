"""Low-level cffi binding layer. Loads the shared library and exposes ffi/lib.

Auteur: David Orel
Version: 1.6.0
"""

import platform
from pathlib import Path

import cffi

ffi = cffi.FFI()

_cdefs_path = Path(__file__).with_name("_cdefs.h")
ffi.cdef(_cdefs_path.read_text(encoding="utf-8"))


def _find_library() -> str:
    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(
        platform.system(), ".so"
    )
    pkg_dir = Path(__file__).parent
    root = pkg_dir.parent.parent.parent  # project root (libembedding/)
    candidates = [
        pkg_dir / f"libembedding{ext}",
        pkg_dir / "lib" / f"libembedding{ext}",
        root / "build" / "Debug" / f"libembedding{ext}",
        root / "build" / "Release" / f"libembedding{ext}",
        root / "build" / "python" / f"libembedding{ext}",
        root / "build" / f"libembedding{ext}",
        root / "build3" / "Release" / f"libembedding{ext}",
        root / "build3" / f"libembedding{ext}",
        root / "build_test" / "Debug" / f"libembedding{ext}",
        root / "build_test" / "Release" / f"libembedding{ext}",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return f"libembedding{ext}"


# On Windows, preload dependent DLLs and ensure the library's directory
# is in the DLL search path so that dependent DLLs (onnxruntime, llama.cpp, etc.) can be found.
if platform.system() == "Windows":
    import ctypes
    import os

    _lib_path_str = _find_library()
    _lib_path = Path(_lib_path_str)
    if _lib_path.exists():
        _lib_dir = str(_lib_path.parent)
        # Preload known dependent DLLs to ensure they are in memory
        for _dep in (
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "ggml.dll",
            "ggml-cpu.dll",
            "ggml-base.dll",
            "llama.dll",
        ):
            _dep_path = Path(_lib_dir) / _dep
            if _dep_path.exists():
                try:
                    ctypes.CDLL(str(_dep_path))
                except OSError:
                    pass
        # Add the library directory to the DLL search path
        ctypes.windll.kernel32.SetDllDirectoryW(_lib_dir)
        # Also add to PATH as fallback
        os.environ["PATH"] = _lib_dir + os.pathsep + os.environ.get("PATH", "")
    lib = ffi.dlopen(_lib_path_str)
else:
    lib = ffi.dlopen(_find_library())
