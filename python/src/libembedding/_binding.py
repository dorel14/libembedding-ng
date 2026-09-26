"""Low-level cffi binding layer. Loads the shared library and exposes ffi/lib.

Auteur: David Orel
Version: 1.8.0
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


# Runtime dependencies that must be discoverable before libembedding is loaded.
# On Windows the loader resolves a DLL's imports from the executable directory,
# the directories registered with SetDllDirectoryW, and PATH. The libcurl build
# that ships inside the package lives in the package directory, which is not the
# same directory as libembedding.dll in a source checkout, so both are registered
# (see _runtime_dirs for the bundled third_party/curl/bin location).
_DEPENDENT_DLLS = (
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "ggml.dll",
    "ggml-cpu.dll",
    "ggml-base.dll",
    "llama.dll",
    "libcurl-x64.dll",
    "libcurl.dll",
)


def _runtime_dirs(lib_path: Path) -> list[Path]:
    """Directories that may contain dependent shared libraries."""
    dirs = [lib_path.parent, Path(__file__).parent]
    # Common build trees, so a source checkout works without extra setup.
    root = Path(__file__).parent.parent.parent.parent
    dirs += [root / "build" / "bin" / "Debug", root / "build" / "bin" / "Release"]
    # Bundled libcurl runtime on Windows (installed by
    # scripts/fetch_windows_libcurl.ps1); absent from a wheel install, hence
    # the is_dir() filter below.
    dirs.append(root / "third_party" / "curl" / "bin")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for d in dirs:
        if d.is_dir() and d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


if platform.system() == "Windows":
    import ctypes
    import os

    _lib_path_str = _find_library()
    _lib_path = Path(_lib_path_str)
    if _lib_path.exists():
        for _dir in _runtime_dirs(_lib_path):
            # Preload known dependent DLLs so the loader finds them eagerly
            for _dep in _DEPENDENT_DLLS:
                _dep_path = _dir / _dep
                if _dep_path.exists():
                    try:
                        ctypes.CDLL(str(_dep_path))
                    except OSError:
                        pass
            ctypes.windll.kernel32.AddDllDirectory(str(_dir))
            ctypes.windll.kernel32.SetDllDirectoryW(str(_dir))
            os.environ["PATH"] = str(_dir) + os.pathsep + os.environ.get("PATH", "")
    lib = ffi.dlopen(_lib_path_str)
else:
    import os

    _lib_path_str = _find_library()
    _lib_path = Path(_lib_path_str)
    if _lib_path.exists():
        _dirs = [str(d) for d in _runtime_dirs(_lib_path)]
        _env_var = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        _existing = os.environ.get(_env_var, "")
        os.environ[_env_var] = os.pathsep.join([*_dirs, _existing]) if _existing else os.pathsep.join(_dirs)
    lib = ffi.dlopen(_lib_path_str)
