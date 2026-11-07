"""Low-level cffi binding layer. Loads the shared library and exposes ffi/lib.

Auteur: David Orel
Version: 1.11.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import platform
import warnings
from pathlib import Path

import cffi

ffi = cffi.FFI()

_cdefs_path = Path(__file__).with_name("_cdefs.h")
ffi.cdef(_cdefs_path.read_text(encoding="utf-8"))


def _library_candidates(pkg_dir: Path, root: Path, ext: str) -> tuple[list[Path], int]:
    """Candidate library paths in lookup order, and how many are package copies.

    The package copies come first: they are what a wheel install ships. The
    build-tree paths only exist in a source checkout.
    """
    packaged = [pkg_dir / f"libembedding{ext}", pkg_dir / "lib" / f"libembedding{ext}"]
    build_tree = [
        root / "build" / "Debug" / f"libembedding{ext}",
        root / "build" / "Release" / f"libembedding{ext}",
        root / "build" / "python" / f"libembedding{ext}",
        root / "build" / f"libembedding{ext}",
        root / "build3" / "Release" / f"libembedding{ext}",
        root / "build3" / f"libembedding{ext}",
        root / "build_test" / "Debug" / f"libembedding{ext}",
        root / "build_test" / "Release" / f"libembedding{ext}",
    ]
    return packaged + build_tree, len(packaged)


def _pick_library(candidates: list[Path], packaged_count: int) -> Path | None:
    """First existing candidate; warns when it is a package copy older than a build.

    In a source checkout the package directory is the wheel staging area: a copy
    left there goes stale as soon as the tree is rebuilt, and every test or
    benchmark run without LIBEMBEDDING_LIBRARY then silently exercises old code.
    """
    existing = [c for c in candidates if c.is_file()]
    if not existing:
        return None
    chosen = existing[0]
    if chosen in candidates[:packaged_count]:
        newer = [
            c for c in existing
            if c in candidates[packaged_count:]
            and c.stat().st_mtime > chosen.stat().st_mtime
        ]
        if newer:
            warnings.warn(
                f"Loading {chosen}, which is older than the build output {newer[0]}. "
                "Set LIBEMBEDDING_LIBRARY to the library to test, or remove the "
                "stale copy.",
                stacklevel=3,
            )
    return chosen


def _find_library() -> str:
    # An explicit path wins: it is the only unambiguous choice in a source
    # checkout that holds several builds.
    override = os.environ.get("LIBEMBEDDING_LIBRARY")
    if override:
        if not Path(override).is_file():
            raise OSError(f"LIBEMBEDDING_LIBRARY points to a missing file: {override}")
        return str(Path(override).resolve())

    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(
        platform.system(), ".so"
    )
    pkg_dir = Path(__file__).parent
    root = pkg_dir.parent.parent.parent  # project root (libembedding/)
    chosen = _pick_library(*_library_candidates(pkg_dir, root, ext))
    return str(chosen.resolve()) if chosen else f"libembedding{ext}"


# Runtime dependencies that must be discoverable before libembedding is loaded.
# On Windows the loader resolves a DLL's imports from the executable directory,
# the directories registered with SetDllDirectoryW, and PATH. The libcurl build
# that ships inside the package lives in the package directory, which is not the
# same directory as libembedding.dll in a source checkout, so both are registered
# (see _runtime_dirs for the bundled third_party/curl/bin location).
# ggml's CPU backend is not listed: with CPU dispatch it is one of several
# ggml-cpu-<variant> modules that ggml scores and loads itself, from the
# library's directory; preloading one here would pick the variant for it.
_DEPENDENT_DLLS = (
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "ggml.dll",
    "ggml-base.dll",
    "llama.dll",
    "libcurl-x64.dll",
    "libcurl.dll",
)


def _runtime_dirs(lib_path: Path) -> list[Path]:
    """Directories that may contain dependent shared libraries, in priority order.

    The library's own directory comes first: a build tree places the matching
    ggml, llama and ONNX Runtime copies next to libembedding. Other build
    directories (build/bin/Debug, build/bin/Release) are deliberately absent:
    taking a dependency from there mixes Debug with Release, or a stale build
    with a fresh one, in the same process.
    """
    root = Path(__file__).parent.parent.parent.parent
    dirs = [
        lib_path.parent,
        # Wheel layout, and the MSVC runtime a wheel bundles.
        Path(__file__).parent,
        # Source-checkout libcurl installed by scripts/fetch_windows_libcurl.ps1;
        # absent from a wheel install, hence the is_dir() filter below.
        root / "third_party" / "curl" / "bin",
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for d in dirs:
        if d.is_dir() and d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def _select_dependencies(dirs: list[Path]) -> list[Path]:
    """One path per known dependency: the copy in the first directory holding it.

    Windows resolves an import by module name to an already loaded DLL, so
    whichever copy is preloaded first is the one libembedding binds to. Loading
    a single copy per name makes that choice explicit and keeps every other copy
    out of the process.
    """
    selected: list[Path] = []
    for name in _DEPENDENT_DLLS:
        for d in dirs:
            if (d / name).is_file():
                selected.append(d / name)
                break
    return selected


if platform.system() == "Windows":
    import ctypes

    _lib_path_str = _find_library()
    _lib_path = Path(_lib_path_str)
    if _lib_path.exists():
        _dirs = _runtime_dirs(_lib_path)
        for _dep_path in _select_dependencies(_dirs):
            try:
                ctypes.CDLL(str(_dep_path))
            except OSError:
                pass
        for _dir in reversed(_dirs):
            ctypes.windll.kernel32.AddDllDirectory(str(_dir))
            os.environ["PATH"] = str(_dir) + os.pathsep + os.environ.get("PATH", "")
        # SetDllDirectoryW keeps a single directory: the library's own.
        ctypes.windll.kernel32.SetDllDirectoryW(str(_dirs[0]))
    lib = ffi.dlopen(_lib_path_str)
else:
    _lib_path_str = _find_library()
    _lib_path = Path(_lib_path_str)
    if _lib_path.exists():
        _dirs = [str(d) for d in _runtime_dirs(_lib_path)]
        _env_var = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        _existing = os.environ.get(_env_var, "")
        os.environ[_env_var] = os.pathsep.join([*_dirs, _existing]) if _existing else os.pathsep.join(_dirs)
    lib = ffi.dlopen(_lib_path_str)
