#!/usr/bin/env python3
"""Synchronise the human-readable version stamps with the canonical version.

The canonical version lives in ``python/pyproject.toml`` and is bumped by
python-semantic-release (see ``[tool.semantic_release] version_toml`` in the
root ``pyproject.toml``). The same bump also updates ``CMakeLists.txt`` and
``include/libembedding/config.h`` through ``version_variables``.

What remains are the *documentation* stamps:

* ``Version: X.Y.Z`` in every C/C++ header of ``include/libembedding`` (~59 files)
* ``Version: X.Y.Z`` in every Python module of ``python/src/libembedding``
* the ``(vX.Y.Z)`` marker in ``python/src/libembedding/_cdefs.h``
* the "Version courante" line in ``AGENTS.md``

python-semantic-release's ``version_variables`` cannot express these: each entry
names a single literal file path (no globs), and the variable name is escaped
before being compiled into the search pattern, so no custom regex is possible.

This script is idempotent, only ever rewrites a stamp that is already a
well-formed ``MAJOR.MINOR.PATCH``, and verifies its own work. It is safe to run
repeatedly and fails loudly rather than leaving a partially synced tree.

Usage::

    python scripts/sync_version.py            # rewrite stamps to the canonical version
    python scripts/sync_version.py --check    # exit 1 if any stamp is stale
    python scripts/sync_version.py --version 9.9.9   # override the target version
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Stamp rules -------------------------------------------------------------
# A rule is (regex, template, targets). The template must reproduce the match
# with {version} substituted; named groups are preserved verbatim.
#
# Scoping matters: AGENTS.md *documents* the header format using
# "Version: 1.4.0" as an example and references llama.cpp "v0.3.0". A global
# "rewrite every version-looking token" pass would corrupt both, so each rule is
# bound to the files where that stamp actually denotes the project version.

HEADER_GLOBS = (
    "include/libembedding/*.h",
    "include/libembedding/*.hpp",
    "include/libembedding/detail/*.hpp",
    "include/libembedding/cpp/*.hpp",
)
PYTHON_GLOBS = ("python/src/libembedding/*.py",)
CONFIG_GLOBS = ("include/libembedding/config.h",)

CANONICAL_TOML = ROOT / "python" / "pyproject.toml"
CDEFS_FILE = ROOT / "python" / "src" / "libembedding" / "_cdefs.h"
AGENTS_FILE = ROOT / "AGENTS.md"

# A strictly-formed semver core. Deliberately narrow: a stamp that does not look
# like a version is left untouched rather than guessed at.
SEMVER_CORE = r"\d+\.\d+\.\d+"


class Rule:
    """One version stamp: how to find it, how to rewrite it, where it lives."""

    def __init__(self, pattern: str, template: str, globs: tuple[str, ...] = ()) -> None:
        self.regex = re.compile(pattern, re.MULTILINE)
        self.template = template
        self.globs = globs

    def files(self) -> list[Path]:
        found: dict[Path, None] = {}
        for pattern in self.globs:
            for path in sorted(ROOT.glob(pattern)):
                if path.is_file():
                    found.setdefault(path.resolve(), None)
        return list(found)

    def apply(self, text: str, version: str) -> tuple[str, int]:
        return self.regex.subn(self.expand(version), text)

    def expand(self, version: str) -> str:
        """Fill the template. ``{version}`` plus the split numeric components.

        A rule may use ``{major}``, ``{minor}`` or ``{patch}`` instead of
        ``{version}`` for the stamps that are not a single ``X.Y.Z`` token,
        such as the ``#define LIBEMBEDDING_VERSION_MAJOR 1`` form.
        """
        major, minor, patch = version.split(".")
        return self.template.format(
            version=version, major=major, minor=minor, patch=patch
        )

    def matches(self, text: str) -> list[str]:
        return [m.group(0) for m in self.regex.finditer(text)]


def _version_macro_rule(part: str) -> Rule:
    """Rule for ``#define LIBEMBEDDING_VERSION_<PART> <n>`` in config.h.

    python-semantic-release's ``version_variables`` only stamps
    ``LIBEMBEDDING_VERSION_STRING``, so the three numeric macros are handled
    here. They are not cosmetic: ``detail/autotune_cache.hpp`` builds the
    autotune cache key from ``MAJOR`` and ``MINOR``, so a stale value makes
    two different releases share their tuning cache.
    """
    return Rule(
        r"^(?P<prefix>#define\s+LIBEMBEDDING_VERSION_" + part + r"\s+)\d+(?P<suffix>\s*)$",
        r"\g<prefix>{" + part.lower() + r"}\g<suffix>",
        CONFIG_GLOBS,
    )


RULES = (
    Rule(
        r"^(?P<prefix>\s*\*\s*Version:\s*)" + SEMVER_CORE + r"(?P<suffix>\s*)$",
        r"\g<prefix>{version}\g<suffix>",
        HEADER_GLOBS,
    ),
    Rule(
        r"^(?P<prefix>Version:\s*)" + SEMVER_CORE + r"(?P<suffix>\s*)$",
        r"\g<prefix>{version}\g<suffix>",
        PYTHON_GLOBS,
    ),
    Rule(r"\(v" + SEMVER_CORE + r"\)", r"(v{version})", ("python/src/libembedding/_cdefs.h",)),
    _version_macro_rule("MAJOR"),
    _version_macro_rule("MINOR"),
    _version_macro_rule("PATCH"),
    Rule(
        r"^(?P<prefix>> \*\*Version courante\*\* : )" + SEMVER_CORE + r"(?P<suffix>\s*)$",
        r"\g<prefix>{version}\g<suffix>",
        ("AGENTS.md",),
    ),
)


def read_text(path: Path) -> str:
    """Read *path* without translating its line endings.

    ``newline=""`` disables universal-newline translation, so CRLF and LF
    survive the round trip untouched. This matters because the repository
    stores LF (``.gitattributes``: ``* text=auto eol=lf``) and the script runs
    on Windows, where the default text mode would rewrite every LF as CRLF and
    turn a one-line version bump into a whole-file diff.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text(path: Path, text: str) -> None:
    """Write *text* to *path* without translating its line endings."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def read_canonical_version() -> str:
    """Return the version declared in ``python/pyproject.toml``."""
    text = read_text(CANONICAL_TOML)
    match = re.search(
        r"^version\s*=\s*[\"'](" + SEMVER_CORE + r")[\"']", text, re.MULTILINE
    )
    if not match:
        msg = f"No version = \"X.Y.Z\" found in {CANONICAL_TOML}"
        raise SystemExit(f"error: {msg}")
    return match.group(1)


def managed_rules() -> list[tuple[Rule, list[Path]]]:
    """Rule/file pairs, dropping any rule that matches nothing on disk."""
    pairs: list[tuple[Rule, list[Path]]] = []
    for rule in RULES:
        paths = rule.files()
        if paths:
            pairs.append((rule, paths))
    return pairs


def sync(version: str, check_only: bool) -> int:
    pairs = managed_rules()
    if not pairs:
        print("error: no version rules matched any file", file=sys.stderr)
        return 1

    stale: list[tuple[Path, str]] = []
    changed = 0
    total_stamps = 0

    for rule, paths in pairs:
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            original = read_text(path)
            found = rule.matches(original)
            total_stamps += len(found)

            if check_only:
                # A stamp is in sync when re-applying the rule with the target
                # version is a no-op. Comparing the rewritten text (rather than
                # looking for the version substring) also works for the stamps
                # that are not a single "X.Y.Z" token, e.g. the numeric macros.
                updated, _ = rule.apply(original, version)
                if updated != original:
                    stale.append((path, f"stamp(s) {found[:3]} do not match {version}"))
                continue

            updated, n = rule.apply(original, version)
            if n and updated != original:
                write_text(path, updated)
                changed += 1
            elif found and not found:
                stale.append((path, "stamp not recognised by any rule"))

    if stale:
        print("error: version stamps out of sync:", file=sys.stderr)
        for path, reason in stale:
            print(f"  - {path.relative_to(ROOT).as_posix()}: {reason}", file=sys.stderr)
        if not check_only:
            print(
                "error: refusing to report success; the stamp format may have changed",
                file=sys.stderr,
            )
        return 1

    if check_only:
        print(f"OK: {total_stamps} stamps across all managed files are at {version}")
        return 0

    print(f"Synced {changed} file(s) to {version} ({total_stamps} stamps checked).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only; exit 1 when a stamp is stale",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="target version (defaults to python/pyproject.toml)",
    )
    args = parser.parse_args()

    version = args.version or read_canonical_version()
    if not re.fullmatch(SEMVER_CORE, version):
        print(f"error: invalid version {version!r}", file=sys.stderr)
        return 1

    return sync(version, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
