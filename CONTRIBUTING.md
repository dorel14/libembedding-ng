# Contributing to libembedding-ng

Thanks for helping. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first. Report
vulnerabilities through [SECURITY.md](SECURITY.md), not public issues.

## Setup

Requirements: CMake >= 3.18, a C++17 compiler, Python >= 3.9.

Windows (PowerShell):

```powershell
pwsh -File scripts/fetch_windows_libcurl.ps1   # libcurl DLL; configure fails without it
cmake -S . -B build
cmake --build build --config Debug
```

Linux / macOS:

```bash
./build.sh
./run_tests.sh
```

Build only inside `build/` and delete build artefacts when you are done. Do not edit
files with scripts or mass-rewrite tools; edit them directly. Files use LF line
endings and no UTF-8 BOM.

## Validation (V0-V5)

Run from the repository root before opening a PR.

| Step | Command | Pass criterion |
|---|---|---|
| V0 | re-read every file you touch | - |
| V2 | `cmake -S . -B build` then `cmake --build build --config Debug` | 0 error, 0 new project warning (ignore `third_party/`) |
| V3 | `ctest --test-dir build -C Debug --timeout 120` | all green, without editing `PATH` |
| V4 | `cd python; pytest tests/ -q` | 0 failure; coverage >= 40 % in CI |
| V4b | `pytest tests/ -q -m "not network"` | passes offline |
| V5 | `ruff check python/src python/tests`; `pyright python/src`; `python scripts/sync_version.py --check` | all green |
| Docs | `python scripts/generate_llm_docs.py` | only if `README.md`, `docs/`, `examples/` or the script changed |

On Windows, `.\run_tests.ps1` builds Debug and runs ctest. A ctest "Timeout" is often
a missing DLL (`0xC0000135`): run the executable directly and read `$LASTEXITCODE`.

If a step is red, fix it before continuing. `git diff --name-only` must list only the
files you meant to change.

## Commits and pull requests

- Conventional commits: `feat:`, `fix(scope):`, `test:`, `docs:`, `perf:`,
  `refactor:`, `ci:`. One tested feature = one commit.
- Open PRs against `dev`. Maintainers promote `dev` to `main`.
- Do not bump the version and do not edit `CHANGELOG.md`: semantic-release derives
  both from the commit messages on `main`. `python scripts/sync_version.py --check`
  must stay green.
- Do not hand-edit `llm.txt` / `llm_full.txt`; regenerate them with the script above.
- New files carry `SPDX-License-Identifier: MIT` and a header with author and version.
- Docs are bilingual: update `docs/` (FR) and `docs/en/` (EN) for any user-visible
  feature.

## Public C API rule

Any change to the public C API must be mirrored in
`python/src/libembedding/_cdefs.h` and in `libembedding.def`, in the same commit.
Otherwise the Python binding raises `AttributeError` or the Windows DLL does not
export the symbol. Update `test_struct_sizes` when a struct changes.

## Scope guardrails

- No vector search (no HNSW, IVF, PQ, sharding, global top-K).
- No `backend=` parameter in new API: the backend is chosen by file extension
  (see `docs/adr/ADR-001-routing-by-extension.md`).
- Registry models must be MIT-compatible and validated (fidelity vs ONNX/PyTorch,
  Recall@K).
- Do not patch `third_party/`.

## Status definitions

- **Implemented**: the code exists and the unit tests pass.
- **Validated**: implemented, and backed by proof: a test of correctness (fidelity,
  Recall@K or the relevant invariant) and a before/after benchmark when performance is
  claimed. Benchmarks use one subprocess per variant, a separate warmup, and the
  median; a difference under 5 % is noise.

Implemented does not mean Validated. Docs and release notes must not call a feature
Validated without that proof.

## Definition of Done

A PR is mergeable when:

- [ ] V2-V5 are green on at least one platform (Debug; Release if build flags changed).
- [ ] New behaviour has a test; a fix has a regression test.
- [ ] Performance claims have a before/after benchmark attached (or are not made).
- [ ] Public C API changes are mirrored in `_cdefs.h` and `libembedding.def`.
- [ ] README / `docs/` (FR and EN) and public header comments are updated;
      `llm.txt` / `llm_full.txt` regenerated when required.
- [ ] No version bump, no `CHANGELOG.md` edit, no secret, no change outside the
      intended files.
- [ ] CI is green.
