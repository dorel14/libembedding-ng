---
layout: default
title: ADR-002 CPU dispatch
nav_exclude: true
---

# ADR-002: CPU dispatch through the upstream ggml mechanism

Status: Accepted (2026-11-08)

## Context

A wheel must run on CPUs from baseline x86-64 to AVX-512 (and NEON/SVE on Arm). A
build with `GGML_NATIVE` targets the build machine and cannot be shipped. Writing our
own runtime dispatch was considered (RFC P2.6).

The part of libembedding compiled by us was measured with and without AVX2:
0.98 to 1.03x, which is noise (decision of 2026-11-08). The hot kernels are in ggml
and ONNX Runtime, not in our code.

## Decision

- Use ggml's own mechanism: `GGML_BACKEND_DL` plus `GGML_CPU_ALL_VARIANTS`, which
  builds one ggml CPU library per instruction-set level and selects the best at run
  time. Controlled by the CMake option `LIBEMBEDDING_CPU_DISPATCH`.
- Default ON, except on Apple and on 32-bit Arm (`CMakeLists.txt`: ggml has no variant
  set for armv7; Apple keeps the native build until the Apple Silicon variants are
  validated). OFF builds a single `ggml-cpu` for the build machine.
- Our own code is no longer compiled with AVX2 flags. One generic wheel per platform,
  no "optimised" wheel.
- ONNX Runtime keeps its own dispatch; we do not touch it.
- No patch of `third_party/` (for instance an MSVC variant-selection quirk on Ice Lake
  is left as is, as it has no effect on the code that runs).
- Variants are validated with Intel SDE in CI.

## Consequences

- Wheels are larger: they ship every x86-64 variant module.
- Backend modules must be found at run time next to the library; the fuzz build turns
  dispatch off to keep one statically linked, fully instrumented ggml.
- Any ggml bump is done in a separate worktree and re-validated under SDE.
- A custom SIMD kernel in libembedding needs a measured gain over ggml/ORT before it
  is accepted.
