---
layout: default
title: ADR-001 Routing by extension
nav_exclude: true
---

# ADR-001: Backend routing by file extension

Status: Accepted (2026-11-08). Amended 2026-11-21: see "Amendment" below.

## Amendment (2026-11-21)

The `backend` field is kept as an official option of the **reranker and dense text**
(`lembed_reranker_options_t.backend`, `lembed_text_options_t.backend`, Python
`Reranker(backend=...)` / `TextEmbedding(backend=...)`): it forces a
backend, and in `AUTO` falls back from ONNX to GGUF when the ONNX files are missing.
Sparse and image have no such field (adding one is an ABI change; sparse would need
a `lembed_sparse_options_v2`). It is no longer slated for
removal in 2.0. The bullets of the Decision section below that call it deprecated or
removed are superseded by this paragraph; the routing by extension is unchanged.

## Context

The library has two backends: ONNX Runtime and llama.cpp (GGUF). Callers should not
need to know which one runs. An explicit `backend` option leaks an implementation
detail into the API and invites inconsistent combinations (a `.onnx` path with the
llama.cpp backend).

Today the C options carry a `backend` field (`lembed_backend_t`:
`LEMBED_BACKEND_ONNX`, `LEMBED_BACKEND_LLAMACPP`, `LEMBED_BACKEND_AUTO`, in
`include/libembedding/types.h`), defaulting to `AUTO`. Python's `Reranker` has a
matching `backend="auto"` argument.

## Decision

- The backend is chosen from the model path or identifier:
  `.gguf` -> llama.cpp; `.onnx`, a Hugging Face identifier or anything unknown ->
  ONNX Runtime (`resolve_effective_backend()` in
  `include/libembedding/detail/session_factory.hpp`).
- No new API takes a `backend=` parameter, in C or Python.
- The existing `backend` field is kept for ABI compatibility and documented as
  deprecated (documentation only, no `deprecated` attribute, which would add warnings
  to our own tests). It is removed in 2.0 (ADR-004).
- A provider requested explicitly but unavailable returns
  `LEMBED_ERROR_UNSUPPORTED`; it does not fall back silently.
- The reranker `AUTO` fallback from ONNX to GGUF stays but must be visible (log and
  introspection).

## Consequences

- Simple mental model: the file you pass decides the runtime.
- A model cannot be forced onto the other backend. Users who need that convert the
  model.
- Until 2.0 the deprecated field stays in the structs and in `_cdefs.h`.
- Backend comparison for one model is done through benchmarks that load both files,
  not through a switch.
