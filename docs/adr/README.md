---
layout: default
title: Architecture decision records
nav_exclude: true
---

# Architecture decision records

Short records of decisions that are costly to reverse. Each ADR has Context,
Decision, Consequences and Status. A superseded ADR is kept and marked as such.

| ADR | Title | Status |
|---|---|---|
| [ADR-001](ADR-001-routing-by-extension.md) | Backend routing by file extension, no `backend=` | Accepted |
| [ADR-002](ADR-002-cpu-dispatch-ggml.md) | CPU dispatch through the upstream ggml mechanism | Accepted |
| [ADR-003](ADR-003-sparse-gguf-ggml-runtime.md) | In-house ggml runtime for sparse GGUF | Accepted |
| [ADR-004](ADR-004-2-0-removals.md) | Removal list for 2.0 | Accepted |

To add one: copy the structure of an existing ADR, take the next number, and add a
row above.
