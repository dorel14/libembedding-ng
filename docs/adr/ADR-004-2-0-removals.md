---
layout: default
title: ADR-004 2.0 removals
nav_exclude: true
---

# ADR-004: Removal list for 2.0

Status: Accepted (2026-11-08). The list is open until 2.0 is cut.

## Context

Several public symbols and fields are deprecated or kept only for ABI stability.
Removing any of them breaks the ABI, so all removals are grouped into one major
version, not scattered over 1.x. 2.0 does not start before the end of phase 3 of the
roadmap (performance and platforms).

Deprecation is documentation only: no `deprecated` attribute, because it would
produce warnings in our own tests.

## Decision

Remove in 2.0 (each symbol verified in the repository):

| Item | Where | Replacement |
|---|---|---|
| ~~`backend` field and `lembed_backend_t` (`LEMBED_BACKEND_*`) in options; Python `Reranker(backend=...)`~~ **Withdrawn 2026-11-21**: kept as an official option of the reranker and of dense text (ADR-001, amendment). `lembed_text_options_t.backend` is honoured: an explicit backend that does not match the path returns `LEMBED_ERROR_UNSUPPORTED` (`detail/text_embedding_impl.hpp`). | `include/libembedding/types.h` | none |
| `lembed_cache_get()` (borrowed pointer, racy) | `include/libembedding/embedding_cache.h` | `lembed_cache_get_copy()` |
| `lembed_sparse_best_config()` | `include/libembedding/sparse_text_embedding.h` | `lembed_sparse_autotune()` (`autotuner.h`) |
| Registry entries under a non-MIT-compatible licence: `jinaai/jina-reranker-v2-base-multilingual` (CC-BY-NC-4.0), `onnx-community/embeddinggemma-300m-ONNX` (Gemma Terms), with their enum values | `include/libembedding/model_registry.h` | MIT/Apache-2.0 replacements, validated (fidelity, Recall@K) |
| `lembed_autotune_unified()`, `lembed_autotune_unified_config()`, `lembed_autotune_unified_clear_cache()` facades | `include/libembedding/autotuner.h` | `lembed_autotune_ex()` |

Until 2.0 the restricted-licence entries stay in the enums, since removing them would
shift indices, and are refused at load time unless
`LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1` is set.

## Consequences

- 1.x keeps every symbol above, with a deprecation note in the headers and docs.
- Removals in 2.0 must be mirrored in `_cdefs.h`, `libembedding.def` and
  `test_struct_sizes`.
- New items are added to this table when they are deprecated, not after.
- `lembed_tune_cache_*` is kept (it is not on this list).
