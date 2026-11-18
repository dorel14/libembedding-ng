---
layout: default
title: ADR-003 Sparse GGUF runtime
nav_exclude: true
---

# ADR-003: In-house ggml runtime for sparse GGUF

Status: Accepted (2026-11-08)

## Context

Sparse (SPLADE) embeddings need the masked-language-model head: logits per token,
then log(1 + ReLU) and a max over tokens. llama.cpp does not expose MLM logits for
the BERT architecture (the graph never assigns `res->t_logits`), so
`llama_get_logits_ith()` cannot be used. The dense path uses `llama_encode()` and
`llama_get_embeddings_seq()` and is unaffected.

Patching the vendored llama.cpp is not allowed (`third_party/` is never patched).

## Decision

- Sparse GGUF runs on a runtime of our own, built on ggml and the `gguf` reader:
  header and weights in `include/libembedding/detail/gguf/gguf_weights.hpp`, the BERT
  graph in `gguf_bert_graph.hpp`, the session in `gguf_sparse_session.hpp`, the entry
  point in `include/libembedding/detail/sparse_text_embedding_gguf_impl.hpp`.
- Batches are capped at a 512-token budget (best throughput of 128 to 2048 measured).
- F32 and F16 weight formats are rejected: no end-to-end gain and about 3.6x RSS.
  Quantised weights (for example Q8_0) are dequantised inside `ggml_mul_mat`; code
  never reads a quantised tensor element by element.
- The GGUF is untrusted input: the loader validates bounds, types and size against the
  file, and is fuzzed (see `SECURITY.md`).
- Scope stays at BERT-family encoders. Generalising the runtime to other architectures
  needs its own ADR first.

## Consequences

- We own a loader and a graph: more code to maintain and fuzz.
- Sparse GGUF does not depend on llama.cpp internals, so llama.cpp bumps do not break
  it, but ggml API changes can.
- Equivalence with the ONNX sparse path is checked by tests, not assumed.
