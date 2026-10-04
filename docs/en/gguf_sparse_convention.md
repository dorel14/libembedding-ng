# GGUF sparse embedding convention

Version: 1.0.0
Status: implemented, covered by `tests/test_gguf_inspect.cpp` (71 assertions)

Defines **what a GGUF file is capable of** — nothing about how it is executed.
This page is the normative reference for that decision; the implementation is
`include/libembedding/gguf_inspect.h`.

---

## 1. Scope, and what it deliberately is not

`lembed_gguf_inspect()` answers one question: *what does this file contain?* It
reads no weights. Opening a file costs a header parse, not a load, so a caller can
classify a model before deciding whether to load it.

It is **not** an inference API. The runtime that consumes these capabilities lives
in P1.5c (`detail/gguf/gguf_sparse_session.hpp`). Keeping the two apart is what
stops the format convention from growing into an inference engine nobody can
review.

---

## 2. Why a convention is needed at all

The GGUF format has no vocabulary for what a head *is*. Converters emit tensors
and stop. The reference sparse implementation documents no metadata key for its
own capability — it derives `has_sparse` from whether `mlm_transform.weight`
exists in the file, and a documented `has_mlm_head` key that llama.cpp itself
never reads.

Three rules follow from that, and every decision below traces back to one of them.

### Rule 1 — capabilities come from tensor presence, never from a name or a flag

Not from an enum, not from a filename, not from a `has_sparse` field.

A flag means trusting each converter in turn. `bert.has_mlm_head` exists in
`cstr/splade-*-GGUF` and is read by nothing; a file advertising it without
shipping the tensor would be believed. Presence cannot lie.

The test suite pins both directions: a file with the tensors and no flag is
recognised, and a file with the flag and no tensors is not.

### Rule 2 — a missing key is reported, never defaulted

A default of 30522 for `vocab_size` does not fail. It produces a plausible,
normalised, entirely wrong sparse vector over the wrong vocabulary, and returns a
success code. That is the failure mode this whole document exists to prevent.

Every reported field carries a `has_*` flag. `has_vocab_size == 0` means *the file
did not say*; it never means 30522. When a key was looked for under the
architecture prefix and not found, its name is listed in `missing_hparams`.

A sparse head **without** a known vocabulary is refused: the capability is
cleared and the formula becomes `UNKNOWN`. It is better to report unusable than to
compute a wrong answer quickly.

### Rule 3 — the two sparse formulas are different algorithms

They share a name and nothing else:

| Path | Formula |
|---|---|
| **SPLADE** (MLM head) | `max` over tokens of `log(1 + relu(vocab_logits))` |
| **BGE-M3 scalar** (`sparse_linear`, `out_dim == 1`) | scatter the per-token weight onto its own token id, `max` over tokens, **no `log(1+x)`, no vocabulary projection** |

They must not share a code path. `lembed_gguf_sparse_formula_t` names which one
applies, and `UNKNOWN` is a legitimate outcome that the caller is expected to
refuse.

---

## 3. Recognised heads

Names are the ones observed in the files examined in
`benchmarks/sparse/LE-9.4-validation.md`.

| Tensors | Capability | Meaning |
|---|---|---|
| `token_embd.weight` | *required* | Without it the file is not a text encoder |
| *(none)* | `DENSE` | The encoder always yields one pooled vector per input |
| `mlm_transform.weight` + `.bias` | `SPLADE` | `GELU(W·h + b)`, dense-to-dense |
| `mlm_ln.weight` + `.bias` | *(refines `SPLADE`)* | LayerNorm after the transform |
| `mlm_bias` | `SPLADE_BIAS` | `[vocab_size]` decoder bias, optional |
| `sparse_linear.weight` + `.bias` | `SPARSE_LINEAR` | BGE-M3 style; shape decides the formula |
| `colbert_linear.weight` + `.bias` | `COLBERT` | One vector per token, not per input |
| `classifier.dense.weight` + `classifier.out_proj.weight` | `RERANKER` | Cross-encoder reranking head |
| `mlm_bias` absent, `mlm_head.weight` absent | `TIED_DECODER` | The vocabulary projection is `token_embd.weight` |

### The tied decoder

`mlm_bias` is stored but the 30522 x 768 projection is **not**: it is tied to
`token_embd.weight` `[768, 30522]`. That is standard BERT MLM weight tying, and it
is why a SPLADE head costs almost nothing on disk. An implementation that looks
for a separate vocabulary matrix will not find one and must fall back to
`token_embd`.

---

## 4. Hyper-parameters

Read as `<architecture>.<field>`, where `<architecture>` is
`general.architecture`. Nothing outside this layer knows an architecture name.

| Key | Field | Notes |
|---|---|---|
| `<arch>.vocab_size` | `vocab_size` | **Required for any sparse claim** |
| `<arch>.embedding_length` | `embedding_length` | Hidden width |
| `<arch>.block_count` | `block_count` | Encoder depth |
| `<arch>.context_length` or `<arch>.max_position_embeddings` | `context_length` | Both spellings are accepted |
| `<arch>.colbert_dim` | `colbert_dim` | Present only when `COLBERT` is |

Special-token ids come from the tokenizer metadata, never from a list:

| Key | Field |
|---|---|
| `tokenizer.ggml.pad_token_id` | `pad_token_id` |
| `tokenizer.ggml.bos_token_id` | `bos_token_id` |
| `tokenizer.ggml.eos_token_id` | `eos_token_id` |
| `tokenizer.ggml.cls_token_id` | `cls_token_id` |
| `tokenizer.ggml.separator_token_id` | `separator_token_id` |

These are vocabulary-specific and a hardcoded list is wrong in both directions.
BERT has `[CLS]`=101 and `[SEP]`=102; XLM-R has `<s>`=0, `</s>`=2, `<pad>`=1. The
reference implementation reads these ids from the metadata and then filters a
hardcoded `{0, 101, 102}` anyway — so on an XLM-R vocabulary `<pad>` and `</s>`
leak straight into the sparse vector. Do not copy that.

---

## 5. Integer types are read permissively

A count may be written `UINT32` or `INT32`; a flag may be `BOOL` or `UINT8`.
Readers here accept either spelling and range-check the unsigned path. Rejecting
the unsigned spelling would report a perfectly good file as missing its
vocabulary — the same class of bug as Rule 2, reached by a different route.

---

## 6. Capabilities are a bitmask, not an enum

```c
typedef uint64_t lembed_gguf_capabilities_t;
#define LEMBED_GGUF_CAP_DENSE          (1 << 0)
#define LEMBED_GGUF_CAP_SPLADE         (1 << 1)
#define LEMBED_GGUF_CAP_SPLADE_BIAS    (1 << 2)
#define LEMBED_GGUF_CAP_SPARSE_LINEAR  (1 << 3)
#define LEMBED_GGUF_CAP_COLBERT        (1 << 4)
#define LEMBED_GGUF_CAP_RERANKER       (1 << 5)
#define LEMBED_GGUF_CAP_TIED_DECODER   (1 << 6)
#define LEMBED_GGUF_CAP_SPECIAL_TOKENS (1 << 7)
```

A bitmask, not an enum, for one reason: an enum forces every new head type to
renumber values already published in the ABI. A bitmask lets a head be added in a
later minor release without touching the ones already shipped, and an older reader
ignores unknown bits — which is the forward compatibility a converter-extended
format needs.

`lembed_gguf_capability_name()` returns `"unknown"` for a bit it does not
recognise, rather than falling through to a neighbouring name.

---

## 7. Adding a head

The head table is data, not code:

```cpp
// detail/gguf/gguf_spec.hpp
inline constexpr CapabilityRule kCapabilityRules[] = {
    {"mlm_transform.weight", "mlm_bias",       LEMBED_GGUF_CAP_SPLADE},
    {"sparse_linear.weight", nullptr,          LEMBED_GGUF_CAP_SPARSE_LINEAR},
    {"colbert_linear.weight", nullptr,         LEMBED_GGUF_CAP_COLBERT},
    {"classifier.dense.weight",
     "classifier.out_proj.weight",             LEMBED_GGUF_CAP_RERANKER},
};
```

Adding a head means **a row and a bit**. It does not mean editing a
classification switch, and no caller changes. The classification loop, the report
and the summary are already generic over the table.

---

## 8. How the reader is built

Two layers, one responsibility each. The binary format is parsed by the ggml
reader already vendored with llama.cpp, which libembedding links against.

| Layer | File | Responsibility |
|---|---|---|
| Probe | `detail/gguf/gguf_probe.hpp` | Typed, type-checked access over `gguf.h`; arch-prefixed key lookup; owns the context. **No semantics.** |
| Spec | `detail/gguf/gguf_spec.hpp` | The convention: the head table, the hparams, the capability rules, the refusals. **No parsing.** |
| Entry | `detail/gguf/gguf_inspect_entry.hpp` | The three C functions. |

A fourth parser inside this project would be a second source of truth for the same
binary layout, free to drift from the first. The probe adds only what `gguf.h`'s
C API cannot express: typed reads that *reject a mismatched type* rather than
returning whatever is in memory, and `"<arch>.<field>"` key construction.

The probe opens with `no_alloc = true`: it wants names, types and scalars, never
tensor contents. Allocating the data blob would read gigabytes to answer a question
about a few hundred header bytes.

---

## 9. Behaviour on files that cannot be used

`lembed_gguf_inspect()` returns `LEMBED_OK` for a readable file that turns out to
be unusable, and reports the refusal in `diagnostic` and `capabilities`. It returns
an error only when the path cannot be read at all.

This matters for directory scanning: a caller walking a model cache keeps going
and still gets the descriptions of the files it *can* read, instead of the first
unusable file aborting the walk.

`lembed_gguf_capability_summary()` reduces a description to one word:

| Result | Condition |
|---|---|
| `"sparse"` | a sparse capability **and** a decided formula |
| `"dense"` | dense available, no usable sparse path |
| `"unsupported"` | no encoder, or a diagnostic with no capability |

---

## 10. Reference files

Shapes recorded from the real artefacts, used to build the test fixtures:

| File | Result |
|---|---|
| `cstr/splade-pp-en-v1-q8_0.gguf` | `sparse`, SPLADE formula, tied decoder, bias present |
| `cstr/splade-v3-q8_0.gguf` | identical shape, different model — the reader must not key on a name |
| `mradermacher/opensearch-neural-sparse-encoding-doc-v2-mini-GGUF` | **not sparse** — the conversion dropped the projection, so `mlm_*` is absent |
| a llama.cpp-converted BERT export (`all-MiniLM-L6-v2-Q4_K_M`) | `dense` — it carries **no** `bert.vocab_size` at all, so no sparse claim is possible |

That third row is the reason the convention exists: without a sparse head the file
is a perfectly good **dense** encoder, and accepting it where a sparse vector was
requested is the failure this page is written to prevent. The fourth row is the
same rule observed on a real conversion rather than a fixture.

> **`splade-v3-GGUF` is under CC-BY-NC-SA-4.0** and is therefore excluded from
> the registry. It is listed here only as a shape reference. The usable candidate
> is `cstr/splade-pp-en-v1-GGUF` (Apache-2.0).

---

## 11. Python

```python
from libembedding import inspect_gguf

d = inspect_gguf("model.gguf")
d.summary            # "sparse" | "dense" | "unsupported"
d.is_sparse          # True only when the formula is decided
d.vocab_size         # None when the file did not say -- never a guess
d.special_token_ids  # from tokenizer.ggml.*_token_id, not a hardcoded list
d.capability_names()
```

---

## 12. Related

- `benchmarks/sparse/LE-9.4-validation.md` — the investigation that produced this
  convention, including the llama.cpp blocker and the CrispEmbed architecture
  reading
- `roadmap-2.md` § P1.5b (this item), § P1.5c (the runtime that consumes it)