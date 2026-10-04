"""
Inspect a GGUF file: what it is capable of, read-only.

This module answers "what does this file contain?" and nothing else. It reads no
weights, so a caller can classify a model before deciding whether to load it --
which matters because a GGUF without a sparse head is a perfectly good *dense*
encoder, and accepting it where a sparse vector was asked for is the mistake this
API is built to prevent.

The convention is documented in ``docs/gguf_sparse_convention.md``.

Author: David Orel
Version: 1.10.1

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._binding import ffi, lib
from ._status import check_status

__all__ = [
    "SPARSE_FORMULA_SCALAR",
    "SPARSE_FORMULA_SPLADE",
    "SPARSE_FORMULA_UNKNOWN",
    "GgufCapabilities",
    "GgufDesc",
    "inspect_gguf",
]

SPARSE_FORMULA_SPLADE = lib.LEMBED_GGUF_SPARSE_FORMULA_SPLADE
SPARSE_FORMULA_SCALAR = lib.LEMBED_GGUF_SPARSE_FORMULA_SCALAR
SPARSE_FORMULA_UNKNOWN = lib.LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN

_SPARSE_FORMULA_NAMES = {
    SPARSE_FORMULA_SPLADE: "splade",
    SPARSE_FORMULA_SCALAR: "scalar",
    SPARSE_FORMULA_UNKNOWN: "unknown",
}

_CAP_NAMES = [
    ("dense", lib.LEMBED_GGUF_CAP_DENSE),
    ("sparse", lib.LEMBED_GGUF_CAP_SPLADE),
    ("sparse_bias", lib.LEMBED_GGUF_CAP_SPLADE_BIAS),
    ("sparse_linear", lib.LEMBED_GGUF_CAP_SPARSE_LINEAR),
    ("colbert", lib.LEMBED_GGUF_CAP_COLBERT),
    ("reranker", lib.LEMBED_GGUF_CAP_RERANKER),
    ("tied_decoder", lib.LEMBED_GGUF_CAP_TIED_DECODER),
    ("special_tokens", lib.LEMBED_GGUF_CAP_SPECIAL_TOKENS),
]


class GgufCapabilities:
    """Capability bits, as named constants.

    A bitmask rather than an enum so a head added in a later release does not
    renumber the ones already published.
    """

    NONE = lib.LEMBED_GGUF_CAP_NONE
    DENSE = lib.LEMBED_GGUF_CAP_DENSE
    SPLADE = lib.LEMBED_GGUF_CAP_SPLADE
    SPLADE_BIAS = lib.LEMBED_GGUF_CAP_SPLADE_BIAS
    SPARSE_LINEAR = lib.LEMBED_GGUF_CAP_SPARSE_LINEAR
    COLBERT = lib.LEMBED_GGUF_CAP_COLBERT
    RERANKER = lib.LEMBED_GGUF_CAP_RERANKER
    TIED_DECODER = lib.LEMBED_GGUF_CAP_TIED_DECODER
    SPECIAL_TOKENS = lib.LEMBED_GGUF_CAP_SPECIAL_TOKENS


@dataclass
class GgufDesc:
    """What a GGUF file contains.

    Every hyper-parameter is reported as present or absent rather than defaulted:
    a guessed vocabulary size does not fail, it produces a plausible vector over
    the wrong vocabulary. ``has_vocab_size=False`` means the file did not say; it
    never means 30522.
    """

    architecture: str
    name: str
    capabilities: int
    sparse_formula: str
    n_tensors: int

    vocab_size: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    context_length: int | None = None
    colbert_dim: int | None = None

    special_token_ids: dict[str, int] = field(default_factory=dict)

    #: Keys looked for under "<arch>." and not found, '|'-joined.
    missing_hparams: str = ""
    #: Why the file is unusable; empty when it is fine.
    diagnostic: str = ""

    @property
    def is_sparse(self) -> bool:
        """True when a sparse vector is computable *and* the formula is decided."""
        return self.summary == "sparse"

    @property
    def is_dense(self) -> bool:
        return bool(self.capabilities & GgufCapabilities.DENSE)

    @property
    def summary(self) -> str:
        """``"sparse"``, ``"dense"`` or ``"unsupported"``."""
        if self.diagnostic and self.capabilities == GgufCapabilities.NONE:
            return "unsupported"
        sparse = self.capabilities & (
            GgufCapabilities.SPLADE | GgufCapabilities.SPARSE_LINEAR
        )
        if sparse and self.sparse_formula != "unknown":
            return "sparse"
        return "dense" if self.is_dense else "unsupported"

    def capability_names(self) -> list[str]:
        """Names of the set capability bits."""
        return [name for name, bit in _CAP_NAMES if self.capabilities & bit]

    def __str__(self) -> str:
        bits = ", ".join(self.capability_names()) or "none"
        return (
            f"{self.summary} [{bits}] arch={self.architecture or '?'} "
            f"vocab={self.vocab_size if self.has_value('vocab') else '?'} "
            f"tensors={self.n_tensors}"
        )

    def has_value(self, field_name: str) -> bool:
        """Whether a hyper-parameter was actually present in the file."""
        return getattr(self, field_name, None) is not None


def _maybe(value: int, present: int) -> int | None:
    """A reported integer, or None when the file did not carry it."""
    return int(value) if present else None


def inspect_gguf(path: str) -> GgufDesc:
    """Inspect a GGUF file without loading its weights.

    Raises on a file that cannot be read at all. A readable file that turns out to
    be unusable is returned with ``diagnostic`` set rather than raised, so walking
    a directory keeps going instead of stopping at the first bad file.
    """
    desc = ffi.new("lembed_gguf_desc_t *")
    check_status(lib.lembed_gguf_inspect(str(path).encode("utf-8"), desc))

    specials = {
        "pad": _maybe(desc.pad_token_id, desc.has_pad_token_id),
        "bos": _maybe(desc.bos_token_id, desc.has_bos_token_id),
        "eos": _maybe(desc.eos_token_id, desc.has_eos_token_id),
        "cls": _maybe(desc.cls_token_id, desc.has_cls_token_id),
        "separator": _maybe(desc.separator_token_id, desc.has_separator_token_id),
    }

    return GgufDesc(
        architecture=ffi.string(desc.architecture).decode("utf-8", "replace"),
        name=ffi.string(desc.name).decode("utf-8", "replace"),
        capabilities=int(desc.capabilities),
        sparse_formula=_SPARSE_FORMULA_NAMES.get(
            int(desc.sparse_formula), "unknown"
        ),
        n_tensors=int(desc.n_tensors),
        vocab_size=_maybe(desc.vocab_size, desc.has_vocab_size),
        embedding_length=_maybe(desc.embedding_length, desc.has_embedding_length),
        block_count=_maybe(desc.block_count, desc.has_block_count),
        context_length=_maybe(desc.context_length, desc.has_context_length),
        colbert_dim=_maybe(desc.colbert_dim, desc.has_colbert_dim),
        special_token_ids={k: v for k, v in specials.items() if v is not None},
        missing_hparams=ffi.string(desc.missing_hparams).decode("utf-8", "replace"),
        diagnostic=ffi.string(desc.diagnostic).decode("utf-8", "replace"),
    )