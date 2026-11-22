"""High-level multi-vector (late interaction, ColBERT) embedding API.

One float32 matrix ``[n_tokens, dim]`` per text instead of one vector. Score a
query against a document with :func:`libembedding.maxsim`. This module embeds;
it does not index or search.

Auteur: David Orel
Version: 1.16.0

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence

import numpy as np

from ._binding import ffi, lib
from ._status import check_status
from .models import _PROVIDER_MAP, _c_string, _desc_from_c
from .types import ModelDesc, Stats

_MODES = {
    "document": lib.LEMBED_MV_DOCUMENT,
    "query": lib.LEMBED_MV_QUERY,
}


def _tri_state(value: bool | None) -> int:
    """None = follow the model (-1), otherwise 0 / 1."""
    return -1 if value is None else int(bool(value))


class MultiVectorEmbedding:
    """Per-token embeddings of a ColBERT-style ONNX model (late interaction).

    Args:
        model_dir: Local directory with the ONNX file (``model.onnx``,
            ``vespa_colbert.onnx`` or ``onnx/model.onnx``), ``tokenizer.json``
            and optionally ``onnx_config.json`` / ``colbert_config.json``.
        provider: Execution provider (``"cpu"``).
        threads: Number of threads (0 = auto).
        batch_size: Internal batch size (0 = 32).
        query_maxlen: Query length override (None = model default, 32).
        doc_maxlen: Document length override (None = model default, 300).
        normalize: L2-normalise every vector.
        query_expansion: Pad queries with ``[MASK]`` up to the query length and
            keep those vectors (None = on).
        attend_to_expansion: With expansion, whether the ``[MASK]`` pads are
            attended (None = follow the model).
    """

    def __init__(
        self,
        model_dir: str,
        *,
        provider: str = "cpu",
        threads: int = 0,
        batch_size: int = 0,
        query_maxlen: int | None = None,
        doc_maxlen: int | None = None,
        normalize: bool = True,
        query_expansion: bool | None = None,
        attend_to_expansion: bool | None = None,
    ):
        self._mv = None
        if not isinstance(model_dir, (str, bytes)) and not hasattr(model_dir, "__fspath__"):
            raise TypeError("model_dir must be a path string")
        model_dir = str(model_dir) if not isinstance(model_dir, bytes) else model_dir.decode()
        if not model_dir:
            raise ValueError("model_dir must not be empty")
        prov = _PROVIDER_MAP.get(provider.lower())
        if prov is None:
            raise ValueError(
                f"Unknown provider '{provider}'. Use: {list(_PROVIDER_MAP.keys())}"
            )
        for label, value in (("query_maxlen", query_maxlen), ("doc_maxlen", doc_maxlen)):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be a positive integer or None")
        if threads < 0 or batch_size < 0:
            raise ValueError("threads and batch_size must be >= 0")

        opts = lib.lembed_multivector_options_default()
        dir_buf = ffi.new("char[]", model_dir.encode("utf-8"))
        opts.model_dir = dir_buf
        opts.provider = prov
        opts.num_threads = threads
        opts.batch_size = batch_size
        opts.query_maxlen = 0 if query_maxlen is None else query_maxlen
        opts.doc_maxlen = 0 if doc_maxlen is None else doc_maxlen
        opts.normalize = int(bool(normalize))
        opts.query_expansion = _tri_state(query_expansion)
        opts.attend_to_expansion = _tri_state(attend_to_expansion)

        mv_ptr = ffi.new("lembed_multivector_t **")
        check_status(lib.lembed_multivector_create(ffi.addressof(opts), mv_ptr))
        self._mv = mv_ptr[0]

    def _require_open(self):
        if self._mv is None:
            raise ValueError("MultiVectorEmbedding is closed")
        return self._mv

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return int(lib.lembed_multivector_dim(self._require_open()))

    @property
    def backend(self) -> str:
        """Runtime serving the model (``"onnxruntime"``)."""
        return _c_string(lib.lembed_multivector_backend_name(self._require_open()))

    def embed(
        self, texts: Sequence[str], *, mode: str = "document"
    ) -> list[np.ndarray]:
        """Embed texts into one float32 matrix ``[n_i, dim]`` each.

        Args:
            texts: Strings to embed (a single ``str`` is refused).
            mode: ``"document"`` or ``"query"``.
        """
        mv = self._require_open()
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a list of strings, not a single string")
        c_mode = _MODES.get(mode)
        if c_mode is None:
            raise ValueError(f"Unknown mode '{mode}'. Use: {list(_MODES.keys())}")
        texts = list(texts)
        n = len(texts)
        if n == 0:
            return []

        encoded = [t.encode("utf-8") for t in texts]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_texts = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_multivectors_t *")
        # cffi releases the GIL for the duration of the C call on its own.
        check_status(lib.lembed_multivector_embed(mv, c_texts, n, c_mode, result))
        try:
            dim = result.dim
            out: list[np.ndarray] = []
            for i in range(result.num_docs):
                start = int(result.offsets[i])
                count = int(result.offsets[i + 1]) - start
                if count <= 0:
                    out.append(np.zeros((0, dim), dtype=np.float32))
                    continue
                buf = ffi.buffer(result.data + start * dim, count * dim * 4)
                out.append(
                    np.frombuffer(buf, dtype=np.float32).reshape(count, dim).copy()
                )
            return out
        finally:
            lib.lembed_multivectors_free(result)

    def query_embed(self, query: str | Sequence[str]) -> Iterator[np.ndarray]:
        """Embed one query or a list of queries (FastEmbed naming)."""
        queries = [query] if isinstance(query, str) else list(query)
        yield from self.embed(queries, mode="query")

    def passage_embed(self, texts: Sequence[str]) -> Iterator[np.ndarray]:
        """Embed documents / passages (FastEmbed naming)."""
        yield from self.embed(texts, mode="document")

    def info(self) -> ModelDesc:
        """Return runtime model descriptor."""
        mv = self._require_open()
        return _desc_from_c(
            lib.lembed_multivector_desc(mv), backend=self.backend, effective_provider="cpu"
        )

    def stats(self) -> Stats:
        """Return runtime usage statistics."""
        s = ffi.new("lembed_stats_t *")
        lib.lembed_multivector_stats(self._require_open(), s)
        return Stats(
            texts_embedded=s.texts_embedded,
            batches_run=s.batches_run,
            avg_latency_ms=s.avg_latency_ms,
        )

    def close(self) -> None:
        """Release the model. Idempotent."""
        mv = getattr(self, "_mv", None)
        if mv is not None:
            self._mv = None
            lib.lembed_multivector_free(mv)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        with contextlib.suppress(Exception):  # interpreter shutdown
            self.close()


LateInteractionTextEmbedding = MultiVectorEmbedding
