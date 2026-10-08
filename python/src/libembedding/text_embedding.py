"""High-level text embedding API."""

# pyright: reportAttributeAccessIssue=false,reportCallIssue=false
from __future__ import annotations

import os
import warnings

import numpy as np  # pyright: ignore[reportMissingImports]

from ._binding import (  # pyright: ignore[reportAttributeAccessIssue,reportMissingImports]
    ffi,
    lib,
)
from ._status import check_status
from .exceptions import LembedError, ModelNotFoundError
from .models import (
    _POOLING_ENUM,
    _PROVIDER_MAP,
    _QUANTIZATION_ENUM,
    _QUANTIZATION_NAMES,
    _desc_from_c,
    _is_gguf_model,
    _is_local_path,
    resolve_text_model,
)
from .types import ModelDesc, Stats

_MODE_TO_MODEL = {
    "fast": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    "balanced": "BAAI/bge-small-en-v1.5",
    "quality": "BAAI/bge-base-en-v1.5",
}

# The C core already L2-normalises every vector before returning it, so
# `normalized=True` is only meaningful for callers that bypassed that path.
# It is kept for API symmetry and is a no-op on already-unit vectors.
_VALID_DTYPES = ("float32", "float16")


def _validate_output_dtype(dtype: str) -> str:
    """Validate the dtype requested by the caller."""
    if dtype not in _VALID_DTYPES:
        raise ValueError(
            f"Unsupported dtype: {dtype!r}. Expected one of {list(_VALID_DTYPES)}."
        )
    return dtype


class TextEmbedding:
    """Generate dense vector embeddings from text.

    Args:
        model_name: HuggingFace model code (e.g. "BAAI/bge-small-en-v1.5")
            or path to a local ONNX model directory.
        provider: Execution provider ("cpu", "cuda", "tensorrt", "rocm").
        threads: Number of threads for inference.
        batch_size: Batch size for inference.
        offline: If True, skip downloads (use cache only).
        show_download_progress: If True, show download progress bar.
        cache_dir: Custom cache directory.
        max_length: Maximum sequence length.
        dim: Embedding dimension (0 = auto).
        pooling: Pooling strategy ("mean", "cls", "max").
        auto_workers: If True, auto-detect optimal worker/session count
            for llama.cpp backend.
        cache_size: Size of LRU embedding cache (0 = disabled).
        quantization: Override model quantization mode ("none", "static",
            "dynamic"). If None, uses the model's default.
        preferred_quantization: Auto-select best quantization ("auto"),
            or one of "none", "static", "dynamic".
        num_threads: Deprecated; use ``threads``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        provider: str = "cpu",
        threads: int = 0,
        batch_size: int = 256,
        offline: bool = False,
        show_download_progress: bool = True,
        cache_dir: str | None = None,
        max_length: int = 0,
        dim: int = 0,
        pooling: str = "mean",
        auto_workers: bool = False,
        cache_size: int = 0,
        quantization: str | None = None,
        preferred_quantization: str | None = None,
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        quant_enum = lib.LEMBED_QUANTIZATION_NONE
        if quantization is not None:
            quant_enum = _QUANTIZATION_ENUM.get(quantization.lower())
            if quant_enum is None:
                raise ValueError(
                    f"Unknown quantization '{quantization}'. Use: {list(_QUANTIZATION_ENUM.keys())}"
                )

        self._ctx = None
        self._dim = 0
        self._batch_size = batch_size
        self._quantization = "none"
        self._quantization_reason = ""

        # Resolve preferred_quantization
        resolved_quant = quant_enum
        if preferred_quantization is not None and preferred_quantization != "none":
            if preferred_quantization == "auto":
                resolved_quant, self._quantization_reason = (
                    self._auto_select_quantization(
                        model_name, provider, threads, batch_size, cache_dir,
                        max_length, dim, pooling, offline, show_download_progress,
                        cache_size, auto_workers,
                    )
                )
                if resolved_quant is None:
                    resolved_quant = lib.LEMBED_QUANTIZATION_NONE
            else:
                pq_enum = _QUANTIZATION_ENUM.get(preferred_quantization.lower())
                if pq_enum is not None:
                    resolved_quant = pq_enum

        # Remember which mode was actually selected: with
        # preferred_quantization="auto" the caller asked for a decision, and
        # nothing else exposes the outcome.
        self._quantization = _QUANTIZATION_NAMES.get(resolved_quant, "none")

        if _is_gguf_model(model_name):
            # GGUF model: use llama.cpp backend
            # Try the local file first, like Reranker does. Order matters: the
            # downloader hands out mixed-separator absolute paths on Windows
            # ("C:\Users\<user>/.cache/libembedding/.../model.gguf"), and splitting
            # such a path on the first "/" yields repo "C:\Users\<user>" plus a
            # *relative* filename. os.path.isfile() then misses the file that is
            # sitting right there, and we take the download branch and fail.
            local_path = os.path.normpath(model_name)
            if os.path.isfile(local_path):
                repo = ""
                filename = local_path
            elif "/" in model_name:
                # "<repo>/<filename>" shorthand
                parts = model_name.split("/", 1)
                repo = parts[0]
                filename = parts[1]
            else:
                raise FileNotFoundError(f"GGUF model not found: '{model_name}'")

            opts = ffi.new("lembed_text_options_t *")
            opts.provider = _PROVIDER_MAP.get(provider, 0)
            opts.num_threads = threads
            opts.batch_size = batch_size
            opts.max_length = max_length
            opts.dim = dim
            opts.pooling = _POOLING_ENUM.get(pooling, 0)
            opts.offline = 1 if offline else 0
            opts.show_download_progress = 1 if show_download_progress else 0
            opts.auto_workers = 1 if auto_workers else 0
            opts.cache_size = cache_size

            if os.path.isfile(filename):
                # Local GGUF file
                ctx_ptr = ffi.new("lembed_text_embedding_t **")
                check_status(
                    lib.lembed_text_embedding_create_from_gguf_path(
                        filename.encode("utf-8"), opts, ctx_ptr
                    )
                )
                self._ctx = ctx_ptr[0]
            else:
                # Download from HuggingFace
                ctx_ptr = ffi.new("lembed_text_embedding_t **")
                check_status(
                    lib.lembed_text_embedding_create_from_gguf_model(
                        repo.encode("utf-8"), filename.encode("utf-8"), opts, ctx_ptr
                    )
                )
                self._ctx = ctx_ptr[0]
        else:
            # ONNX backend (default)
            if _is_local_path(model_name):
                # Local ONNX model directory
                model_dir = model_name
            else:
                # HuggingFace model code: ensure it is cached
                idx = resolve_text_model(model_name)
                if idx < 0:
                    raise ModelNotFoundError(7, f"Unknown model: {model_name}")
                info = ffi.new("lembed_model_info_t *")
                lib.lembed_get_text_model_info(idx, info)

                model_dir = ffi.new("char **")
                check_status(
                    lib.lembed_ensure_text_model(
                        idx,
                        cache_dir.encode("utf-8") if cache_dir else ffi.NULL,
                        1 if show_download_progress else 0,
                        1 if offline else 0,
                        model_dir,
                    )
                )
                model_dir_ptr = model_dir[0]
                model_dir = ffi.string(model_dir_ptr).decode("utf-8")
                lib.lembed_free_string(model_dir_ptr)

            if _is_local_path(model_name):
                opts = ffi.new("lembed_text_options_t *")
                opts.provider = _PROVIDER_MAP.get(provider, 0)
                opts.num_threads = threads
                opts.batch_size = batch_size
                opts.max_length = max_length
                opts.dim = dim
                opts.pooling = _POOLING_ENUM.get(pooling, 0)
                opts.offline = 1 if offline else 0
                opts.show_download_progress = 1 if show_download_progress else 0
                opts.auto_workers = 1 if auto_workers else 0
                opts.cache_size = cache_size
                ctx_ptr = ffi.new("lembed_text_embedding_t **")
                check_status(
                    lib.lembed_text_embedding_create_from_path(
                        model_dir.encode("utf-8"), opts, ctx_ptr
                    )
                )
            else:
                opts = ffi.new("lembed_text_options_v2_t *")
                # Registry index the session is built from. Without it the
                # zero-initialised struct points at entry 0, so every
                # HuggingFace name silently loaded all-MiniLM-L6-v2.
                opts.base.model = resolve_text_model(model_name)
                opts.base.provider = _PROVIDER_MAP.get(provider, 0)
                opts.base.num_threads = threads
                opts.base.batch_size = batch_size
                opts.base.max_length = max_length
                opts.base.dim = dim
                opts.base.pooling = _POOLING_ENUM.get(pooling, 0)
                opts.base.offline = 1 if offline else 0
                opts.base.show_download_progress = 1 if show_download_progress else 0
                opts.base.auto_workers = 1 if auto_workers else 0
                opts.base.cache_size = cache_size
                opts.quantization = resolved_quant
                ctx_ptr = ffi.new("lembed_text_embedding_t **")
                check_status(lib.lembed_text_embedding_create_v2(opts, ctx_ptr))
            self._ctx = ctx_ptr[0]

        self._dim = lib.lembed_text_embedding_dim(self._ctx)
        self._model_name = model_name

    @staticmethod
    def _auto_select_quantization(
        model_name: str, provider: str, threads: int, batch_size: int,
        cache_dir: str | None, max_length: int, dim: int, pooling: str,
        offline: bool, show_download_progress: bool,
        cache_size: int, auto_workers: bool,
    ) -> tuple[int | None, str]:
        """Benchmark the quantization variants this model actually ships.

        Thin delegation to the C implementation, which is the one that matters:
        it warms up separately from the timed run, sizes the corpus, skips
        variants whose weights are absent instead of failing, and caches the
        decision per (model, machine, library version).

        The previous Python implementation benchmarked five texts with no warmup
        and no cache, so it measured session construction rather than throughput,
        re-ran on every single construction, and probed "static" -- a mode the
        registry ships no variant for, so two of its three attempts always threw.

        Returns ``(resolved quantization, reason)``, or ``(None, "")`` when nothing could
        be measured, in which case the caller falls back to FP32.

        The reason travels back in the return value rather than through a class
        attribute: a staticmethod cannot set instance state, and stashing it on
        the class meant a constructor that bailed out early read whatever the
        previous one had left there -- so two threads building models
        concurrently could report each other's rationale.
        """
        try:
            model_index = resolve_text_model(model_name)
        except (LembedError, ValueError):
            return (None, "")

        choice = ffi.new("lembed_quantization_choice_t *")
        try:
            check_status(
                lib.lembed_quantization_auto_select(
                    model_index,
                    int(threads),
                    int(batch_size),
                    # num_docs: the C side clamps this to a sane range.
                    0,
                    # dry_run = 0, so this call is the one that persists the
                    # decision. create_v2 is handed the resolved mode below, so
                    # it never re-measures; if this passed dry_run instead, the
                    # decision would be thrown away and every construction would
                    # benchmark again.
                    0,
                    choice,
                )
            )
        except (LembedError, OSError):
            return (None, "")

        if choice.num_measured <= 0:
            return (None, "")
        reason = ffi.string(choice.reason).decode("utf-8", "replace")
        return (int(choice.quantization), reason)

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def quantization(self) -> str:
        """Quantization mode actually in use: "none", "static", "dynamic" or "fp16".

        With ``preferred_quantization="auto"`` this reports the mode that was
        selected, not the request.
        """
        return self._quantization

    @property
    def quantization_reason(self) -> str:
        """Why ``preferred_quantization="auto"`` picked that mode.

        Empty unless auto-selection ran for this construction: an explicit mode
        has nothing to justify, and a model with no quantized sibling resolves
        to itself without measuring anything.
        """
        return self._quantization_reason

    @property
    def batch_size(self) -> int:
        """Batch size for inference."""
        return self._batch_size

    @property
    def model_name(self) -> str:
        """Model name or path."""
        return self._model_name

    @property
    def name(self) -> str:
        """Model name alias."""
        return self._model_name

    def info(self) -> ModelDesc:
        """Get runtime model descriptor."""
        desc_ptr = lib.lembed_text_embedding_desc_v2(self._ctx)
        return _desc_from_c(desc_ptr.base, desc_ptr)

    @classmethod
    def from_mode(cls, mode: str = "balanced", **kwargs):
        """Create TextEmbedding from a quality/speed mode.

        Args:
            mode: One of "fast", "balanced", "quality".
            **kwargs: Additional arguments passed to TextEmbedding.

        Returns:
            TextEmbedding instance.
        """
        mode = mode.lower()
        if mode not in _MODE_TO_MODEL:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from {list(_MODE_TO_MODEL.keys())}"
            )
        return cls(_MODE_TO_MODEL[mode], **kwargs)

    def embed(self, texts: list[str], *, batch_size: int | None = None,
              dtype: str = "float32", normalized: bool = False) -> np.ndarray:
        """Embed texts into dense vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size override (None = use constructor default).
            dtype: Output dtype, "float32" or "float16".
            normalized: L2 normalize before returning.

        Returns:
            numpy array of shape (len(texts), dim).
        """
        n = len(texts)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        _validate_output_dtype(dtype)

        c_texts = ffi.new("char*[]", n)
        c_strs = []
        for i, t in enumerate(texts):
            c_strs.append(ffi.new("char[]", t.encode("utf-8")))
            c_texts[i] = c_strs[i]

        result = ffi.new("lembed_embeddings_t *")
        bs = self._batch_size if batch_size is None else batch_size
        check_status(lib.lembed_text_embedding_embed(self._ctx, c_texts, n, bs, result))

        try:
            dim = result.dim
            total = result.num_embeddings * dim
            arr = np.frombuffer(
                ffi.buffer(result.data, total * 4), dtype=np.float32
            ).copy()
        finally:
            lib.lembed_embeddings_free(result)

        arr = arr.reshape(n, dim)
        if normalized:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            arr = arr / norms
        if dtype == "float16":
            arr = arr.astype(np.float16)
        return arr

    def embed_bytes(self, texts: list[str], *, dtype: str = "float32",
                    normalized: bool = False) -> bytes:
        """Retourne les embeddings comme bloc mémoire brut.

        Args:
            texts: List of strings to embed.
            dtype: Output dtype, "float32" or "float16".
            normalized: L2 normalize before returning.

        Returns:
            bytes containing the embeddings (no numpy overhead).
        """
        n = len(texts)
        if n == 0:
            return b""

        _validate_output_dtype(dtype)

        c_texts = ffi.new("char*[]", n)
        c_strs = []
        for i, t in enumerate(texts):
            c_strs.append(ffi.new("char[]", t.encode("utf-8")))
            c_texts[i] = c_strs[i]

        result = ffi.new("lembed_embeddings_t *")
        check_status(lib.lembed_text_embedding_embed(self._ctx, c_texts, n, 0, result))

        try:
            dim = result.dim
            total = result.num_embeddings * dim
            arr = np.frombuffer(
                ffi.buffer(result.data, total * 4), dtype=np.float32
            ).copy()
        finally:
            lib.lembed_embeddings_free(result)

        if normalized:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            arr = arr / norms
        if dtype == "float16":
            arr = arr.astype(np.float16)
        return arr.tobytes()

    def embed_iter(self, texts: list[str], *, dtype: str = "float32",
                   normalized: bool = False):
        """Yield one vector per text, without allocating a global array.

        Args:
            texts: List of strings to embed.
            dtype: Output dtype, "float32" or "float16".
            normalized: L2 normalize before yielding.

        Yields:
            numpy array of shape (dim,) for each text.
        """
        n = len(texts)
        if n == 0:
            return

        _validate_output_dtype(dtype)

        c_texts = ffi.new("char*[]", n)
        c_strs = []
        for i, t in enumerate(texts):
            c_strs.append(ffi.new("char[]", t.encode("utf-8")))
            c_texts[i] = c_strs[i]

        result = ffi.new("lembed_embeddings_t *")
        check_status(lib.lembed_text_embedding_embed(self._ctx, c_texts, n, 0, result))

        dim = result.dim
        try:
            for i in range(n):
                # cffi pointer arithmetic: result.data is a float*, so the
                # offset is expressed in floats (not bytes). Slicing a cdata
                # pointer would return a Python list, not a buffer.
                base = result.data + i * dim
                vec = np.frombuffer(
                    ffi.buffer(base, dim * 4), dtype=np.float32
                ).copy()
                if normalized:
                    norm = float(np.linalg.norm(vec))
                    if norm > 0.0:
                        vec = vec / norm
                if dtype == "float16":
                    vec = vec.astype(np.float16)
                yield vec
        finally:
            lib.lembed_embeddings_free(result)

    def embed_stream(
        self, texts: list[str], callback, batch_size: int | None = None
    ) -> None:
        """Embed texts as a stream.

        Args:
            texts: List of strings to embed.
            callback: Called for each embedding with (array, dim, userdata).
            batch_size: Batch size override (None = use constructor default).
        """
        n = len(texts)
        if n == 0:
            return

        c_texts = ffi.new("char*[]", n)
        c_strs = []
        for i, t in enumerate(texts):
            c_strs.append(ffi.new("char[]", t.encode("utf-8")))
            c_texts[i] = c_strs[i]

        bs = self._batch_size if batch_size is None else batch_size

        @ffi.callback("void(const float*, int, void*)")
        def cb(data, dim, userdata):
            arr = np.frombuffer(ffi.buffer(data, dim * 4), dtype=np.float32).copy()
            callback(arr, dim, userdata)

        lib.lembed_text_embedding_embed_stream(self._ctx, c_texts, n, bs, cb, ffi.NULL)

    def stats(self) -> Stats:
        """Get runtime statistics."""
        result = ffi.new("lembed_stats_v2_t *")
        lib.lembed_text_embedding_stats_v2(self._ctx, result)
        return Stats(
            texts_embedded=result.base.texts_embedded,
            batches_run=result.base.batches_run,
            avg_latency_ms=result.base.avg_latency_ms,
            cache_hits=result.cache_hits,
            cache_misses=result.cache_misses,
        )

    def embed_batched(self, texts: list[str], batch_size: int | None = None):
        """Embed texts in batches, yielding numpy arrays."""
        n = len(texts)
        if n == 0:
            return

        bs = self._batch_size if batch_size is None else batch_size
        if bs <= 0:
            bs = 32

        for i in range(0, n, bs):
            batch = texts[i : i + bs]
            embeddings = self.embed(batch, batch_size=bs)
            yield from embeddings

    def close(self) -> None:
        """Release the underlying C resources."""
        if self._ctx is not None:
            lib.lembed_text_embedding_free(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"TextEmbedding(dim={self._dim})"


# Re-export TextEmbeddingPool from pool module (imported last: pool.py imports
# TextEmbedding from this module, so the import cannot stay at the top).
from .pool import TextEmbeddingPool  # noqa: F401
