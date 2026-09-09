"""High-level text embedding API."""

from __future__ import annotations

import warnings

import numpy as np

from ._binding import ffi, lib
from ._status import check_status
from .models import (
    resolve_text_model,
    list_text_models,
    _PROVIDER_MAP,
    _POOLING_ENUM,
    _desc_from_c,
    _is_local_path,
    _is_gguf_model,
)
from .types import ModelDesc, Stats, TuningResult
from .exceptions import ModelNotFoundError
from .autotune import autotune, _sample_corpus
from .backend import detect_backend, _BACKEND_ENUM

_MODE_TO_MODEL = {
    "fast": "paraphrase-ml-minilm-l12-v2-q",
    "balanced": "BAAI/bge-small-en-v1.5",
    "quality": "BAAI/bge-base-en-v1.5",
}


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
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        self._ctx = None
        self._dim = 0
        self._batch_size = batch_size

        if _is_gguf_model(model_name):
            # GGUF model: use llama.cpp backend
            if "/" in model_name:
                parts = model_name.split("/", 1)
                repo = parts[0]
                filename = parts[1]
            else:
                # Try local path
                import os
                if os.path.isfile(model_name):
                    repo = ""
                    filename = model_name
                else:
                    raise FileNotFoundError(f"GGUF model not found: '{model_name}'")

            opts = ffi.new("lembed_text_options_t *")
            opts.provider = _BACKEND_ENUM.get(provider, 0)
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
                check_status(lib.lembed_text_embedding_create_from_gguf_path(
                    filename.encode("utf-8"), opts, ctx_ptr))
                self._ctx = ctx_ptr[0]
            else:
                # Download from HuggingFace
                ctx_ptr = ffi.new("lembed_text_embedding_t **")
                check_status(lib.lembed_text_embedding_create_from_gguf_model(
                    repo.encode("utf-8"), filename.encode("utf-8"),
                    opts, ctx_ptr))
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
                    raise ModelNotFoundError(f"Unknown model: {model_name}")
                info = ffi.new("lembed_model_info_t *")
                lib.lembed_get_text_model_info(idx, info)
                code = ffi.string(info.model_code).decode("utf-8")
                file_name = ffi.string(info.model_file).decode("utf-8")

                from .autotune import clear_autotune_cache
                model_dir = ffi.new("char **")
                check_status(lib.lembed_ensure_text_model(
                    idx,
                    cache_dir.encode("utf-8") if cache_dir else ffi.NULL,
                    1 if show_download_progress else 0,
                    1 if offline else 0,
                    model_dir,
                ))
                model_dir_ptr = model_dir[0]
                model_dir = ffi.string(model_dir_ptr).decode("utf-8")
                lib.lembed_free_string(model_dir_ptr)

            opts = ffi.new("lembed_text_options_t *")
            opts.provider = _BACKEND_ENUM.get(provider, 0)
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
            check_status(lib.lembed_text_embedding_create(
                opts, ctx_ptr))
            self._ctx = ctx_ptr[0]

        self._dim = lib.lembed_text_embedding_dim(self._ctx)
        self._model_name = model_name

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

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
        desc_ptr = lib.lembed_text_embedding_desc(self._ctx)
        return _desc_from_c(desc_ptr)

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
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(_MODE_TO_MODEL.keys())}")
        return cls(_MODE_TO_MODEL[mode], **kwargs)

    def embed(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed texts into dense vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size override (None = use default).

        Returns:
            numpy array of shape (len(texts), dim).
        """
        n = len(texts)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        c_texts = ffi.new("char*[]", n)
        c_strs = []
        for i, t in enumerate(texts):
            c_strs.append(ffi.new("char[]", t.encode("utf-8")))
            c_texts[i] = c_strs[i]

        result = ffi.new("lembed_embeddings_t *")
        bs = 0 if batch_size is None else batch_size
        check_status(lib.lembed_text_embedding_embed(
            self._ctx, c_texts, n, bs, result))

        dim = result.dim
        total = result.num_embeddings * dim
        arr = np.frombuffer(ffi.buffer(result.data, total * 4), dtype=np.float32).copy()
        lib.lembed_embeddings_free(result)
        return arr.reshape(n, dim)

    def embed_stream(self, texts: list[str], callback, batch_size: int | None = None) -> None:
        """Embed texts as a stream.

        Args:
            texts: List of strings to embed.
            callback: Called for each embedding with (array, dim, userdata).
            batch_size: Batch size override.
        """
        n = len(texts)
        if n == 0:
            return

        c_texts = ffi.new("char*[]", n)
        c_strs = []
        for i, t in enumerate(texts):
            c_strs.append(ffi.new("char[]", t.encode("utf-8")))
            c_texts[i] = c_strs[i]

        bs = 0 if batch_size is None else batch_size

        @ffi.callback("void(const float*, int, void*)")
        def cb(data, dim, userdata):
            arr = np.frombuffer(ffi.buffer(data, dim * 4), dtype=np.float32).copy()
            callback(arr, dim, userdata)

        lib.lembed_text_embedding_embed_stream(
            self._ctx, c_texts, n, bs, cb, ffi.NULL)

    def stats(self) -> Stats:
        """Get runtime statistics."""
        result = ffi.new("lembed_stats_t *")
        lib.lembed_text_embedding_stats(self._ctx, result)
        return Stats(
            texts_embedded=result.texts_embedded,
            batches_run=result.batches_run,
            avg_latency_ms=result.avg_latency_ms,
        )

    def embed_batched(self, texts: list[str], batch_size: int | None = None):
        """Embed texts in batches, yielding numpy arrays."""
        n = len(texts)
        if n == 0:
            return

        bs = 0 if batch_size is None else batch_size
        actual_bs = bs if bs > 0 else self._batch_size
        if actual_bs <= 0:
            actual_bs = 32

        for i in range(0, n, actual_bs):
            batch = texts[i:i + actual_bs]
            embeddings = self.embed(batch, batch_size=actual_bs)
            for emb in embeddings:
                yield emb

    def close(self) -> None:
        """Release the underlying C resources."""
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"TextEmbedding(dim={self._dim})"


# Re-export TextEmbeddingPool from pool module
from .pool import TextEmbeddingPool  # noqa: E402, F401

