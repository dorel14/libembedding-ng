"""High-level image embedding API.

Auteur: David Orel
Version: 1.4.0
"""

from __future__ import annotations

import warnings

import numpy as np

from ._binding import ffi, lib
from ._status import check_status
from .exceptions import ModelNotFoundError
from .models import (
    _PROVIDER_MAP,
    _desc_from_c,
    _is_local_path,
    list_image_models,
    resolve_image_model,
)
from .types import ImageTuningResult, ModelDesc, Stats


class ImageEmbedding:
    """Generate dense vector embeddings from images.

    Args:
        model_name: Model name (e.g. "openai/clip-vit-base-patch32") or local
            directory path.
        provider: Execution provider.
        cache_dir: Model cache directory.
        threads: Number of threads (0 = auto).
        batch_size: Internal batch size (default 32).
        offline: If True, use cached models only (default False).
        show_download_progress: Show download progress bar.
        dim: Output dimension for local models without config.json (0 = auto).
        num_threads: Deprecated; use ``threads``.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/clip-ViT-B-32-vision",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        threads: int = 0,
        batch_size: int = 30,
        offline: bool = False,
        show_download_progress: bool = True,
        dim: int = 0,
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        opts = lib.lembed_image_options_default()
        opts.provider = _PROVIDER_MAP[provider.lower()]
        opts.device_id = device_id
        self._cache_dir_buf = (
            ffi.new("char[]", cache_dir.encode("utf-8")) if cache_dir else ffi.NULL
        )
        opts.cache_dir = self._cache_dir_buf
        opts.num_threads = threads
        opts.batch_size = batch_size
        opts.offline = int(offline)
        opts.show_download_progress = int(show_download_progress)
        opts.dim = dim

        ctx_ptr = ffi.new("lembed_image_embedding_t **")

        try:
            model_idx = resolve_image_model(model_name)
            opts.model = model_idx
            check_status(lib.lembed_image_embedding_create(
                ffi.addressof(opts), ctx_ptr))
        except ModelNotFoundError:
            if _is_local_path(model_name):
                check_status(lib.lembed_image_embedding_create_from_path(
                    model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr))
            else:
                raise

        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_image_embedding_free)
        self._dim = lib.lembed_image_embedding_dim(self._ctx)
        self._batch_size = batch_size

    @staticmethod
    def list_supported_models():
        """Return a list of all supported image embedding models."""
        return list_image_models()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        """Configured internal batch size."""
        return self._batch_size

    def info(self) -> ModelDesc:
        """Return runtime model descriptor."""
        desc_ptr = lib.lembed_image_embedding_desc(self._ctx)
        return _desc_from_c(desc_ptr)

    @property
    def name(self) -> str:
        """Model name or local path."""
        name_ptr = lib.lembed_image_embedding_model_name(self._ctx)
        return ffi.string(name_ptr).decode("utf-8", errors="replace") if name_ptr else ""

    def embed_files(self, paths: list[str], *, batch_size: int | None = None) -> np.ndarray:
        """Embed images from file paths.

        Returns:
            numpy array of shape (len(paths), dim) with dtype float32.
        """
        n = len(paths)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        bs = 0 if batch_size is None else batch_size

        encoded = [p.encode("utf-8") for p in paths]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_paths = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_embeddings_t *")
        check_status(
            lib.lembed_image_embedding_embed_files(self._ctx, c_paths, n, bs, result)
        )

        try:
            buf = ffi.buffer(result.data, result.num_embeddings * result.dim * 4)
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            return arr.reshape(result.num_embeddings, result.dim)
        finally:
            lib.lembed_embeddings_free(result)

    def embed_bytes(self, images: list[bytes], *, batch_size: int | None = None) -> np.ndarray:
        """Embed images from raw bytes (JPEG, PNG, etc.).

        Returns:
            numpy array of shape (len(images), dim) with dtype float32.
        """
        n = len(images)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        bs = 0 if batch_size is None else batch_size

        c_data = ffi.new("unsigned char*[]", n)
        c_sizes = ffi.new("int[]", n)
        kept = []
        for i, img in enumerate(images):
            buf = ffi.new("unsigned char[]", img)
            kept.append(buf)
            c_data[i] = buf
            c_sizes[i] = len(img)

        result = ffi.new("lembed_embeddings_t *")
        check_status(
            lib.lembed_image_embedding_embed_bytes(
                self._ctx, c_data, c_sizes, n, bs, result
            )
        )

        try:
            buf = ffi.buffer(result.data, result.num_embeddings * result.dim * 4)
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            return arr.reshape(result.num_embeddings, result.dim)
        finally:
            lib.lembed_embeddings_free(result)

    def stats(self) -> Stats:
        """Return runtime usage statistics."""
        s = ffi.new("lembed_stats_t *")
        lib.lembed_image_embedding_stats(self._ctx, s)
        return Stats(
            texts_embedded=s.texts_embedded,
            batches_run=s.batches_run,
            avg_latency_ms=s.avg_latency_ms,
        )

    def close(self) -> None:
        """Release the underlying C resources."""
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def image_autotune(
    model_name: str = "openai/clip-vit-base-patch32-quantized",
    *,
    full: bool = False,
) -> ImageTuningResult:
    """Run auto-tuning to find optimal image embedding configuration.

    Args:
        model_name: Model name (e.g. "openai/clip-vit-base-patch32-quantized")
        full: If True, run FULL mode (30-120s), else QUICK (5-15s)

    Returns:
        ImageTuningResult with optimal threads, batch_size.

    Example:
        >>> result = image_autotune("openai/clip-vit-base-patch32-quantized")
        >>> print(f"Optimal: {result.threads} threads, batch={result.batch_size}")
    """
    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    result = ffi.new("lembed_image_tuning_result_t *")

    # Resolve model code
    code = model_name
    models = list_image_models()
    for m in models:
        if model_name in (m.model_name, m.model_code):
            code = m.model_code
            break

    check_status(lib.lembed_image_autotune(code.encode("utf-8"), mode, result))

    return ImageTuningResult(
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
    )

