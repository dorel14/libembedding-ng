"""High-level sparse text embedding API.

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
    list_sparse_models,
    resolve_sparse_model,
)
from .types import ModelDesc, SparseEmbedding, SparseTuningResult, Stats


class SparseTextEmbedding:
    """Generate sparse vector embeddings (term weights) from text.

    Args:
        model_name: HuggingFace model name or local directory path.
        provider: Execution provider.
        cache_dir: Model cache directory.
        max_length: Max token length (0 = model default).
        threads: Number of threads (0 = auto).
        batch_size: Internal batch size (default 256).
        offline: If True, use cached models only (default False).
        show_download_progress: Show download progress bar.
        top_terms: Max number of terms to keep per document (0 = all).
        min_weight: Minimum weight threshold for pruning (0.0 = no pruning).
        num_threads: Deprecated; use ``threads``.
    """

    def __init__(
        self,
        model_name: str = "prithvida/SPLADE_PP_en_v1",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        max_length: int = 0,
        threads: int = 0,
        batch_size: int = 256,
        offline: bool = False,
        show_download_progress: bool = True,
        top_terms: int = 0,
        min_weight: float = 0.0,
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        opts = lib.lembed_sparse_options_default()
        opts.provider = _PROVIDER_MAP[provider.lower()]
        opts.device_id = device_id
        self._cache_dir_buf = (
            ffi.new("char[]", cache_dir.encode("utf-8")) if cache_dir else ffi.NULL
        )
        opts.cache_dir = self._cache_dir_buf
        opts.max_length = max_length
        opts.num_threads = threads
        opts.batch_size = batch_size
        opts.offline = int(offline)
        opts.show_download_progress = int(show_download_progress)
        opts.top_k = top_terms
        opts.min_weight = min_weight

        ctx_ptr = ffi.new("lembed_sparse_embedding_ctx_t **")

        try:
            model_idx = resolve_sparse_model(model_name)
            opts.model = model_idx
            check_status(lib.lembed_sparse_text_embedding_create(
                ffi.addressof(opts), ctx_ptr))
        except ModelNotFoundError:
            if _is_local_path(model_name):
                check_status(lib.lembed_sparse_text_embedding_create_from_path(
                    model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr))
            else:
                raise

        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_sparse_text_embedding_free)
        self._batch_size = batch_size
        self._sparse_opts = opts

    @staticmethod
    def list_supported_models():
        """Return a list of all supported sparse embedding models."""
        return list_sparse_models()

    @property
    def batch_size(self) -> int:
        """Configured internal batch size."""
        return self._batch_size

    def embed(self, texts: list[str], *, batch_size: int | None = None) -> list[SparseEmbedding]:
        """Embed texts into sparse vectors.

        Returns:
            List of SparseEmbedding with indices (int32) and values (float32).
        """
        n = len(texts)
        if n == 0:
            return []

        bs = 0 if batch_size is None else batch_size

        encoded = [t.encode("utf-8") for t in texts]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_texts = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_sparse_embeddings_t *")
        check_status(
            lib.lembed_sparse_text_embedding_embed(self._ctx, c_texts, n, bs, ffi.NULL, result)
        )

        try:
            embeddings = []
            for i in range(result.count):
                item = result.items[i]
                indices = np.frombuffer(
                    ffi.buffer(item.indices, item.length * 4), dtype=np.int32
                ).copy()
                values = np.frombuffer(
                    ffi.buffer(item.values, item.length * 4), dtype=np.float32
                ).copy()
                embeddings.append(SparseEmbedding(indices=indices, values=values))
            return embeddings
        finally:
            lib.lembed_sparse_embeddings_free(result)

    def info(self) -> ModelDesc:
        """Return runtime model descriptor."""
        desc_ptr = lib.lembed_sparse_text_embedding_desc(self._ctx)
        return _desc_from_c(desc_ptr)

    @property
    def name(self) -> str:
        """Model name or local path."""
        name_ptr = lib.lembed_sparse_text_embedding_model_name(self._ctx)
        return ffi.string(name_ptr).decode("utf-8", errors="replace") if name_ptr else ""

    def stats(self) -> Stats:
        """Return runtime usage statistics."""
        s = ffi.new("lembed_stats_t *")
        lib.lembed_sparse_text_embedding_stats(self._ctx, s)
        return Stats(
            texts_embedded=s.texts_embedded,
            batches_run=s.batches_run,
            avg_latency_ms=s.avg_latency_ms,
        )

    def close(self) -> None:
        self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def sparse_autotune(
    model_name: str = "prithivida/Splade_PP_en_v1",
    *,
    full: bool = False,
) -> SparseTuningResult:
    """Run auto-tuning to find optimal sparse embedding configuration.

    Args:
        model_name: Model name (e.g. "prithivida/Splade_PP_en_v1")
        full: If True, run FULL mode (30-120s), else QUICK (5-15s)

    Returns:
        SparseTuningResult with optimal top_k, min_weight, storage_format.

    Example:
        >>> result = sparse_autotune("prithivida/Splade_PP_en_v1")
        >>> print(f"Optimal: top_k={result.top_k}, storage={result.storage_format}")
    """
    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    result = ffi.new("lembed_sparse_tuning_result_t *")

    # Resolve model code
    code = model_name
    models = list_sparse_models()
    for m in models:
        if model_name in (m.model_name, m.model_code):
            code = m.model_code
            break

    check_status(lib.lembed_sparse_autotune(code.encode("utf-8"), mode, result))

    return SparseTuningResult(
        top_k=result.top_k,
        min_weight=result.min_weight,
        storage_format=result.storage_format,
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
    )

