"""High-level sparse text embedding API.

Auteur: David Orel
Version: 1.11.0
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

_STORAGE_FORMAT_MAP = {
    "dict": lib.LEMBED_SPARSE_FORMAT_DICT,
    "index_order": lib.LEMBED_SPARSE_FORMAT_INDEX_ORDER,
}


def _storage_format(value: str) -> int:
    """Map a storage_format name to its C constant, or raise."""
    fmt = _STORAGE_FORMAT_MAP.get(value.lower())
    if fmt is None:
        raise ValueError(
            f"Unknown storage_format '{value}'. Use: {list(_STORAGE_FORMAT_MAP.keys())}"
        )
    return fmt


def _build_options(
    *,
    provider: str,
    device_id: int,
    cache_dir,
    max_length: int,
    threads: int,
    batch_size: int,
    offline: bool,
    show_download_progress: bool,
    top_terms: int,
    min_weight: float,
    storage_format: str,
):
    """Fill a C options struct, returning it together with the cache_dir buffer.

    The buffer has to outlive the call that stores the pointer into the struct,
    so it is returned and kept on the instance. Shared by every constructor of
    SparseTextEmbedding so the three entry points cannot drift apart.
    """
    opts = lib.lembed_sparse_options_default()
    opts.provider = _PROVIDER_MAP[provider.lower()]
    opts.device_id = device_id
    cache_buf = ffi.new("char[]", cache_dir.encode("utf-8")) if cache_dir else ffi.NULL
    opts.cache_dir = cache_buf
    opts.max_length = max_length
    opts.num_threads = threads
    opts.batch_size = batch_size
    opts.offline = int(offline)
    opts.show_download_progress = int(show_download_progress)
    opts.top_k = top_terms
    opts.min_weight = min_weight
    opts.storage_format = _storage_format(storage_format)
    return opts, cache_buf


class SparseTextEmbedding:
    """Generate sparse vector embeddings (term weights) from text.

    Two backends behind one API: ONNX Runtime from the model registry, and this
    library's own ggml runtime from a GGUF file. See :meth:`from_gguf` for the
    latter -- a local path ending in ``.gguf`` is routed to it automatically.

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
        storage_format: Output format - "dict" (default, sorted by weight desc),
            "index_order" (sorted by index asc).
        num_threads: Deprecated; use ``threads``.
    """

    def __init__(
        self,
        model_name: str = "prithivida/Splade_PP_en_v1",
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
        storage_format: str = "dict",
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        opts, self._cache_dir_buf = _build_options(
            provider=provider,
            device_id=device_id,
            cache_dir=cache_dir,
            max_length=max_length,
            threads=threads,
            batch_size=batch_size,
            offline=offline,
            show_download_progress=show_download_progress,
            top_terms=top_terms,
            min_weight=min_weight,
            storage_format=storage_format,
        )

        ctx_ptr = ffi.new("lembed_sparse_embedding_ctx_t **")

        try:
            model_idx = resolve_sparse_model(model_name)
            opts.model = model_idx
            check_status(
                lib.lembed_sparse_text_embedding_create(ffi.addressof(opts), ctx_ptr)
            )
        except ModelNotFoundError:
            if _is_local_path(model_name):
                # A .gguf path is routed to the GGUF runtime by the C layer.
                check_status(
                    lib.lembed_sparse_text_embedding_create_from_path(
                        model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr
                    )
                )
            else:
                raise

        self._finish(ctx_ptr, batch_size, opts)

    @classmethod
    def from_gguf(
        cls,
        path: str,
        *,
        max_length: int = 0,
        threads: int = 0,
        batch_size: int = 256,
        top_terms: int = 0,
        min_weight: float = 0.0,
        storage_format: str = "dict",
        **kwargs,
    ) -> SparseTextEmbedding:
        """Load a SPLADE model from a local ``.gguf`` file.

        Runs on this library's own ggml graph, not on ONNX Runtime and not on
        llama.cpp. The file must carry an MLM/SPLADE head over a declared
        vocabulary; a dense GGUF is refused with a reason rather than accepted.

        Args:
            path: Path to the GGUF file.
            max_length: Max token length (0 = the model's own context length).
            threads: Number of threads (0 = auto).
            batch_size: Internal batch size (default 256).
            top_terms: Max number of terms to keep per document (0 = all).
            min_weight: Minimum weight threshold for pruning (0.0 = no pruning).
            storage_format: "dict" or "index_order".
            **kwargs: ``provider``, ``device_id``, ``cache_dir``, ``offline``,
                ``show_download_progress`` -- accepted for signature symmetry.

        Returns:
            A SparseTextEmbedding backed by the GGUF runtime.
        """
        opts, cache_buf = _build_options(
            provider=kwargs.pop("provider", "cpu"),
            device_id=kwargs.pop("device_id", 0),
            cache_dir=kwargs.pop("cache_dir", None),
            max_length=max_length,
            threads=threads,
            batch_size=batch_size,
            offline=kwargs.pop("offline", False),
            show_download_progress=kwargs.pop("show_download_progress", True),
            top_terms=top_terms,
            min_weight=min_weight,
            storage_format=storage_format,
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        ctx_ptr = ffi.new("lembed_sparse_embedding_ctx_t **")
        check_status(
            lib.lembed_sparse_text_embedding_create_from_gguf_path(
                path.encode("utf-8"), ffi.addressof(opts), ctx_ptr
            )
        )
        obj = cls.__new__(cls)
        obj._cache_dir_buf = cache_buf
        obj._finish(ctx_ptr, batch_size, opts)
        return obj

    @classmethod
    def from_gguf_model(
        cls,
        repo: str,
        filename: str,
        *,
        max_length: int = 0,
        threads: int = 0,
        batch_size: int = 256,
        top_terms: int = 0,
        min_weight: float = 0.0,
        storage_format: str = "dict",
        **kwargs,
    ) -> SparseTextEmbedding:
        """Download a GGUF sparse model from HuggingFace and load it.

        Args:
            repo: HuggingFace repository, e.g. "cstr/splade-pp-en-v1-GGUF".
            filename: File name inside the repository, e.g.
                "splade-pp-en-v1-q8_0.gguf".
            max_length: Max token length (0 = the model's own context length).
            threads: Number of threads (0 = auto).
            batch_size: Internal batch size (default 256).
            top_terms: Max number of terms to keep per document (0 = all).
            min_weight: Minimum weight threshold for pruning (0.0 = no pruning).
            storage_format: "dict" or "index_order".
            **kwargs: ``provider``, ``device_id``, ``cache_dir``, ``offline``,
                ``show_download_progress``.

        Returns:
            A SparseTextEmbedding backed by the GGUF runtime.
        """
        opts, cache_buf = _build_options(
            provider=kwargs.pop("provider", "cpu"),
            device_id=kwargs.pop("device_id", 0),
            cache_dir=kwargs.pop("cache_dir", None),
            max_length=max_length,
            threads=threads,
            batch_size=batch_size,
            offline=kwargs.pop("offline", False),
            show_download_progress=kwargs.pop("show_download_progress", True),
            top_terms=top_terms,
            min_weight=min_weight,
            storage_format=storage_format,
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        ctx_ptr = ffi.new("lembed_sparse_embedding_ctx_t **")
        check_status(
            lib.lembed_sparse_text_embedding_create_from_gguf_model(
                repo.encode("utf-8"),
                filename.encode("utf-8"),
                ffi.addressof(opts),
                ctx_ptr,
            )
        )
        obj = cls.__new__(cls)
        obj._cache_dir_buf = cache_buf
        obj._finish(ctx_ptr, batch_size, opts)
        return obj

    def _finish(self, ctx_ptr, batch_size: int, opts) -> None:
        """Adopt a freshly created context. Shared by the three constructors."""
        self._ctx = ctx_ptr[0]
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

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        top_terms: int | None = None,
        min_weight: float | None = None,
        storage_format: str | None = None,
    ) -> list[SparseEmbedding]:
        """Embed texts into sparse vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size override (None = use constructor default).
            top_terms: Override max terms per document (None = use constructor value).
            min_weight: Override minimum weight threshold (None = use constructor value).
            storage_format: Override output format - "dict" or "index_order"
                (None = use constructor value).

        Returns:
            List of SparseEmbedding with indices (int32) and values (float32).
        """
        n = len(texts)
        if n == 0:
            return []

        bs = self._batch_size if batch_size is None else batch_size

        encoded = [t.encode("utf-8") for t in texts]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_texts = ffi.new("char*[]", c_strs)

        # Prepare sparse options for per-call overrides
        sparse_opts = ffi.NULL
        sparse_opts_ptr = ffi.NULL
        if top_terms is not None or min_weight is not None or storage_format is not None:
            sparse_opts_ptr = ffi.new("lembed_sparse_options_t *")
            sparse_opts = sparse_opts_ptr[0]
            # Copy current context options as base
            sparse_opts.top_k = self._sparse_opts.top_k
            sparse_opts.min_weight = self._sparse_opts.min_weight
            sparse_opts.storage_format = self._sparse_opts.storage_format
            # Apply overrides
            if top_terms is not None:
                sparse_opts.top_k = top_terms
            if min_weight is not None:
                sparse_opts.min_weight = min_weight
            if storage_format is not None:
                sparse_opts.storage_format = _storage_format(storage_format)

        result = ffi.new("lembed_sparse_embeddings_t *")
        check_status(
            lib.lembed_sparse_text_embedding_embed(
                self._ctx, c_texts, n, bs, sparse_opts_ptr, result
            )
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
        return (
            ffi.string(name_ptr).decode("utf-8", errors="replace") if name_ptr else ""
        )

    def stats(self) -> Stats:
        """Return runtime usage statistics."""
        s = ffi.new("lembed_stats_v2_t *")
        lib.lembed_sparse_text_embedding_stats_v2(self._ctx, s)
        return Stats(
            texts_embedded=s.base.texts_embedded,
            batches_run=s.base.batches_run,
            avg_latency_ms=s.base.avg_latency_ms,
            cache_hits=s.cache_hits,
            cache_misses=s.cache_misses,
        )

    def close(self) -> None:
        if self._ctx is not None:
            lib.lembed_sparse_text_embedding_free(self._ctx)
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


def sparse_best_config(
    model_name: str = "prithivida/Splade_PP_en_v1",
    texts: list[str] | None = None,
) -> SparseTuningResult:
    """Find optimal sparse configuration by benchmarking variants.

    Args:
        model_name: Model name or local path.
        texts: Optional list of sample texts for benchmarking.

    Returns:
        SparseTuningResult with optimal top_k, min_weight, storage_format.

    Example:
        >>> result = sparse_best_config("prithivida/Splade_PP_en_v1")
        >>> print(f"Optimal: top_k={result.top_k}, storage={result.storage_format}")
    """
    if texts is None:
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "The quick brown fox jumps over the lazy dog",
            "Embeddings are dense vector representations of text",
            "Natural language processing understanding text semantics",
            "Deep learning models learn hierarchical representations",
        ]

    n = len(texts)
    encoded = [t.encode("utf-8") for t in texts]
    c_strs = [ffi.new("char[]", e) for e in encoded]
    c_texts = ffi.new("char*[]", c_strs)

    result = ffi.new("lembed_sparse_tuning_result_t *")
    check_status(
        lib.lembed_sparse_best_config(
            model_name.encode("utf-8"), c_texts, n, result
        )
    )

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
