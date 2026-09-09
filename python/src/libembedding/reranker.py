"""High-level document reranker API.

Auteur: David Orel
Version: 1.4.0
"""

from __future__ import annotations

import warnings

from ._binding import ffi, lib
from ._status import check_status
from .backend import backend_to_enum, _BACKEND_ONNX
from .exceptions import ModelNotFoundError
from .models import (
    _PROVIDER_MAP,
    _desc_from_c,
    _is_local_path,
    list_reranker_models,
    resolve_reranker_model,
)
from .types import ModelDesc, RerankerTuningResult, RerankResult, Stats


class Reranker:
    """Score and sort documents by relevance to a query.

    Args:
        model_name: Model name (e.g. "BAAI/bge-reranker-base") or local
            directory path.
        provider: Execution provider.
        cache_dir: Model cache directory.
        max_length: Max token length (0 = model default).
        threads: Number of threads (0 = auto).
        batch_size: Internal batch size (default 256).
        offline: If True, use cached models only (default False).
        show_download_progress: Show download progress bar.
        backend: Backend to use ("auto", "onnx", or "llama").
        num_threads: Deprecated; use ``threads``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        provider: str = "cpu",
        device_id: int = 0,
        cache_dir: str | None = None,
        max_length: int = 0,
        threads: int = 0,
        batch_size: int = 256,
        offline: bool = False,
        show_download_progress: bool = True,
        backend: str = "auto",
        num_threads: int | None = None,
    ):
        if num_threads is not None:
            warnings.warn(
                "num_threads is deprecated, use threads",
                DeprecationWarning,
                stacklevel=2,
            )
            threads = num_threads

        opts = lib.lembed_reranker_options_default()
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
        opts.backend = backend_to_enum(backend)

        ctx_ptr = ffi.new("lembed_reranker_t **")

        backend_enum = opts.backend

        try:
            model_idx = resolve_reranker_model(model_name)
            opts.model = model_idx
            check_status(lib.lembed_reranker_create(ffi.addressof(opts), ctx_ptr))
        except ModelNotFoundError:
            if backend_enum == _BACKEND_ONNX:
                raise
            if _is_local_path(model_name):
                if model_name.lower().endswith(".gguf"):
                    check_status(lib.lembed_reranker_create_from_gguf_path(
                        model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr))
                else:
                    check_status(lib.lembed_reranker_create_from_path(
                        model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr))
            else:
                check_status(lib.lembed_reranker_create_from_gguf_path(
                    model_name.encode("utf-8"), ffi.addressof(opts), ctx_ptr))

        self._ctx = ffi.gc(ctx_ptr[0], lib.lembed_reranker_free)
        self._batch_size = batch_size

    @staticmethod
    def list_supported_models():
        """Return a list of all supported reranker models."""
        return list_reranker_models()

    @staticmethod
    def auto(profile: str = "balanced", **kwargs) -> Reranker:
        """Create a Reranker with automatic configuration based on profile.

        This is the recommended way to create a Reranker Ã¢â‚¬â€ it automatically
        selects the optimal model and configuration for your hardware.

        Args:
            profile: "fast" (INT8, minimal latency), "balanced" (~300ms), or "quality" (FP32, best ranking)
            **kwargs: Additional arguments passed to Reranker constructor

        Returns:
            Configured Reranker instance.

        Example:
            >>> reranker = Reranker.auto("fast")  # INT8, minimal latency
            >>> reranker = Reranker.auto("balanced")  # Good quality/speed
            >>> reranker = Reranker.auto("quality")  # FP32, best ranking
        """
        return Reranker_auto(profile, **kwargs)

    def rerank(
        self, query: str, documents: list[str], *, batch_size: int | None = None
    ) -> list[RerankResult]:
        """Score and sort documents by relevance.

        Args:
            query: The query text.
            documents: List of document texts.
            batch_size: Internal batch size (None = use constructor default).

        Returns:
            List of RerankResult sorted by score (descending).
        """
        n = len(documents)
        if n == 0:
            return []

        bs = 0 if batch_size is None else batch_size

        c_query = ffi.new("char[]", query.encode("utf-8"))
        encoded = [d.encode("utf-8") for d in documents]
        c_strs = [ffi.new("char[]", e) for e in encoded]
        c_docs = ffi.new("char*[]", c_strs)

        result = ffi.new("lembed_rerank_results_t *")
        check_status(
            lib.lembed_reranker_rerank(self._ctx, c_query, c_docs, n, bs, result)
        )

        try:
            return [
                RerankResult(index=result.items[i].index, score=result.items[i].score)
                for i in range(result.count)
            ]
        finally:
            lib.lembed_rerank_results_free(result)

    def info(self) -> ModelDesc:
        """Return runtime model descriptor."""
        desc_ptr = lib.lembed_reranker_desc(self._ctx)
        return _desc_from_c(desc_ptr)

    @property
    def name(self) -> str:
        """Model name or local path."""
        name_ptr = lib.lembed_reranker_model_name(self._ctx)
        return ffi.string(name_ptr).decode("utf-8", errors="replace") if name_ptr else ""

    def stats(self) -> Stats:
        """Return runtime usage statistics."""
        s = ffi.new("lembed_stats_t *")
        lib.lembed_reranker_stats(self._ctx, s)
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


def reranker_autotune(
    model_name: str = "jinaai/jina-reranker-v1-turbo-en-quantized",
    *,
    full: bool = False,
    objective: str = "balanced",
) -> RerankerTuningResult:
    """Run auto-tuning to find optimal reranker configuration.

    Args:
        model_name: Model name (e.g. "jinaai/jina-reranker-v1-turbo-en-quantized")
        full: If True, run FULL mode (30-120s), else QUICK (5-15s)
        objective: "latency", "throughput", "balanced", or "memory"

    Returns:
        RerankerTuningResult with optimal threads, batch_size, max_tokens.

    Example:
        >>> result = reranker_autotune("jinaai/jina-reranker-v1-turbo-en-quantized", objective="latency")
        >>> print(f"Optimal: {result.threads} threads, batch={result.batch_size}, tokens={result.max_tokens}")
    """
    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    obj_map = {
        "latency": lib.LEMBED_OBJECTIVE_LATENCY,
        "throughput": lib.LEMBED_OBJECTIVE_THROUGHPUT,
        "balanced": lib.LEMBED_OBJECTIVE_BALANCED,
        "memory": lib.LEMBED_OBJECTIVE_MEMORY,
    }
    if objective not in obj_map:
        raise ValueError(f"Unknown objective '{objective}'. Use: {list(obj_map.keys())}")

    result = ffi.new("lembed_reranker_tuning_result_t *")

    # Resolve model name to code
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(lib.lembed_list_reranker_models(models_ptr, count_ptr))
    code = model_name
    for i in range(count_ptr[0]):
        name = ffi.string(models_ptr[0][i].model_name).decode()
        model_code = ffi.string(models_ptr[0][i].model_code).decode()
        if model_name in (name, model_code):
            code = model_code
            break

    check_status(lib.lembed_reranker_autotune(code.encode("utf-8"), mode, obj_map[objective], result))

    return RerankerTuningResult(
        threads=result.threads,
        batch_size=result.batch_size,
        max_tokens=result.max_tokens,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        memory_mb=result.memory_mb,
    )


def reranker_autotune_constrained(
    model_name: str = "jinaai/jina-reranker-v1-turbo-en-quantized",
    *,
    full: bool = False,
    objective: str = "balanced",
    min_tokens: int = 64,
    max_latency_ms: float = 500.0,
) -> RerankerTuningResult:
    """Run auto-tuning with quality constraints.

    Args:
        model_name: Model name
        full: If True, run FULL mode (30-120s), else QUICK (5-15s)
        objective: "latency", "throughput", "balanced", or "memory"
        min_tokens: Minimum acceptable max_tokens (quality constraint)
        max_latency_ms: Maximum acceptable latency in ms

    Returns:
        RerankerTuningResult that satisfies constraints.

    Example:
        >>> result = reranker_autotune_constrained(min_tokens=128, max_latency_ms=200)
        >>> print(f"Config: {result.threads} threads, {result.max_tokens} tokens")
    """
    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    obj_map = {
        "latency": lib.LEMBED_OBJECTIVE_LATENCY,
        "throughput": lib.LEMBED_OBJECTIVE_THROUGHPUT,
        "balanced": lib.LEMBED_OBJECTIVE_BALANCED,
        "memory": lib.LEMBED_OBJECTIVE_MEMORY,
    }
    if objective not in obj_map:
        raise ValueError(f"Unknown objective '{objective}'. Use: {list(obj_map.keys())}")

    result = ffi.new("lembed_reranker_tuning_result_t *")

    # Resolve model name to code
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(lib.lembed_list_reranker_models(models_ptr, count_ptr))
    code = model_name
    for i in range(count_ptr[0]):
        name = ffi.string(models_ptr[0][i].model_name).decode()
        model_code = ffi.string(models_ptr[0][i].model_code).decode()
        if model_name in (name, model_code):
            code = model_code
            break

    check_status(lib.lembed_reranker_autotune_constrained(
        code.encode("utf-8"), mode, obj_map[objective], min_tokens, max_latency_ms, result))

    return RerankerTuningResult(
        threads=result.threads,
        batch_size=result.batch_size,
        max_tokens=result.max_tokens,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        memory_mb=result.memory_mb,
    )


def reranker_auto_config(
    model_name: str = "jinaai/jina-reranker-v1-turbo-en-quantized",
    target_latency_ms: float = 500.0,
) -> RerankerTuningResult:
    """Auto-configure reranker to fit within a latency budget.

    Args:
        model_name: Model name
        target_latency_ms: Maximum acceptable latency in ms (e.g. 500)

    Returns:
        RerankerTuningResult that fits within the budget.

    Example:
        >>> result = reranker_auto_config(target_latency_ms=300)
        >>> print(f"Config: {result.threads} threads, {result.max_tokens} tokens")
    """
    result = ffi.new("lembed_reranker_tuning_result_t *")

    # Resolve model code
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(lib.lembed_list_reranker_models(models_ptr, count_ptr))
    code = model_name
    for i in range(count_ptr[0]):
        name = ffi.string(models_ptr[0][i].model_name).decode()
        model_code = ffi.string(models_ptr[0][i].model_code).decode()
        if model_name in (name, model_code):
            code = model_code
            break

    check_status(lib.lembed_reranker_auto_config(code.encode("utf-8"), target_latency_ms, result))

    return RerankerTuningResult(
        threads=result.threads,
        batch_size=result.batch_size,
        max_tokens=result.max_tokens,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        memory_mb=result.memory_mb,
    )


def clear_reranker_autotune_cache(model_name: str | None = None) -> None:
    """Clear reranker autotune cache for a model (or all models if None)."""
    lib.lembed_reranker_autotune_clear_cache(
        model_name.encode("utf-8") if model_name else ffi.NULL
    )


def reranker_auto_config_profile(
    profile: str = "balanced",
    model_name: str = "jinaai/jina-reranker-v1-turbo-en-quantized",
) -> RerankerTuningResult:
    """Auto-configure reranker using a profile.

    Args:
        profile: "fast" (INT8, minimal latency), "balanced" (~300ms), or "quality" (FP32, best ranking)
        model_name: Model name (ignored for "fast"/"quality" which use optimized defaults)

    Returns:
        RerankerTuningResult for the profile.

    Example:
        >>> result = reranker_auto_config_profile("fast")
        >>> print(f"Config: {result.threads} threads, {result.max_tokens} tokens")
    """
    profile_map = {
        "fast": lib.LEMBED_PROFILE_INTERACTIVE,
        "interactive": lib.LEMBED_PROFILE_INTERACTIVE,
        "balanced": lib.LEMBED_PROFILE_BALANCED,
        "quality": lib.LEMBED_PROFILE_QUALITY,
    }
    if profile not in profile_map:
        raise ValueError(f"Unknown profile '{profile}'. Use: {list(profile_map.keys())}")

    result = ffi.new("lembed_reranker_tuning_result_t *")
    check_status(lib.lembed_reranker_auto_config_profile(model_name.encode("utf-8"), profile_map[profile], result))

    return RerankerTuningResult(
        threads=result.threads,
        batch_size=result.batch_size,
        max_tokens=result.max_tokens,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        memory_mb=result.memory_mb,
    )


def Reranker_auto(
    profile: str = "balanced",
    **kwargs,
) -> Reranker:
    """Create a Reranker with automatic configuration based on profile.

    This is the recommended way to create a Reranker Ã¢â‚¬â€ it automatically
    selects the optimal model and configuration for your hardware.

    Args:
        profile: "fast" (INT8, minimal latency), "balanced" (~300ms), or "quality" (FP32, best ranking)
        **kwargs: Additional arguments passed to Reranker constructor

    Returns:
        Configured Reranker instance.

    Example:
        >>> # Fast: INT8, minimal latency
        >>> reranker = Reranker.auto("fast")
        >>>
        >>> # Balanced: good quality/speed tradeoff
        >>> reranker = Reranker.auto("balanced")
        >>>
        >>> # Quality: FP32, best ranking
        >>> reranker = Reranker.auto("quality")
    """
    # Select model based on profile
    model_map = {
        "fast": "jinaai/jina-reranker-v1-turbo-en-quantized",
        "balanced": "jinaai/jina-reranker-v1-turbo-en-quantized",
        "quality": "jinaai/jina-reranker-v1-turbo-en",
    }
    model_name = model_map.get(profile, "jinaai/jina-reranker-v1-turbo-en-quantized")

    # Get optimal config
    config = reranker_auto_config_profile(profile, model_name)

    # Create reranker with optimal config
    return Reranker(
        model_name,
        threads=config.threads,
        batch_size=config.batch_size,
        max_length=config.max_tokens,
        **kwargs,
    )

