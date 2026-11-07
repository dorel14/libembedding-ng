"""Auto-tuning functions for optimal embedding configuration."""

from __future__ import annotations

from ._binding import ffi, lib
from ._status import check_status
from .models import resolve_text_model
from .sampling import _sample_corpus
from .types import ModelSelectionResult, TuningResult, UnifiedTuningResult

# The C tuner benchmarks every configuration against the whole corpus, so an
# unbounded user corpus would make a single call arbitrarily long. Cap it.
_MAX_AUTOTUNE_SAMPLE = 100


def _resolve_model_code(model_name: str) -> str:
    """Map a model name or code to the code the C layer expects."""
    idx = resolve_text_model(model_name)
    if idx < 0:
        return model_name
    info = ffi.new("lembed_model_info_t *")
    lib.lembed_get_text_model_info(idx, info)
    return ffi.string(info.model_code).decode("utf-8")


def _to_c_string_array(texts: list[str]) -> tuple:
    """Build a ``const char* const*`` for ``texts``.

    Returns the array and the list of owning ``char[]`` buffers: cffi may
    collect them as soon as they become unreachable, and the C call reads
    through them, so the caller must keep both alive across the call.
    """
    owners = [ffi.new("char[]", t.encode("utf-8")) for t in texts]
    return ffi.new("char*[]", owners), owners


def _run_autotune(
    model_name: str,
    mode,
    texts: list[str] | None = None,
    max_sample_size: int = _MAX_AUTOTUNE_SAMPLE,
) -> TuningResult:
    """Run the text auto-tuner, on a user corpus or on the synthetic one."""
    code = _resolve_model_code(model_name)
    result = ffi.new("lembed_tuning_result_t *")

    if texts:
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(
                    f"texts must contain str, got {type(text).__name__}"
                )
        sampled = _sample_corpus(list(texts), max_sample_size)
        c_texts, _owners = _to_c_string_array(sampled)
        check_status(
            lib.lembed_autotune_custom(
                code.encode("utf-8"), c_texts, len(sampled), mode, result
            )
        )
    else:
        check_status(lib.lembed_autotune(code.encode("utf-8"), mode, result))

    return TuningResult(
        workers=result.workers,
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
    )


def _do_autotune(
    model_name: str,
    provider: str = "cpu",
    texts: list[str] | None = None,
    max_sample_size: int = _MAX_AUTOTUNE_SAMPLE,
) -> TuningResult:
    """Run autotune with cache lookup."""
    return _run_autotune(
        model_name,
        lib.LEMBED_AUTOTUNE_QUICK,
        texts,
        max_sample_size,
    )


def autotune(
    model_name: str = "BAAI/bge-small-en-v1.5",
    *,
    full: bool = False,
    texts: list[str] | None = None,
    max_sample_size: int = _MAX_AUTOTUNE_SAMPLE,
) -> TuningResult:
    """Run auto-tuning to find optimal configuration for a model.

    Args:
        model_name: HuggingFace model code (e.g. "Qdrant/all-MiniLM-L6-v2-onnx")
        full: If True, run exhaustive tuning (30-120s). Otherwise quick (5-15s).
        texts: Optional custom corpus for benchmarking. If None, a synthetic
            corpus is generated. The corpus is stratified-sampled down to
            ``max_sample_size`` first, because the tuner benchmarks every
            configuration against the whole corpus.
        max_sample_size: Upper bound on the number of texts sent to the tuner.

    Returns:
        TuningResult with optimal workers, threads, batch_size.
    """
    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    return _run_autotune(model_name, mode, texts, max_sample_size)


def auto_select_model(
    use_case: str = "balanced",
) -> ModelSelectionResult:
    """Automatically select the best model and configuration for your hardware.

    Args:
        use_case: One of "speed", "quality", or "balanced" (default).

    Returns:
        ModelSelectionResult with optimal model and configuration.
    """
    result = ffi.new("lembed_model_selection_t *")
    check_status(lib.lembed_auto_select_model(use_case.encode("utf-8"), result))

    return ModelSelectionResult(
        model_code=ffi.string(result.model_code).decode("utf-8"),
        model_name=ffi.string(result.model_name).decode("utf-8"),
        dim=result.dim,
        workers=result.workers,
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
        score=result.score,
    )


def clear_autotune_cache(model_name: str | None = None) -> None:
    """Clear autotune cache for a model (or all models if None)."""
    lib.lembed_autotune_clear_cache(
        model_name.encode("utf-8") if model_name else ffi.NULL
    )


_TASK_EMBEDDING = 0
_TASK_RERANKING = 1
_TASK_IMAGE = 2
_TASK_SPARSE = 3

_TASKS = {
    "embedding": _TASK_EMBEDDING,
    "reranking": _TASK_RERANKING,
    "image": _TASK_IMAGE,
    "sparse": _TASK_SPARSE,
}

_OBJECTIVES = {
    "latency": 0,
    "throughput": 1,
    "balanced": 2,
    "memory": 3,
}

_DEFAULT_MODELS = {
    "embedding": "BAAI/bge-small-en-v1.5",
    "reranking": "jinaai/jina-reranker-v1-turbo-en-quantized",
}


def _check_texts(texts: list[str]) -> None:
    for text in texts:
        if not isinstance(text, str):
            raise TypeError(f"texts must contain str, got {type(text).__name__}")


def _sample_workload(task: str, texts: list[str], max_sample_size: int) -> list[str]:
    """Bound the workload the C tuner benchmarks every configuration on.

    For reranking the last text is the query: only the documents are sampled,
    and the query stays last.
    """
    if task == "reranking" and len(texts) > 1:
        docs = _sample_corpus(list(texts[:-1]), max(1, max_sample_size - 1))
        return [*docs, texts[-1]]
    return _sample_corpus(list(texts), max_sample_size)


def autotune_unified(
    task: str = "embedding",
    model_name: str | None = None,
    *,
    full: bool = False,
    objective: str | None = None,
    texts: list[str] | None = None,
    max_latency_ms: float = 0.0,
    min_tokens: int = 0,
    max_sample_size: int = _MAX_AUTOTUNE_SAMPLE,
) -> UnifiedTuningResult:
    """Unified auto-tune entry point for all task types.

    Results are cached per model, machine, runtime, workload, objective, mode
    and constraints, so a repeated call returns in milliseconds.

    Args:
        task: "embedding", "reranking", "image", or "sparse".
        model_name: Model name (default for "embedding" and "reranking" only).
        full: If True, run the FULL grid, else the QUICK one.
        objective: "latency", "throughput", "balanced" or "memory". None keeps
            the task default: "throughput" for embedding, sparse and image,
            "balanced" for reranking.
        texts: Optional custom workload. Embedding and sparse: the texts to
            embed. Reranking: the last text is the query, the others are the
            documents. Not supported for image. Sampled down to
            ``max_sample_size`` first.
        max_latency_ms: p95 budget for one tuning call (the whole workload),
            0 for none. When nothing fits, the best unconstrained configuration
            is returned: compare ``p95_latency_ms`` with the budget.
        min_tokens: Reranking only, smallest acceptable ``max_tokens``.
        max_sample_size: Upper bound on the number of texts sent to the tuner.

    Returns:
        UnifiedTuningResult with the optimal configuration.
    """
    if task not in _TASKS:
        raise ValueError(f"Unknown task '{task}'. Use: {list(_TASKS.keys())}")
    if objective is not None and objective not in _OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{objective}'. Use: {list(_OBJECTIVES.keys())}"
        )
    if texts and task == "image":
        raise ValueError("A custom workload is not supported for task 'image'.")
    if min_tokens and task != "reranking":
        raise ValueError("min_tokens only applies to task 'reranking'.")
    if max_latency_ms < 0 or min_tokens < 0:
        raise ValueError("max_latency_ms and min_tokens must be >= 0.")

    if model_name is None:
        model_name = _DEFAULT_MODELS.get(task)
        if model_name is None:
            raise ValueError(
                f"No default model for task '{task}'. Please specify model_name."
            )

    resolved_name = _resolve_model_code(model_name) if task == "embedding" else model_name

    options = lib.lembed_autotune_options_default(_TASKS[task])
    options.mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    if objective is not None:
        options.objective = _OBJECTIVES[objective]
    options.max_latency_ms = float(max_latency_ms)
    options.min_tokens = int(min_tokens)

    _owners: list = []
    if texts:
        _check_texts(texts)
        sampled = _sample_workload(task, texts, max_sample_size)
        c_texts, _owners = _to_c_string_array(sampled)
        options.texts = c_texts
        options.n_texts = len(sampled)

    p_options = ffi.new("lembed_autotune_options_t *", options)
    result = ffi.new("lembed_unified_tuning_result_t *")
    check_status(lib.lembed_autotune_ex(resolved_name.encode("utf-8"), p_options, result))

    return UnifiedTuningResult(
        task=task,
        threads=result.threads,
        batch_size=result.batch_size,
        workers=result.workers,
        max_tokens=result.max_tokens,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        memory_mb=result.memory_mb,
        top_k=result.top_k,
        min_weight=result.min_weight,
        storage_format=result.storage_format,
    )


def clear_autotune_cache_unified(task: str, model_name: str | None = None) -> None:
    """Clear the autotune cache of one task for a model (or all if None)."""
    if task not in _TASKS:
        raise ValueError(f"Unknown task '{task}'. Use: {list(_TASKS.keys())}")
    lib.lembed_autotune_unified_clear_cache(
        _TASKS[task], model_name.encode("utf-8") if model_name else ffi.NULL
    )


def cleanup_model_cache(
    keep_models: list[str] | None = None,
    *,
    cache_dir: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Clean up model cache, keeping only specified models.

    Args:
        keep_models: List of model identifiers to keep (e.g., ["BAAI/bge-small-en-v1.5"]).
            If None, all models are kept (no-op unless dry_run=True).
        cache_dir: Custom cache directory (None = default).
        dry_run: If True, only report what would be deleted without actually deleting.

    Returns:
        Dictionary with keys: "deleted_count", "freed_bytes", "details" (list of paths).

    Example:
        >>> result = cleanup_model_cache(keep_models=["BAAI/bge-small-en-v1.5"])
        >>> print(f"Deleted {result['deleted_count']} models, freed {result['freed_bytes']} bytes")
    """
    if keep_models is None:
        keep_models = []

    c_keep = ffi.new("char*[]", len(keep_models) + 1)
    c_strs = []
    for i, m in enumerate(keep_models):
        c_strs.append(ffi.new("char[]", m.encode("utf-8")))
        c_keep[i] = c_strs[i]
    c_keep[len(keep_models)] = ffi.NULL

    cache_dir_c = ffi.NULL
    if cache_dir:
        cache_dir_c = ffi.new("char[]", cache_dir.encode("utf-8"))

    deleted_count = ffi.new("size_t *")
    freed_bytes = ffi.new("uint64_t *")

    check_status(
        lib.lembed_cleanup_model_cache(
            cache_dir_c, c_keep, 1 if dry_run else 0, deleted_count, freed_bytes
        )
    )

    return {
        "deleted_count": deleted_count[0],
        "freed_bytes": freed_bytes[0],
        "details": [],
    }


def cleanup_model_cache_except(
    active_model: str,
    *,
    cache_dir: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Clean up model cache, keeping only the active model.

    Args:
        active_model: Model identifier to keep (e.g., "BAAI/bge-small-en-v1.5").
        cache_dir: Custom cache directory (None = default).
        dry_run: If True, only report what would be deleted without actually deleting.

    Returns:
        Dictionary with keys: "deleted_count", "freed_bytes", "details" (list of paths).

    Example:
        >>> result = cleanup_model_cache_except(active_model="BAAI/bge-small-en-v1.5")
        >>> print(f"Deleted {result['deleted_count']} models, freed {result['freed_bytes']} bytes")
    """
    active_model_c = ffi.new("char[]", active_model.encode("utf-8"))
    cache_dir_c = ffi.NULL
    if cache_dir:
        cache_dir_c = ffi.new("char[]", cache_dir.encode("utf-8"))

    deleted_count = ffi.new("size_t *")
    freed_bytes = ffi.new("uint64_t *")

    check_status(
        lib.lembed_cleanup_model_cache_except(
            cache_dir_c, active_model_c, 1 if dry_run else 0, deleted_count, freed_bytes
        )
    )

    return {
        "deleted_count": deleted_count[0],
        "freed_bytes": freed_bytes[0],
        "details": [],
    }
