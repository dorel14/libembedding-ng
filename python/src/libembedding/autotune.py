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


def autotune_unified(
    task: str = "embedding",
    model_name: str | None = None,
    *,
    full: bool = False,
) -> UnifiedTuningResult:
    """Unified auto-tune entry point for all task types.

    Args:
        task: "embedding", "reranking", "image", or "sparse"
        model_name: Model name (default depends on task)
        full: If True, run FULL mode (30-120s), else QUICK (5-15s)

    Returns:
        UnifiedTuningResult with optimal configuration.
    """
    task_map = {
        "embedding": _TASK_EMBEDDING,
        "reranking": _TASK_RERANKING,
        "image": _TASK_IMAGE,
        "sparse": _TASK_SPARSE,
    }
    if task not in task_map:
        raise ValueError(f"Unknown task '{task}'. Use: {list(task_map.keys())}")

    if model_name is None:
        defaults = {
            "embedding": "BAAI/bge-small-en-v1.5",
            "reranking": "jinaai/jina-reranker-v1-turbo-en-quantized",
            "image": None,
            "sparse": None,
        }
        model_name = defaults.get(task)
        if model_name is None:
            raise ValueError(
                f"No default model for task '{task}'. Please specify model_name."
            )

    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    result = ffi.new("lembed_unified_tuning_result_t *")

    resolved_name = model_name
    if task == "embedding":
        idx = resolve_text_model(model_name)
        if idx >= 0:
            info = ffi.new("lembed_model_info_t *")
            lib.lembed_get_text_model_info(idx, info)
            resolved_name = ffi.string(info.model_code).decode("utf-8")

    check_status(
        lib.lembed_autotune_unified(
            task_map[task], resolved_name.encode("utf-8"), mode, result
        )
    )

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
