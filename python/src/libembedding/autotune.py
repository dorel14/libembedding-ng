"""Auto-tuning functions for optimal embedding configuration."""

from __future__ import annotations

from ._binding import ffi, lib
from ._status import check_status
from .models import resolve_text_model
from .types import TuningResult, ModelSelectionResult, UnifiedTuningResult
from .sampling import _sample_corpus


def _do_autotune(model_name: str, provider: str = "cpu",
                 texts: list[str] | None = None, max_sample_size: int = 100) -> TuningResult:
    """Run autotune with cache lookup."""
    model_idx = resolve_text_model(model_name)
    if model_idx >= 0:
        info_ptr = ffi.new("lembed_model_info_t *")
        lib.lembed_get_text_model_info(model_idx, info_ptr)
        code = ffi.string(info_ptr.model_code).decode("utf-8")
    else:
        code = model_name

    if texts:
        sampled = _sample_corpus(texts, max_sample_size)
        n = len(sampled)
        result = ffi.new("lembed_tuning_result_t *")
        c_texts = ffi.new("char*[]", n)
        c_strs = []
        for i, t in enumerate(sampled):
            c_strs.append(ffi.new("char[]", t.encode("utf-8")))
            c_texts[i] = c_strs[i]
        check_status(lib.lembed_autotune_custom(
            code.encode("utf-8"), c_texts, n,
            lib.LEMBED_AUTOTUNE_QUICK, result))
    else:
        result = ffi.new("lembed_tuning_result_t *")
        check_status(lib.lembed_autotune(
            code.encode("utf-8"), lib.LEMBED_AUTOTUNE_QUICK, result))

    return TuningResult(
        workers=result.workers,
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
    )


def autotune(
    model_name: str = "BAAI/bge-small-en-v1.5",
    *,
    full: bool = False,
) -> TuningResult:
    """Run auto-tuning to find optimal configuration for a model.

    Args:
        model_name: HuggingFace model code (e.g. "Qdrant/all-MiniLM-L6-v2-onnx")
        full: If True, run exhaustive tuning (30-120s). Otherwise quick (5-15s).

    Returns:
        TuningResult with optimal workers, threads, batch_size.
    """
    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    result = ffi.new("lembed_tuning_result_t *")

    model_idx = resolve_text_model(model_name)
    if model_idx >= 0:
        info = ffi.new("lembed_model_info_t *")
        lib.lembed_get_text_model_info(model_idx, info)
        code = ffi.string(info.model_code).decode("utf-8")
    else:
        code = model_name

    check_status(lib.lembed_autotune(code.encode("utf-8"), mode, result))

    return TuningResult(
        workers=result.workers,
        threads=result.threads,
        batch_size=result.batch_size,
        throughput_docs_sec=result.throughput_docs_sec,
        latency_ms=result.latency_ms,
        memory_mb=result.memory_mb,
    )


def auto_select_model(
    use_case: str = "balanced",
) -> "ModelSelectionResult":
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


def clear_autotune_cache(model_name: str = None) -> None:
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
    model_name: str = None,
    *,
    full: bool = False,
) -> "UnifiedTuningResult":
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
            raise ValueError(f"No default model for task '{task}'. Please specify model_name.")

    mode = lib.LEMBED_AUTOTUNE_FULL if full else lib.LEMBED_AUTOTUNE_QUICK
    result = ffi.new("lembed_unified_tuning_result_t *")

    resolved_name = model_name
    if task == "embedding":
        idx = resolve_text_model(model_name)
        if idx >= 0:
            info = ffi.new("lembed_model_info_t *")
            lib.lembed_get_text_model_info(idx, info)
            resolved_name = ffi.string(info.model_code).decode("utf-8")

    check_status(lib.lembed_autotune_unified(task_map[task], resolved_name.encode("utf-8"), mode, result))

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
