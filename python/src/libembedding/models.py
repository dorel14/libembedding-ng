"""Model name resolution and registry queries.

Auteur: David Orel
Version: 1.4.0
"""

import os

from ._binding import ffi, lib
from ._status import check_status
from .exceptions import ModelNotFoundError
from .types import ModelDesc, ModelInfo

_PROVIDER_MAP = {
    "cpu": 0,
    "cuda": 1,
    "coreml": 2,
    "directml": 3,
    "tensorrt": 4,
    "llama": 5,
    "llamacpp": 5,
}

_PROVIDER_NAMES = {v: k for k, v in _PROVIDER_MAP.items()}
# Prefer "llama" for provider value 5
_PROVIDER_NAMES[5] = "llama"

_POOLING_NAMES = {0: "cls", 1: "mean"}
_POOLING_ENUM = {v: k for k, v in _POOLING_NAMES.items()}

_QUANTIZATION_NAMES = {0: "none", 1: "static", 2: "dynamic"}


def _model_info_from_c(info) -> ModelInfo:
    return ModelInfo(
        model_name=ffi.string(info.model_name).decode(),
        model_code=ffi.string(info.model_code).decode(),
        model_file=ffi.string(info.model_file).decode(),
        description=ffi.string(info.description).decode(),
        dim=info.dim,
        max_tokens=info.max_tokens,
        pooling=_POOLING_NAMES.get(info.pooling, "unknown"),
        quantization=_QUANTIZATION_NAMES.get(info.quantization, "unknown"),
    )


def resolve_text_model(model_name: str) -> int:
    idx = lib.lembed_find_text_model_by_code(model_name.encode("utf-8"))
    if idx >= 0:
        return idx
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(lib.lembed_list_text_models(models_ptr, count_ptr))
    for i in range(count_ptr[0]):
        if ffi.string(models_ptr[0][i].model_name).decode() == model_name:
            return i
    raise ModelNotFoundError(7, "Model not found", f"No text model matching '{model_name}'")


def resolve_sparse_model(model_name: str) -> int:
    idx = lib.lembed_find_sparse_model_by_code(model_name.encode("utf-8"))
    if idx >= 0:
        return idx
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(lib.lembed_list_sparse_models(models_ptr, count_ptr))
    for i in range(count_ptr[0]):
        if ffi.string(models_ptr[0][i].model_name).decode() == model_name:
            return i
    raise ModelNotFoundError(7, "Model not found", f"No sparse model matching '{model_name}'")


def resolve_image_model(model_name: str) -> int:
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(lib.lembed_list_image_models(models_ptr, count_ptr))
    for i in range(count_ptr[0]):
        name = ffi.string(models_ptr[0][i].model_name).decode()
        code = ffi.string(models_ptr[0][i].model_code).decode()
        if model_name in (name, code):
            return i
    raise ModelNotFoundError(7, "Model not found", f"No image model matching '{model_name}'")


def resolve_reranker_model(model_name: str) -> int:
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(lib.lembed_list_reranker_models(models_ptr, count_ptr))
    for i in range(count_ptr[0]):
        name = ffi.string(models_ptr[0][i].model_name).decode()
        code = ffi.string(models_ptr[0][i].model_code).decode()
        if model_name in (name, code):
            return i
    raise ModelNotFoundError(7, "Model not found", f"No reranker model matching '{model_name}'")


def _list_models(list_fn) -> list[ModelInfo]:
    models_ptr = ffi.new("lembed_model_info_t const **")
    count_ptr = ffi.new("int *")
    check_status(list_fn(models_ptr, count_ptr))
    return [_model_info_from_c(models_ptr[0][i]) for i in range(count_ptr[0])]


def list_text_models() -> list[ModelInfo]:
    return _list_models(lib.lembed_list_text_models)


def list_sparse_models() -> list[ModelInfo]:
    return _list_models(lib.lembed_list_sparse_models)


def list_image_models() -> list[ModelInfo]:
    return _list_models(lib.lembed_list_image_models)


def list_reranker_models() -> list[ModelInfo]:
    return _list_models(lib.lembed_list_reranker_models)


def _desc_from_c(desc_ptr) -> ModelDesc:
    """Convert a C lembed_model_desc_t pointer to ModelDesc."""
    name = ffi.string(desc_ptr.name).decode("utf-8", errors="replace") if desc_ptr.name else ""
    return ModelDesc(
        name=name,
        dimension=desc_ptr.dimension,
        max_length=desc_ptr.max_length,
        pooling=_POOLING_NAMES.get(desc_ptr.pooling, "unknown"),
        num_threads=desc_ptr.num_threads,
        batch_size=desc_ptr.batch_size,
        provider=_PROVIDER_NAMES.get(desc_ptr.provider, "unknown"),
        device_id=desc_ptr.device_id,
    )


def _is_local_path(model_name: str) -> bool:
    """True if the given string looks like an existing local directory path or GGUF file."""
    if model_name.endswith(".gguf"):
        return os.path.isfile(model_name)
    return os.path.exists(model_name)


def _is_gguf_model(model_name: str) -> bool:
    """True if the model name looks like a GGUF file path."""
    return model_name.endswith((".gguf", ".GGUF"))

