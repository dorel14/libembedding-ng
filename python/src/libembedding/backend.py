"""Backend auto-detection for libembedding."""

from __future__ import annotations

import os

from .models import _is_gguf_model, _is_local_path

# Backend enum values (must match types.h)
_BACKEND_ONNX = 0
_BACKEND_LLAMACPP = 1
_BACKEND_AUTO = 2

# Mapping from string to enum
_BACKEND_ENUM = {
    "cpu": _BACKEND_ONNX,
    "onnx": _BACKEND_ONNX,
    "llama": _BACKEND_LLAMACPP,
    "llamacpp": _BACKEND_LLAMACPP,
    "auto": _BACKEND_AUTO,
}


def detect_backend(model_name: str, backend: str = "auto") -> str:
    """Detect which backend to use based on model name and user preference.

    Args:
        model_name: Model name, path, or HuggingFace ID
        backend: "auto", "onnx", or "llama"

    Returns:
        Resolved backend string: "onnx" or "llama"
    """
    if backend != "auto":
        return backend

    # Explicit file extension
    lowered = model_name.lower()
    if lowered.endswith(".gguf"):
        return "llama"
    if lowered.endswith(".onnx"):
        return "onnx"

    # Local path detection
    if _is_local_path(model_name):
        if os.path.isdir(model_name):
            for f in os.listdir(model_name):
                if f.endswith(".onnx"):
                    return "onnx"
                if f.endswith(".gguf"):
                    return "llama"
        return "onnx"

    # HuggingFace model ID
    if _is_gguf_model(model_name):
        return "llama"

    return "onnx"


def backend_to_enum(backend: str) -> int:
    """Convert backend string to C enum value."""
    return _BACKEND_ENUM.get(backend.lower(), _BACKEND_AUTO)
