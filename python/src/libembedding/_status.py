"""Status code checking Ã¢â‚¬â€ raises Python exceptions from C status codes.

Auteur: David Orel
Version: 1.4.0
"""

from ._binding import ffi, lib
from .exceptions import (
    BatchSizeError,
    DownloadError,
    InvalidArgumentError,
    IOError,
    LembedError,
    LlamaError,
    ModelNotFoundError,
    OnnxRuntimeError,
    OutOfMemoryError,
    TokenizerError,
    UnsupportedError,
)

_STATUS_MAP = {
    1: InvalidArgumentError,
    2: OutOfMemoryError,
    3: OnnxRuntimeError,
    4: TokenizerError,
    5: DownloadError,
    6: IOError,
    7: ModelNotFoundError,
    8: UnsupportedError,
    9: BatchSizeError,
    10: LlamaError,
}


def check_status(status: int) -> None:
    if status == 0:
        return
    msg = ffi.string(lib.lembed_status_message(status)).decode("utf-8", errors="replace")
    detail = ffi.string(lib.lembed_last_error()).decode("utf-8", errors="replace")
    exc_cls = _STATUS_MAP.get(status, LembedError)
    raise exc_cls(status, msg, detail)


