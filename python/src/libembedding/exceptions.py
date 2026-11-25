"""Exception hierarchy mapped from lembed_status_t.

Auteur: David Orel
Version: 1.16.3
"""


class LembedError(Exception):
    """Base exception for libembedding errors."""

    def __init__(self, status_code: int, message: str, detail: str = ""):
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(f"{message}: {detail}" if detail else message)


class InvalidArgumentError(LembedError):
    pass


class OutOfMemoryError(LembedError):
    pass


class OnnxRuntimeError(LembedError):
    pass


class TokenizerError(LembedError):
    pass


class DownloadError(LembedError):
    pass


class IOError(LembedError):
    pass


class ModelNotFoundError(LembedError):
    pass


class UnsupportedError(LembedError):
    pass


class BatchSizeError(LembedError):
    pass


class LlamaError(LembedError):
    pass


class CacheMissError(LembedError):
    pass


class GgufError(LembedError):
    pass


class RouterBusyError(LembedError):
    """The router queue is full: the request was rejected without waiting.

    Raised by ``TextEmbeddingRouter`` when ``max_queue`` requests are already
    waiting for a session. Map it to a retryable error (HTTP 503) in a server.
    """

    def __init__(self, message: str = "router queue is full", detail: str = ""):
        # No native status behind this error: 0 is LEMBED_OK, so use -1.
        super().__init__(-1, message, detail)


class RouterClosedError(LembedError):
    """The router is closed (or closing): it accepts no new request."""

    def __init__(self, message: str = "router is closed", detail: str = ""):
        super().__init__(-1, message, detail)
