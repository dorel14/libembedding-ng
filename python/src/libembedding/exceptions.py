"""Exception hierarchy mapped from lembed_status_t.

Auteur: David Orel
Version: 1.4.0
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

