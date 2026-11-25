"""Session router: strict-FIFO scheduling of requests over a set of sessions.

``_SessionRouter`` is the private, generic core. A native context runs one call
at a time (``call_mutex``), so concurrency comes from several sessions. The
router hands each request to a free session, queues the others in arrival
order, and bounds the queue.

Hand-off rule: when a session is released and somebody waits, the session goes
straight to the oldest waiter and never passes through the free list. A thread
that releases its session therefore cannot take it back before the thread it
woke up (the starvation seen with a plain ``queue.Queue``).

Auteur: David Orel
Version: 1.16.3

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import numpy as np  # pyright: ignore[reportMissingImports]

from .benchmark import detect_hardware
from .exceptions import RouterBusyError, RouterClosedError
from .models import _is_gguf_model
from .text_embedding import TextEmbedding

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass(frozen=True)
class RouterStats:
    """Snapshot of a router's counters.

    Attributes:
        sessions: Number of sessions the router owns.
        in_flight: Sessions currently handed out (running a call).
        queued: Requests waiting for a session.
        completed: Calls finished (success or error) since creation.
        rejected: Requests refused because the queue was full.
        timeouts: Requests that gave up waiting for a session.
        max_wait_ms: Longest wait for a session, in milliseconds.
    """

    sessions: int
    in_flight: int
    queued: int
    completed: int
    rejected: int
    timeouts: int
    max_wait_ms: float


class _SessionRouter(Generic[_T]):
    """Strict-FIFO router over a fixed set of sessions.

    Args:
        sessions: The sessions to route over (at least one). The router owns
            them: ``close`` closes them, and only when they are idle.
        max_queue: Maximum number of waiting requests; 0 = unbounded. A request
            arriving when the queue is full raises ``RouterBusyError``.
        acquire_timeout: Default maximum wait for a session in seconds
            (None = wait forever). A per-call ``timeout`` overrides it.
        closer: Called with each session on close (default: ``session.close()``).
    """

    def __init__(
        self,
        sessions: Sequence[_T],
        *,
        max_queue: int = 64,
        acquire_timeout: float | None = None,
        closer: Callable[[_T], None] | None = None,
    ) -> None:
        if len(sessions) < 1:
            raise ValueError("a router needs at least one session")
        if max_queue < 0:
            raise ValueError(f"max_queue must be >= 0, got {max_queue}")
        self._lock = threading.Lock()
        # Signalled when a session finished closing (see close()).
        self._cond = threading.Condition(self._lock)
        self._free: deque[_T] = deque(sessions)
        self._waiters: deque[Future[Any]] = deque()
        self._n = len(sessions)
        self._open = self._n
        self._in_flight = 0
        self._closed = False
        self._max_queue = max_queue
        self._acquire_timeout = acquire_timeout
        self._closer = closer
        self._completed = 0
        self._rejected = 0
        self._timeouts = 0
        self._max_wait = 0.0

    # ------------------------------------------------------------ properties

    @property
    def size(self) -> int:
        """Number of sessions owned by the router."""
        return self._n

    @property
    def closed(self) -> bool:
        """True once ``close`` was called."""
        return self._closed

    def stats(self) -> RouterStats:
        """Return a consistent snapshot of the counters."""
        with self._lock:
            return RouterStats(
                sessions=self._n,
                in_flight=self._in_flight,
                queued=len(self._waiters),
                completed=self._completed,
                rejected=self._rejected,
                timeouts=self._timeouts,
                max_wait_ms=self._max_wait * 1000.0,
            )

    # ----------------------------------------------------------- scheduling

    def _enqueue(self) -> tuple[Any, Future[Any] | None]:
        """Take a session now, or join the FIFO.

        Returns ``(session, None)`` when a session was free and nobody waited,
        else ``(None, future)``: the future later resolves to the session (or
        to ``RouterClosedError``). Raises ``RouterClosedError`` or
        ``RouterBusyError`` immediately.
        """
        with self._lock:
            if self._closed:
                raise RouterClosedError()
            if self._free and not self._waiters:
                self._in_flight += 1
                return self._free.popleft(), None
            if self._max_queue and len(self._waiters) >= self._max_queue:
                self._rejected += 1
                raise RouterBusyError(
                    detail=f"{len(self._waiters)} requests already waiting"
                )
            fut: Future[Any] = Future()
            self._waiters.append(fut)
            return None, fut

    def _note_wait(self, started: float) -> None:
        waited = time.perf_counter() - started
        with self._lock:
            self._max_wait = max(self._max_wait, waited)

    def _cancel_waiter(self, fut: Future[Any]) -> bool:
        """Withdraw a waiter. False when a session was already handed to it."""
        if not fut.cancel():
            return False
        with self._lock:
            try:
                self._waiters.remove(fut)
            except ValueError:
                pass
        return True

    def _acquire(self, timeout: float | None) -> Any:
        """Block until a session is handed over (strict FIFO)."""
        started = time.perf_counter()
        session, fut = self._enqueue()
        if fut is None:
            return session
        if timeout is None:
            timeout = self._acquire_timeout
        try:
            session = fut.result(timeout)
        except _FutureTimeoutError:
            self._give_back(fut)
            with self._lock:
                self._timeouts += 1
            raise TimeoutError(f"no session available within {timeout} s") from None
        except BaseException:
            # KeyboardInterrupt and friends: never leave a waiter behind.
            self._give_back(fut)
            raise
        self._note_wait(started)
        return session

    def _give_back(self, fut: Future[Any]) -> None:
        """Abandon a wait: withdraw the waiter, or return a session it received.

        If ``cancel`` loses the race, a session is being (or was) handed to the
        waiter: it is ours, so pass it on instead of leaking it.
        """
        if self._cancel_waiter(fut):
            return
        try:
            session = fut.result()
        except RouterClosedError:
            return
        self._release(session)

    def _release(self, session: _T) -> None:
        """Return a session: hand it to the oldest live waiter, else free it.

        ``set_result`` runs outside the lock because done-callbacks (asyncio
        delivery) may call back into the router.
        """
        with self._lock:
            self._completed += 1
            while self._waiters:
                waiter = self._waiters.popleft()
                # Marks the waiter RUNNING, so a concurrent cancel() now fails
                # and the waiter is guaranteed to receive the session.
                if waiter.set_running_or_notify_cancel():
                    break
            else:
                if not self._closed:
                    self._free.append(session)
                    self._in_flight -= 1
                    return
                waiter = None
        if waiter is not None:
            waiter.set_result(session)
            return
        # Closed while the session was out: close it now that it is idle.
        try:
            self._close_session(session)
        finally:
            with self._lock:
                self._in_flight -= 1
                self._cond.notify_all()

    def _close_session(self, session: _T) -> None:
        try:
            if self._closer is not None:
                self._closer(session)
            else:
                close = getattr(session, "close", None)
                if close is not None:
                    close()
        finally:
            with self._lock:
                self._open -= 1
                self._cond.notify_all()

    # ------------------------------------------------------------------ API

    def run(self, fn: Callable[[_T], _R], timeout: float | None = None) -> _R:
        """Run ``fn(session)`` in the calling thread on a session in FIFO order.

        Raises:
            RouterBusyError: The queue is full.
            RouterClosedError: The router is closed.
            TimeoutError: No session became free within the delay.
        """
        session = self._acquire(timeout)
        try:
            return fn(session)
        finally:
            self._release(session)

    def close(self, *, wait: bool = True) -> None:
        """Close the router and its sessions.

        New requests raise ``RouterClosedError``; waiters are woken with it. A
        busy session is never closed: it is closed when its call returns.
        With ``wait=True`` this blocks until every session is closed; with
        ``wait=False`` it returns at once and the last call to finish closes
        the remaining sessions. Idempotent.
        """
        with self._lock:
            self._closed = True
            waiters = list(self._waiters)
            self._waiters.clear()
            idle = list(self._free)
            self._free.clear()
        for waiter in waiters:
            if waiter.set_running_or_notify_cancel():
                waiter.set_exception(RouterClosedError())
        first_error: BaseException | None = None
        for session in idle:
            try:
                self._close_session(session)
            except Exception as exc:  # noqa: BLE001 - keep closing the others
                if first_error is None:
                    first_error = exc
        if wait:
            with self._cond:
                self._cond.wait_for(lambda: self._open == 0)
        if first_error is not None:
            raise first_error


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------

_MAX_SESSIONS = 8
_POLICIES = ("latency", "balanced", "indexing")


def _physical_cores() -> int:
    """Physical core count P used to size the presets (never < 1)."""
    try:
        cores = int(detect_hardware().physical_cores)
    except Exception:  # noqa: BLE001 - detection must never block construction
        cores = 0
    if cores < 1:
        # Logical count halved: assume SMT. Over-counting would oversubscribe.
        cores = max(1, (os.cpu_count() or 2) // 2)
    return cores


def _preset(policy: str, cores: int, gguf: bool) -> tuple[int, int]:
    """Return ``(sessions, threads)`` for a policy on ``cores`` physical cores.

    ONNX: latency 1 x P, balanced 2 x floor(P/2), indexing min(P, 8) x 1.
    GGUF (one thread per session): latency 1, balanced ceil(P/2), indexing
    min(P, 8) sessions. Sessions are capped at 8 and sessions x threads never
    exceeds P.
    """
    if gguf:
        sessions = {
            "latency": 1,
            "balanced": (cores + 1) // 2,
            "indexing": cores,
        }[policy]
        return max(1, min(sessions, _MAX_SESSIONS)), 1
    if policy == "latency":
        return 1, cores
    if policy == "balanced":
        if cores // 2 < 1:
            return 1, cores
        return 2, cores // 2
    return max(1, min(cores, _MAX_SESSIONS)), 1


class TextEmbeddingRouter:
    """Serve concurrent embedding requests from several ``TextEmbedding`` sessions.

    A native context runs one call at a time, so a server that embeds from many
    threads (or asyncio tasks) needs several sessions. The router owns them and
    hands each request to a free one in strict arrival order. When all sessions
    are busy the request waits in a bounded FIFO. This differs from
    ``TextEmbeddingPool``, which splits *one* large call across workers: the
    router keeps each call whole and runs many calls at once.

    Args:
        model_name: Same as ``TextEmbedding``.
        policy: Preset that sizes sessions and threads from the physical core
            count P: ``"latency"`` (1 session x P threads, the default),
            ``"balanced"`` (2 x P/2) or ``"indexing"`` (min(P, 8) x 1). GGUF
            models always use 1 thread per session. Ignored when ``sessions``
            is given.
        sessions: Explicit number of sessions (disables ``policy``).
        threads: Intra-op threads per session. Defaults to the preset value, or
            P // sessions when ``sessions`` is given (1 for GGUF).
        max_queue: Maximum waiting requests; one more raises
            ``RouterBusyError`` at once. 0 = unbounded.
        acquire_timeout: Default maximum wait for a session, in seconds
            (None = wait forever). Raises ``TimeoutError``.
        **session_kwargs: Passed to every ``TextEmbedding`` (``batch_size``,
            ``cache_size``, ``quantization``, ``backend``...). Note that
            ``cache_size`` is per session: the LRU cache is not shared.

    Raises:
        ResourceWarning: (warning) sessions x threads exceeds P.

    Example::

        with TextEmbeddingRouter("BAAI/bge-small-en-v1.5") as router:
            vectors = router.embed(["a query"])          # from any thread
            vectors = await router.aembed(["a query"])   # from asyncio
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        policy: Literal["latency", "balanced", "indexing"] | None = "latency",
        sessions: int | None = None,
        threads: int | None = None,
        max_queue: int = 64,
        acquire_timeout: float | None = None,
        **session_kwargs: Any,
    ) -> None:
        for name, value in (("sessions", sessions), ("threads", threads)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        if isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue < 0:
            raise ValueError(f"max_queue must be an int >= 0, got {max_queue!r}")
        if acquire_timeout is not None and acquire_timeout <= 0:
            raise ValueError(f"acquire_timeout must be > 0, got {acquire_timeout!r}")
        if sessions is None and policy not in _POLICIES:
            raise ValueError(
                f"policy must be one of {_POLICIES} (or give sessions=), got {policy!r}"
            )

        gguf = _is_gguf_model(model_name) or str(
            session_kwargs.get("backend", "auto")
        ).lower() in ("llama", "llamacpp", "llama.cpp")
        cores = _physical_cores()
        if sessions is None:
            assert policy is not None
            n_sessions, preset_threads = _preset(policy, cores, gguf)
            self._policy: str | None = policy
        else:
            n_sessions = sessions
            preset_threads = 1 if gguf else max(1, cores // n_sessions)
            self._policy = None
        n_threads = threads if threads is not None else preset_threads
        if gguf and threads is None:
            n_threads = 1
        if n_sessions * n_threads > cores:
            warnings.warn(
                f"TextEmbeddingRouter: {n_sessions} sessions x {n_threads} threads "
                f"= {n_sessions * n_threads} exceeds the {cores} physical cores; "
                "oversubscription usually lowers throughput.",
                ResourceWarning,
                stacklevel=2,
            )

        models: list[TextEmbedding] = []
        try:
            for i in range(n_sessions):
                kwargs = dict(session_kwargs)
                kwargs["threads"] = n_threads
                if i > 0:
                    kwargs["show_download_progress"] = False
                models.append(TextEmbedding(model_name, **kwargs))
        except BaseException:
            for m in models:
                m.close()
            raise

        self._sessions = n_sessions
        self._threads = n_threads
        self._dim = models[0].dim
        self._core: _SessionRouter[TextEmbedding] = _SessionRouter(
            models, max_queue=max_queue, acquire_timeout=acquire_timeout
        )

    # ------------------------------------------------------------ properties

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def sessions(self) -> int:
        """Number of ``TextEmbedding`` sessions."""
        return self._sessions

    @property
    def threads(self) -> int:
        """Intra-op threads per session."""
        return self._threads

    @property
    def policy(self) -> str | None:
        """Preset in use, or None when ``sessions`` was given explicitly."""
        return self._policy

    def stats(self) -> RouterStats:
        """Snapshot of in-flight, queued, rejected and timed-out requests."""
        return self._core.stats()

    # ------------------------------------------------------------------ sync

    def embed(
        self, texts: list[str], *, timeout: float | None = None, **embed_kwargs: Any
    ) -> np.ndarray:
        """Embed texts on the next free session, in arrival order.

        Runs in the calling thread; safe to call from many threads at once.

        Args:
            texts: Strings to embed. The whole list runs on one session.
            timeout: Maximum wait for a session in seconds (None = the
                ``acquire_timeout`` of the router). The native call itself is
                not interruptible and not covered by this delay.
            **embed_kwargs: ``batch_size``, ``dtype``, ``normalized`` as in
                ``TextEmbedding.embed``.

        Returns:
            Array of shape ``(len(texts), dim)``.

        Raises:
            RouterBusyError: ``max_queue`` requests are already waiting.
            RouterClosedError: The router is closed.
            TimeoutError: No session became free within the delay.
        """
        return self._core.run(lambda m: m.embed(texts, **embed_kwargs), timeout)

    def close(self, *, wait: bool = True) -> None:
        """Close the router and release every session.

        New requests raise ``RouterClosedError`` and queued requests are woken
        with it. A session is never closed while a call runs on it. With
        ``wait=True`` (default) this blocks until in-flight calls finish and all
        sessions are closed. Idempotent.
        """
        self._core.close(wait=wait)

    def __enter__(self) -> TextEmbeddingRouter:  # noqa: PYI034 - no Self on 3.9
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"TextEmbeddingRouter(sessions={self._sessions}, "
            f"threads={self._threads}, dim={self._dim}, policy={self._policy!r})"
        )
