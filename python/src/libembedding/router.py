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
Version: 1.16.6

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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
        # Dedicated executor for native calls made through arun(); created on
        # first use so synchronous users never spawn a thread.
        self._executor: ThreadPoolExecutor | None = None
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

    def _get_executor(self) -> ThreadPoolExecutor:
        """Dedicated executor: as many workers as sessions, never the loop's.

        A job only starts once it holds a session, so at most ``size`` jobs
        exist at a time and none ever sits in the executor's own queue.
        """
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._n, thread_name_prefix="lembed-router"
                )
            return self._executor

    async def _await_session(
        self,
        loop: asyncio.AbstractEventLoop,
        fut: Future[Any],
        timeout: float | None,
        started: float,
    ) -> Any:
        """Wait for a hand-off without blocking the loop or leaking a session.

        ``asyncio.wrap_future`` is not enough here: if the task is cancelled
        (or times out) at the moment ``_release`` hands it a session, the
        cancellation reaches the source future too late and the session would
        be lost. Delivery therefore goes through ``box`` and a flag that are
        only touched from the loop thread, which makes the two orders
        (delivered first / abandoned first) mutually exclusive.
        """
        if timeout is None:
            timeout = self._acquire_timeout
        afut: asyncio.Future[None] = loop.create_future()
        box: list[Any] = []
        abandoned = False

        def deliver(f: Future[Any]) -> None:  # loop thread
            if f.cancelled():
                return
            exc = f.exception()
            if exc is not None:
                if not afut.done():
                    afut.set_exception(exc)
                return
            session = f.result()
            if abandoned or afut.done():
                self._release(session)
            else:
                box.append(session)
                afut.set_result(None)

        def on_done(f: Future[Any]) -> None:  # any thread
            try:
                loop.call_soon_threadsafe(deliver, f)
            except RuntimeError:
                # Loop already closed: nobody can use the session.
                if not f.cancelled() and f.exception() is None:
                    self._release(f.result())

        fut.add_done_callback(on_done)
        try:
            await asyncio.wait_for(afut, timeout)
        except BaseException as exc:
            abandoned = True
            if box:
                # Delivered, but the task died before it could use it.
                self._release(box.pop())
            else:
                # Withdraw; if a session is already on its way, ``deliver``
                # sees ``abandoned`` and passes it on.
                self._cancel_waiter(fut)
            if isinstance(exc, asyncio.TimeoutError):
                with self._lock:
                    self._timeouts += 1
                raise TimeoutError(f"no session available within {timeout} s") from None
            raise
        self._note_wait(started)
        return box.pop()

    async def arun(
        self, fn: Callable[[_T], _R], timeout: float | None = None
    ) -> _R:
        """Async ``run``: wait in the loop, run ``fn(session)`` in the executor.

        The wait for a session is an ``await`` (no thread parked). ``fn`` runs
        on the router's dedicated executor, never on the loop's default one.
        Cancelling the task while it waits withdraws it; cancelling it during
        ``fn`` raises ``CancelledError`` at once, but ``fn`` cannot be
        interrupted: it finishes in the executor, its result is discarded, and
        only then is the session handed on.
        """
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        session, fut = self._enqueue()
        if fut is not None:
            session = await self._await_session(loop, fut, timeout, started)
        # From here the session is ours and must come back on every path.

        def job() -> _R:
            try:
                return fn(session)
            finally:
                self._release(session)

        try:
            cf = self._get_executor().submit(job)
        except BaseException as exc:
            self._release(session)
            if isinstance(exc, RuntimeError):  # executor already shut down
                raise RouterClosedError() from None
            raise
        # Cancelled before it started: job() will never run, so give back here.
        cf.add_done_callback(
            lambda f: self._release(session) if f.cancelled() else None
        )
        return await asyncio.wrap_future(cf, loop=loop)

    async def aclose(self) -> None:
        """Async ``close(wait=True)`` that does not block the loop."""
        loop = asyncio.get_running_loop()
        closer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lembed-close")
        try:
            await loop.run_in_executor(closer, lambda: self.close(wait=True))
        finally:
            closer.shutdown(wait=False)

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
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            # wait=True: every call finished (sessions are closed), so the
            # workers are idle and this returns at once.
            executor.shutdown(wait=wait)
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
        RuntimeWarning: (warning) sessions x threads exceeds P. Not a
            ``ResourceWarning``: Python hides those by default.

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
                RuntimeWarning,
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

    async def aembed(
        self, texts: list[str], *, timeout: float | None = None, **embed_kwargs: Any
    ) -> np.ndarray:
        """Async ``embed``: the event loop is never blocked.

        The wait for a session is an ``await`` in the loop (no thread is parked
        and the queue stays strict FIFO); the native call runs on an executor
        dedicated to this router, so the loop's default executor stays free.
        Python 3.9+ (no ``asyncio.timeout`` needed).

        Cancellation:
            * while waiting for a session: the request is withdrawn cleanly;
            * during the native call: ``CancelledError`` is raised at once but
              the call cannot be interrupted. It finishes in the executor, its
              result is discarded, and only then does the session serve the
              next request. Bound ``batch_size`` and the size of ``texts`` on
              the server side so that one abandoned call stays short.

        Args and Raises are those of :meth:`embed`; ``timeout`` covers the wait
        for a session only.
        """
        return await self._core.arun(lambda m: m.embed(texts, **embed_kwargs), timeout)

    async def aclose(self) -> None:
        """Async :meth:`close` (waits for in-flight calls without blocking the loop)."""
        await self._core.aclose()

    async def __aenter__(self) -> TextEmbeddingRouter:  # noqa: PYI034 - no Self on 3.9
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def close(self, *, wait: bool = True) -> None:
        """Close the router and release every session.

        From a coroutine use :meth:`aclose`: this call blocks the thread until
        in-flight calls finish. New requests raise ``RouterClosedError`` and queued requests are woken
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
