"""Tests for the session router (core scheduling, hermetic).

The core tests use fake sessions driven by ``threading.Event``: no model, no
network.

Auteur: David Orel

SPDX-License-Identifier: MIT
"""

import asyncio
import random
import threading
import time
from typing import ClassVar

import numpy as np
import pytest

from libembedding.exceptions import LembedError, RouterBusyError, RouterClosedError
from libembedding.router import _SessionRouter


class FakeSession:
    """Session that detects concurrent use and closing while busy."""

    def __init__(self, ident=0):
        self.ident = ident
        self.busy = False
        self.closed = False
        self.uses = 0
        self._guard = threading.Lock()

    def use(self, work=None):
        with self._guard:
            assert not self.busy, "two calls on one session at the same time"
            assert not self.closed, "call on a closed session"
            self.busy = True
        try:
            if work is not None:
                work()
            self.uses += 1
        finally:
            self.busy = False

    def close(self):
        assert not self.busy, "session closed while busy"
        self.closed = True


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    spins = 0
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        spins += 1
        # Yield first (fast on the usual path), then back off.
        time.sleep(0 if spins < 5000 else 0.0005)


def _hold(router, gate, started):
    """Start a thread that keeps a session busy until ``gate`` is set."""

    def work(session):
        started.set()
        gate.wait(10)

    t = threading.Thread(target=lambda: router.run(work))
    t.start()
    assert started.wait(5)
    return t


def test_exceptions_are_lembed_errors():
    assert issubclass(RouterBusyError, LembedError)
    assert issubclass(RouterClosedError, LembedError)
    assert "full" in str(RouterBusyError())
    assert "closed" in str(RouterClosedError())


def test_requires_a_session():
    with pytest.raises(ValueError):
        _SessionRouter([])
    with pytest.raises(ValueError):
        _SessionRouter([FakeSession()], max_queue=-1)


def test_run_returns_result_and_stats():
    router = _SessionRouter([FakeSession(0), FakeSession(1)])
    assert router.run(lambda s: s.ident >= 0) is True
    st = router.stats()
    assert st.sessions == 2
    assert st.in_flight == 0
    assert st.queued == 0
    assert st.completed == 1
    router.close()


def test_error_releases_session():
    sessions = [FakeSession()]
    router = _SessionRouter(sessions)

    def boom(session):
        raise LembedError(3, "native failure")

    with pytest.raises(LembedError):
        router.run(boom)
    assert router.stats().in_flight == 0
    assert router.run(lambda s: 42) == 42
    router.close()


@pytest.mark.parametrize("n_threads", [16])
def test_fifo_order_without_inversion(n_threads):
    """Acquisition order equals arrival order, over 1000 rounds."""
    router = _SessionRouter([FakeSession()], max_queue=0)
    # Persistent client threads (thread creation would dominate the runtime):
    # each round, the main thread holds the only session, releases the clients
    # one at a time and waits until each one is queued, then lets go.
    go = [threading.Semaphore(0) for _ in range(n_threads)]
    done = threading.Semaphore(0)
    order = []
    stop = []

    def client(i):
        while True:
            go[i].acquire()
            if stop:
                return
            router.run(lambda s: order.append(i))
            done.release()

    threads = [threading.Thread(target=client, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    try:
        for _ in range(1000):
            del order[:]
            held = router._acquire(None)
            for i in range(n_threads):
                go[i].release()
                # Arrival order is only defined once the client is queued.
                _wait_until(lambda i=i: router.stats().queued == i + 1)
            router._release(held)
            for _ in range(n_threads):
                assert done.acquire(timeout=5)
            assert order == list(range(n_threads))
    finally:
        stop.append(True)
        for g in go:
            g.release()
        for t in threads:
            t.join()
    assert router.stats().in_flight == 0
    router.close()


def test_no_barging_on_release():
    """A releasing thread cannot take its session back before the waiter."""
    router = _SessionRouter([FakeSession()], max_queue=0)
    gate, started = threading.Event(), threading.Event()
    got = []

    def hog(session):
        started.set()
        gate.wait(10)

    def hog_thread():
        router.run(hog)
        # Immediately ask again: the waiter was queued first and must win.
        router.run(lambda s: got.append("hog"))

    t = threading.Thread(target=hog_thread)
    t.start()
    assert started.wait(5)
    w = threading.Thread(target=lambda: router.run(lambda s: got.append("waiter")))
    w.start()
    _wait_until(lambda: router.stats().queued == 1)
    gate.set()
    t.join()
    w.join()
    assert got == ["waiter", "hog"]
    router.close()


def test_fairness_jain_index():
    sessions = [FakeSession(0), FakeSession(1)]
    router = _SessionRouter(sessions, max_queue=0)
    n_clients = 8
    counts = [0] * n_clients
    stop = threading.Event()

    def client(i):
        while not stop.is_set():
            router.run(lambda s: time.sleep(0.0005))
            counts[i] += 1

    threads = [threading.Thread(target=client, args=(i,)) for i in range(n_clients)]
    for t in threads:
        t.start()
    time.sleep(0.8)
    stop.set()
    for t in threads:
        t.join()
    assert min(counts) > 0
    jain = sum(counts) ** 2 / (n_clients * sum(c * c for c in counts))
    assert jain >= 0.95, (jain, counts)
    router.close()


def test_backpressure_rejects_immediately():
    router = _SessionRouter([FakeSession()], max_queue=2)
    gate, started = threading.Event(), threading.Event()
    holder = _hold(router, gate, started)
    waiters = [
        threading.Thread(target=lambda: router.run(lambda s: None)) for _ in range(2)
    ]
    for w in waiters:
        w.start()
    _wait_until(lambda: router.stats().queued == 2)

    t0 = time.perf_counter()
    with pytest.raises(RouterBusyError):
        router.run(lambda s: None)
    assert time.perf_counter() - t0 < 0.5
    assert router.stats().rejected == 1
    assert router.stats().queued == 2

    gate.set()
    holder.join()
    for w in waiters:
        w.join()
    assert router.stats().in_flight == 0
    router.close()


def test_unbounded_queue_when_max_queue_zero():
    router = _SessionRouter([FakeSession()], max_queue=0)
    gate, started = threading.Event(), threading.Event()
    holder = _hold(router, gate, started)
    waiters = [
        threading.Thread(target=lambda: router.run(lambda s: None)) for _ in range(40)
    ]
    for w in waiters:
        w.start()
    _wait_until(lambda: router.stats().queued == 40)
    gate.set()
    holder.join()
    for w in waiters:
        w.join()
    assert router.stats().rejected == 0
    router.close()


def test_timeout_does_not_leak_session():
    router = _SessionRouter([FakeSession()])
    gate, started = threading.Event(), threading.Event()
    holder = _hold(router, gate, started)
    with pytest.raises(TimeoutError):
        router.run(lambda s: None, timeout=0.05)
    st = router.stats()
    assert st.timeouts == 1
    assert st.queued == 0
    gate.set()
    holder.join()
    assert router.stats().in_flight == 0
    assert router.run(lambda s: "ok") == "ok"
    router.close()


def test_default_acquire_timeout():
    router = _SessionRouter([FakeSession()], acquire_timeout=0.05)
    gate, started = threading.Event(), threading.Event()
    holder = _hold(router, gate, started)
    with pytest.raises(TimeoutError):
        router.run(lambda s: None)
    gate.set()
    holder.join()
    router.close()


def test_timeout_race_stress_keeps_every_session():
    """Tiny delays racing with hand-offs: no session is ever lost."""
    sessions = [FakeSession(i) for i in range(3)]
    router = _SessionRouter(sessions, max_queue=0)

    def worker(seed):
        for k in range(150):
            delay = ((seed * 31 + k * 17) % 7) * 0.0003
            try:
                router.run(lambda s: s.use(lambda: time.sleep(0.0002)), timeout=delay)
            except TimeoutError:
                pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st = router.stats()
    assert st.in_flight == 0
    assert st.queued == 0
    assert len(router._free) == len(sessions)
    router.close()


def test_close_wakes_waiters_and_waits_for_in_flight():
    session = FakeSession()
    router = _SessionRouter([session], max_queue=0)
    gate, started = threading.Event(), threading.Event()
    holder = _hold(router, gate, started)
    errors = []

    def waiter():
        try:
            router.run(lambda s: None)
        except RouterClosedError as exc:
            errors.append(exc)

    waiters = [threading.Thread(target=waiter) for _ in range(3)]
    for w in waiters:
        w.start()
    _wait_until(lambda: router.stats().queued == 3)

    closer = threading.Thread(target=router.close)
    closer.start()
    for w in waiters:
        w.join(5)
    assert len(errors) == 3
    # The session is busy: close() must still be waiting and must not have
    # closed it.
    time.sleep(0.05)
    assert closer.is_alive()
    assert not session.closed

    with pytest.raises(RouterClosedError):
        router.run(lambda s: None)

    gate.set()
    holder.join()
    closer.join(5)
    assert not closer.is_alive()
    assert session.closed


def test_close_without_wait_closes_after_last_call():
    session = FakeSession()
    router = _SessionRouter([session])
    gate, started = threading.Event(), threading.Event()
    holder = _hold(router, gate, started)
    router.close(wait=False)
    assert not session.closed
    gate.set()
    holder.join()
    _wait_until(lambda: session.closed)


def test_close_is_idempotent_and_closes_idle_sessions():
    sessions = [FakeSession(0), FakeSession(1)]
    router = _SessionRouter(sessions)
    router.close()
    router.close()
    assert all(s.closed for s in sessions)
    assert router.closed


# ---------------------------------------------------------------------------
# TextEmbeddingRouter (fake TextEmbedding, hermetic)
# ---------------------------------------------------------------------------


class FakeModel:
    """Stand-in for TextEmbedding recording its construction arguments."""

    instances: ClassVar[list] = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.dim = 4
        self.closed = False
        self.calls = []
        FakeModel.instances.append(self)

    def embed(self, texts, **kwargs):
        assert not self.closed
        self.calls.append((list(texts), kwargs))
        return np.full((len(texts), self.dim), float(len(texts)), dtype=np.float32)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_router(monkeypatch):
    """Router factory on FakeModel with a pinned physical core count."""
    import libembedding.router as router_mod

    FakeModel.instances = []
    monkeypatch.setattr(router_mod, "TextEmbedding", FakeModel)

    def make(cores=4, model="m", **kwargs):
        monkeypatch.setattr(router_mod, "_physical_cores", lambda: cores)
        return router_mod.TextEmbeddingRouter(model, **kwargs)

    yield make
    for m in FakeModel.instances:
        m.close()


@pytest.mark.parametrize(
    ("policy", "cores", "expected"),
    [
        ("latency", 4, (1, 4)),
        ("balanced", 4, (2, 2)),
        ("indexing", 4, (4, 1)),
        ("latency", 16, (1, 16)),
        ("balanced", 16, (2, 8)),
        ("indexing", 16, (8, 1)),  # capped at 8 sessions
        ("balanced", 1, (1, 1)),  # P/2 < 1: falls back on latency
        ("balanced", 2, (2, 1)),
        ("indexing", 1, (1, 1)),
    ],
)
def test_onnx_presets(fake_router, policy, cores, expected):
    r = fake_router(cores=cores, policy=policy)
    assert (r.sessions, r.threads) == expected
    assert r.policy == policy
    assert all(m.kwargs["threads"] == expected[1] for m in FakeModel.instances)
    r.close()


@pytest.mark.parametrize(
    ("policy", "cores", "sessions"),
    [
        ("latency", 4, 1),
        ("balanced", 4, 2),
        ("balanced", 5, 3),
        ("indexing", 4, 4),
        ("indexing", 32, 8),
    ],
)
def test_gguf_presets_use_one_thread_per_session(
    fake_router, policy, cores, sessions
):
    r = fake_router(cores=cores, model="x/model.gguf", policy=policy)
    assert (r.sessions, r.threads) == (sessions, 1)
    r.close()


def test_default_policy_is_latency(fake_router):
    r = fake_router(cores=4)
    assert (r.policy, r.sessions, r.threads) == ("latency", 1, 4)
    r.close()


def test_sessions_override_disables_policy(fake_router):
    r = fake_router(cores=8, sessions=4)
    assert r.policy is None
    assert (r.sessions, r.threads) == (4, 2)
    r.close()


def test_explicit_threads_override(fake_router):
    r = fake_router(cores=8, policy="balanced", threads=3)
    assert (r.sessions, r.threads) == (2, 3)
    r.close()


def test_budget_overrun_warns(fake_router):
    with pytest.warns(RuntimeWarning, match="exceeds"):
        r = fake_router(cores=4, sessions=2, threads=4)
    r.close()


def test_presets_stay_within_budget(fake_router, recwarn):
    for policy in ("latency", "balanced", "indexing"):
        fake_router(cores=6, policy=policy).close()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "fast"},
        {"policy": None},
        {"sessions": 0},
        {"threads": 0},
        {"sessions": True},
        {"max_queue": -1},
        {"acquire_timeout": 0},
    ],
)
def test_invalid_arguments(fake_router, kwargs):
    with pytest.raises(ValueError):
        fake_router(**kwargs)
    assert FakeModel.instances == []


def test_failed_construction_closes_built_sessions(monkeypatch):
    import libembedding.router as router_mod

    FakeModel.instances = []
    built = []

    def flaky(model_name, **kwargs):
        if len(built) == 2:
            raise LembedError(7, "model not found")
        m = FakeModel(model_name, **kwargs)
        built.append(m)
        return m

    monkeypatch.setattr(router_mod, "TextEmbedding", flaky)
    monkeypatch.setattr(router_mod, "_physical_cores", lambda: 4)
    with pytest.raises(LembedError):
        router_mod.TextEmbeddingRouter("m", sessions=3, threads=1)
    assert len(built) == 2
    assert all(m.closed for m in built)


def test_session_kwargs_and_download_progress(fake_router):
    r = fake_router(cores=4, policy="indexing", cache_size=8, batch_size=16)
    shown = [m.kwargs.get("show_download_progress") for m in FakeModel.instances]
    assert shown == [None, False, False, False]
    assert all(m.kwargs["cache_size"] == 8 for m in FakeModel.instances)
    assert all(m.kwargs["batch_size"] == 16 for m in FakeModel.instances)
    r.close()


def test_embed_stats_and_context_manager(fake_router):
    with fake_router(cores=2, policy="indexing") as r:
        out = r.embed(["a", "b", "c"], normalized=True, batch_size=2)
        assert out.shape == (3, r.dim)
        used = [m for m in FakeModel.instances if m.calls]
        assert used[0].calls == [(["a", "b", "c"], {"normalized": True, "batch_size": 2})]
        st = r.stats()
        assert st.sessions == 2
        assert st.in_flight == 0
        assert st.completed == 1
        assert "TextEmbeddingRouter" in repr(r)
    assert all(m.closed for m in FakeModel.instances)
    with pytest.raises(RouterClosedError):
        r.embed(["x"])


def test_embed_default_and_per_call_timeout(fake_router):
    r = fake_router(cores=2, policy="latency", acquire_timeout=0.05)
    gate, started = threading.Event(), threading.Event()
    holder = _hold(r._core, gate, started)
    with pytest.raises(TimeoutError):
        r.embed(["a"])
    with pytest.raises(TimeoutError):
        r.embed(["a"], timeout=0.01)
    gate.set()
    holder.join()
    assert r.stats().timeouts == 2
    r.close()


def test_embed_concurrent_threads(fake_router):
    r = fake_router(cores=4, policy="indexing")
    results = []

    def work(n):
        for _ in range(20):
            results.append(r.embed(["t"] * n).shape)

    threads = [threading.Thread(target=work, args=(1 + i % 3,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 160
    assert r.stats().in_flight == 0
    r.close()


# ---------------------------------------------------------------------------
# Integration: real model from the cache (marked `network` by the fixture)
# ---------------------------------------------------------------------------


def _min_cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((a * b).sum(axis=1).min())


@pytest.mark.parametrize("policy", ["balanced", "indexing"])
def test_router_matches_single_session(bge_small, policy):
    from libembedding import TextEmbeddingRouter

    ref = bge_small()
    texts = [f"sentence number {i} about embeddings and routers" for i in range(8)]
    try:
        expected = ref.embed(texts)
        with TextEmbeddingRouter(
            ref.model_name, policy=policy, offline=True, show_download_progress=False
        ) as router:
            assert router.dim == ref.dim
            for batch in (1, 8):
                got = router.embed(texts[:batch])
                assert got.shape == (batch, ref.dim)
                assert _min_cosine(got, expected[:batch]) >= 0.9999

            # Concurrent requests land on different sessions: same vectors.
            outs = [None] * 8

            def work(i):
                outs[i] = router.embed(texts[i : i + 1])

            threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for i, out in enumerate(outs):
                assert _min_cosine(out, expected[i : i + 1]) >= 0.9999
            assert router.stats().in_flight == 0
    finally:
        ref.close()


# ---------------------------------------------------------------------------
# asyncio (fake sessions, hermetic)
# ---------------------------------------------------------------------------


async def _spin(n):
    for _ in range(n):
        await asyncio.sleep(0)


async def _wait_async(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.0005)


async def _settle(router, timeout=5.0):
    """Wait until every session is back (no call running, nobody queued)."""
    await _wait_async(
        lambda: router.stats().in_flight == 0 and router.stats().queued == 0,
        timeout,
    )


def test_aembed_basic_and_async_context_manager(fake_router):
    async def main():
        async with fake_router(cores=2, policy="indexing") as r:
            out = await r.aembed(["a", "b"], normalized=True)
            assert out.shape == (2, r.dim)
            used = [m for m in FakeModel.instances if m.calls]
            assert used[0].calls == [(["a", "b"], {"normalized": True})]
            assert r.stats().in_flight == 0
        return r

    r = asyncio.run(main())
    assert all(m.closed for m in FakeModel.instances)
    with pytest.raises(RouterClosedError):
        asyncio.run(r.aembed(["x"]))


def test_async_error_releases_session():
    router = _SessionRouter([FakeSession()])

    def boom(session):
        raise LembedError(3, "native failure")

    async def main():
        with pytest.raises(LembedError):
            await router.arun(boom)
        await _settle(router)
        assert await router.arun(lambda s: 7) == 7

    asyncio.run(main())
    router.close()


def test_async_fifo_order():
    router = _SessionRouter([FakeSession()], max_queue=0)
    gate = threading.Event()
    order = []

    async def main():
        holder = asyncio.ensure_future(router.arun(lambda s: gate.wait(10)))
        await _wait_async(lambda: router.stats().in_flight == 1)
        tasks = []
        for i in range(24):
            tasks.append(
                asyncio.ensure_future(router.arun(lambda s, i=i: order.append(i)))
            )
            await _spin(1)  # each task reaches the queue before the next one
        assert router.stats().queued == 24
        gate.set()
        await asyncio.gather(holder, *tasks)

    asyncio.run(main())
    assert order == list(range(24))
    router.close()


def test_async_timeout_and_backpressure():
    router = _SessionRouter([FakeSession()], max_queue=1)
    gate = threading.Event()

    async def main():
        holder = asyncio.ensure_future(router.arun(lambda s: gate.wait(10)))
        await _wait_async(lambda: router.stats().in_flight == 1)

        waiter = asyncio.ensure_future(router.arun(lambda s: None, timeout=0.05))
        await _wait_async(lambda: router.stats().queued == 1)
        t0 = time.perf_counter()
        with pytest.raises(RouterBusyError):
            await router.arun(lambda s: None)
        assert time.perf_counter() - t0 < 0.5
        with pytest.raises(TimeoutError):
            await waiter
        st = router.stats()
        assert (st.timeouts, st.rejected, st.queued) == (1, 1, 0)

        gate.set()
        await holder
        await _settle(router)
        assert await router.arun(lambda s: "ok") == "ok"

    asyncio.run(main())
    router.close()


def test_async_cancel_while_waiting():
    router = _SessionRouter([FakeSession()], max_queue=0)
    gate = threading.Event()
    ran = []

    async def main():
        holder = asyncio.ensure_future(router.arun(lambda s: gate.wait(10)))
        await _wait_async(lambda: router.stats().in_flight == 1)
        waiter = asyncio.ensure_future(router.arun(lambda s: ran.append(1)))
        await _wait_async(lambda: router.stats().queued == 1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert router.stats().queued == 0
        gate.set()
        await holder
        await _settle(router)

    asyncio.run(main())
    assert ran == []
    assert len(router._free) == 1
    router.close()


def test_async_cancel_during_native_call():
    router = _SessionRouter([FakeSession()], max_queue=0)
    release_call = threading.Event()
    finished = threading.Event()

    def slow(s):
        release_call.wait(10)
        finished.set()

    async def main():
        task = asyncio.ensure_future(router.arun(slow))
        await _wait_async(lambda: router.stats().in_flight == 1)
        t0 = time.perf_counter()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Immediate: the native call is still running.
        assert time.perf_counter() - t0 < 0.5
        assert not finished.is_set()
        assert router.stats().in_flight == 1
        # The next request has to wait for the abandoned call.
        nxt = asyncio.ensure_future(router.arun(lambda s: "next"))
        await _wait_async(lambda: router.stats().queued == 1)
        release_call.set()
        assert await nxt == "next"
        await _settle(router)

    asyncio.run(main())
    assert finished.is_set()
    assert len(router._free) == 1
    router.close()


def test_async_cancellation_race_stress():
    """10 000 random cancellations / timeouts racing with hand-offs."""
    sessions = [FakeSession(0), FakeSession(1)]
    router = _SessionRouter(sessions, max_queue=0)
    rng = random.Random(20261125)
    unexpected = []
    counts = {"cancel_calls": 0, "cancelled": 0, "timeouts": 0, "done": 0}

    async def one(timeout):
        try:
            await router.arun(lambda s: s.use(), timeout=timeout)
            counts["done"] += 1
        except asyncio.CancelledError:
            counts["cancelled"] += 1
            raise
        except TimeoutError:
            counts["timeouts"] += 1
        except Exception as exc:  # noqa: BLE001 - recorded, asserted below
            unexpected.append(exc)

    async def main():
        while counts["cancel_calls"] < 10_000:
            tasks = [
                asyncio.ensure_future(
                    one(rng.random() * 0.0005 if rng.random() < 0.25 else None)
                )
                for _ in range(100)
            ]
            for t in tasks:
                await _spin(rng.randrange(0, 3))
                t.cancel()
                counts["cancel_calls"] += 1
            await asyncio.gather(*tasks, return_exceptions=True)
        await _settle(router)

    asyncio.run(main())
    assert unexpected == []
    st = router.stats()
    assert st.in_flight == 0
    assert st.queued == 0
    assert len(router._free) == len(sessions)
    assert counts["cancelled"] > 0
    # Both sessions must still be usable (nothing stuck, nothing closed).
    assert router.run(lambda s: s.use()) is None
    router.close()


def test_async_close_wakes_waiters_and_waits_for_in_flight():
    session = FakeSession()
    router = _SessionRouter([session], max_queue=0)
    gate = threading.Event()

    async def main():
        holder = asyncio.ensure_future(router.arun(lambda s: gate.wait(10)))
        await _wait_async(lambda: router.stats().in_flight == 1)
        waiters = [asyncio.ensure_future(router.arun(lambda s: None)) for _ in range(3)]
        await _wait_async(lambda: router.stats().queued == 3)

        closing = asyncio.ensure_future(router.aclose())
        results = await asyncio.gather(*waiters, return_exceptions=True)
        assert all(isinstance(r, RouterClosedError) for r in results)
        # The loop stays responsive and the busy session is not closed.
        await asyncio.sleep(0.05)
        assert not closing.done()
        assert not session.closed
        with pytest.raises(RouterClosedError):
            await router.arun(lambda s: None)

        gate.set()
        await holder
        await asyncio.wait_for(closing, 5)
        assert session.closed

    asyncio.run(main())


def test_default_loop_executor_stays_free():
    router = _SessionRouter([FakeSession(0), FakeSession(1)], max_queue=0)
    gate = threading.Event()
    threads = []

    def blocked(s):
        threads.append(threading.current_thread().name)
        gate.wait(10)

    async def main():
        loop = asyncio.get_running_loop()
        tasks = [asyncio.ensure_future(router.arun(blocked)) for _ in range(6)]
        await _wait_async(lambda: router.stats().in_flight == 2)
        await _wait_async(lambda: router.stats().queued == 4)
        # Router saturated: the loop's default executor must still answer.
        t0 = time.perf_counter()
        assert await loop.run_in_executor(None, lambda: 42) == 42
        assert await asyncio.to_thread(lambda: 43) == 43
        assert time.perf_counter() - t0 < 1.0
        gate.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert threads
    assert all(name.startswith("lembed-router") for name in threads)
    router.close()
