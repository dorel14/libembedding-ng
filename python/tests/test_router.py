"""Tests for the session router (core scheduling, hermetic).

The core tests use fake sessions driven by ``threading.Event``: no model, no
network.

Auteur: David Orel

SPDX-License-Identifier: MIT
"""

import threading
import time

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
