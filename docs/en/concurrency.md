---
nav_exclude: true
---

# Concurrent requests: `TextEmbeddingRouter`

> **Français:** [Requêtes concurrentes](../concurrency.html)

A native context runs **one call at a time** (`call_mutex`). A server that embeds text from several threads or `asyncio` tasks therefore needs several sessions and a scheduler. `TextEmbeddingRouter` provides both: N `TextEmbedding` sessions, a bounded strict-FIFO queue, a wait timeout, presets derived from measurements, and an `asyncio` API.

Python only (no C API), for **dense text** (ONNX and GGUF), from 1.17.

## When to use it

| Need | Tool |
|---|---|
| Many short requests (1 to 8 texts) in parallel: web server, API | **`TextEmbeddingRouter`** |
| One large batch to spread over several cores | `TextEmbeddingPool` (unchanged) |
| A single caller | `TextEmbedding` |

`TextEmbeddingPool` splits *one* call into chunks. The router keeps each call whole and runs several at the same time.

## Quick start

```python
from libembedding import TextEmbeddingRouter

with TextEmbeddingRouter("BAAI/bge-small-en-v1.5") as router:
    vectors = router.embed(["a query"])          # from any thread
    print(vectors.shape, router.sessions, router.threads)
```

From `asyncio`:

```python
import asyncio
from libembedding import TextEmbeddingRouter

async def main():
    async with TextEmbeddingRouter("BAAI/bge-small-en-v1.5") as router:
        vectors = await router.aembed(["a query"])

asyncio.run(main())
```

## Parameters

| Parameter | Default | Role |
|---|---|---|
| `model_name` | `BAAI/bge-small-en-v1.5` | As `TextEmbedding` |
| `policy` | `"latency"` | Preset (see below): `"latency"`, `"balanced"`, `"indexing"` |
| `sessions` | `None` | Explicit session count (disables `policy`) |
| `threads` | `None` | Intra-op threads per session |
| `max_queue` | `64` | Maximum waiters; the next request raises `RouterBusyError` at once. `0` = unbounded |
| `acquire_timeout` | `None` | Maximum wait for a session in seconds (`TimeoutError`); `None` = forever |
| `**session_kwargs` | | Passed to every `TextEmbedding` (`batch_size`, `cache_size`, `quantization`, `backend`...) |

`embed(texts, *, timeout=None, **kwargs)` and `aembed(...)` accept `batch_size`, `dtype` and `normalized` like `TextEmbedding.embed`. `timeout` overrides `acquire_timeout` for the call and only covers **waiting for a session**, not the native call.

## Presets

P is the number of **physical cores** (`detect_hardware().physical_cores`; falling back to half the logical cores). Sessions are capped at 8.

| Policy | ONNX (sessions x threads) | Use |
|---|---|---|
| `latency` (default) | 1 x P | Lightly loaded server, lowest latency |
| `balanced` | 2 x floor(P/2) | Loaded server |
| `indexing` | min(P, 8) x 1 | Batch indexing, maximum throughput |

For **GGUF** models each session uses 1 thread, and the session count is 1, ceil(P/2) or min(P, 8) depending on the preset.

### What was measured

T32 benchmark: i7-1065G7 (4 physical cores), ONNX Runtime 1.31.0, bge-small and MiniLM-L6 FP32, 3 interleaved rounds. Reference is 1x4 (one session, 4 threads):

| Configuration | Throughput at concurrency >= 4 | p95, batch 1, bge | RSS (bge) |
|---|---|---|---|
| 1x4 (`latency`) | reference | 9 ms | 243 MB |
| 2x2 (`balanced`) | +25 to 27 % (batch 1) | 13 to 15 ms | 382 MB |
| 4x1 (`indexing`) | +50 to 70 % (batch 1), +21 to 34 % (batch 8) | 24 ms | 658 MB |

- 1x8 (more threads than physical cores) **loses 25 to 30 %**: this is why presets never exceed P.
- Sessions produce identical vectors (difference 0).
- Memory grows roughly linearly with the session count (ORT does not share weights).

> **Limits of these numbers.** One machine, 4 cores. Values for P != 4, ARM and the GGUF presets are **assumed, not measured**. Use `sessions=` / `threads=` to force your own setup, and measure on your hardware.

### Thread budget

If `sessions x threads` exceeds P, the constructor emits a `RuntimeWarning` (not an error). Presets never exceed it.

## Queueing and backpressure

- **Strict FIFO**: a released session is handed **directly** to the oldest waiter and never goes back to the free list. A thread returning its session cannot take it back before the thread that was waiting (no starvation).
- **Bounded**: with `max_queue` requests already waiting, the next one raises `RouterBusyError` immediately, without blocking.
- **Timeout**: the standard `TimeoutError` when no session frees up in time. No session is lost in that case.
- **Native errors** (`LembedError`): propagated unchanged, the session is returned.

`RouterBusyError` and `RouterClosedError` subclass `LembedError`. For an HTTP server: `RouterBusyError` -> 503, `TimeoutError` -> 504.

## `asyncio`

`aembed` waits for a session **in the loop** (no parked thread, FIFO preserved), then runs the native call on a `ThreadPoolExecutor` **dedicated to the router** (never the loop's default executor, which stays free even when the router is saturated). Works on Python 3.9+.

### Cancellation: what is possible

| When | Behaviour |
|---|---|
| Waiting for a session | The request is cleanly removed from the queue. If a session was being handed to it at that very moment, it is returned and goes to the next waiter. |
| **During the native call** | `CancelledError` is raised **immediately**, but the native call **cannot be interrupted**: it finishes in the executor, its result is discarded, and only then is the session returned. |

Consequence: a cancellation does not free the session right away. **Bound `batch_size` and the size of `texts` on the server side**, or an abandoned large batch ties up a session until it completes. The `timeout` does not cover the native call either.

## Closing

`close(wait=True)` (or `await aclose()`):

1. rejects new requests (`RouterClosedError`);
2. wakes queued requests with `RouterClosedError`;
3. waits for in-flight calls;
4. closes the sessions. A busy session is **never** closed.

Inside a coroutine use `aclose()` (or `async with`): `close()` blocks the thread. `close(wait=False)` returns at once; the last call to finish closes the remaining sessions.

## Observability

`router.stats()` returns a `RouterStats`: `sessions`, `in_flight`, `queued`, `completed`, `rejected`, `timeouts`, `max_wait_ms`.

## Other classes: `asyncio.to_thread`

The router only covers dense text. For `Reranker`, `SparseTextEmbedding` or `MultiVectorEmbedding`, the workaround is:

```python
scores = await asyncio.to_thread(reranker.rerank, query, documents)
```

Limits: the default executor is shared with the whole application; no strict FIFO, no backpressure, no timeout; and one call at a time per instance (the native context is locked). For parallelism, create several instances and manage the queue yourself.

## Example: embedding provider for a search engine (Whoosh-NG style)

```python
import asyncio
from fastapi import FastAPI, HTTPException
from libembedding import RouterBusyError, TextEmbeddingRouter

router = TextEmbeddingRouter(
    "BAAI/bge-small-en-v1.5",
    policy="balanced",
    max_queue=64,
    acquire_timeout=2.0,
    batch_size=32,          # bounded: a cancelled request still finishes its native call
)

class LibembeddingProvider:
    """EmbeddingProvider protocol: embed, embed_batch, dimension."""

    @property
    def dimension(self) -> int:
        return router.dim

    def embed(self, text: str):
        return router.embed([text])[0]

    def embed_batch(self, texts: list[str]):
        return router.embed(texts)

    async def aembed_batch(self, texts: list[str]):
        return await router.aembed(texts)

app = FastAPI()
provider = LibembeddingProvider()

@app.post("/embed")
async def embed(texts: list[str]):
    try:
        vectors = await provider.aembed_batch(texts[:64])
    except RouterBusyError:
        raise HTTPException(503, "busy")
    except TimeoutError:
        raise HTTPException(504, "timeout")
    return {"dim": router.dim, "vectors": vectors.tolist()}

@app.on_event("shutdown")
async def shutdown():
    await router.aclose()
```

For offline indexing, create a second instance with `policy="indexing"` for the duration of the job: memory doubles.

## Caveats

- **LRU cache**: `cache_size` is **per session**. The cache is not shared; the hit rate is divided by the session count.
- **Memory**: `indexing` costs about 2.7 times the memory of `latency` (bge-small, T32 measurement).
- **Containers and hybrid CPUs**: core detection ignores cgroup quotas. Force `sessions=` / `threads=`.
- **Context after a native error**: assumed healthy (the lock is released, no partial state); not verified for every error kind.
