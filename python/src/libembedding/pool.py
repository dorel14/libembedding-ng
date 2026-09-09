"""Multi-worker text embedding pool for maximum throughput."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .autotune import _do_autotune
from .text_embedding import TextEmbedding


class TextEmbeddingPool:
    """Multi-worker text embedding pool for maximum throughput.

    Creates multiple TextEmbedding instances and distributes work across
    them using threads.

    Args:
        model_name: Same as TextEmbedding.
        workers: Number of worker sessions (0 = auto with autotune).
        threads_per_worker: Threads per worker (default 1).
        batch_size: Internal batch size per worker.
        provider: Same as TextEmbedding.
        offline: Same as TextEmbedding.
        autotune: If True, auto-tune workers/threads/batch.
        autotune_texts: Optional list of texts for autotune benchmark.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        workers: int = 0,
        threads_per_worker: int = 1,
        batch_size: int = 256,
        provider: str = "cpu",
        offline: bool = False,
        show_download_progress: bool = True,
        cache_dir: str | None = None,
        max_length: int = 0,
        dim: int = 0,
        pooling: str = "mean",
        autotune: bool = False,
        autotune_texts: list[str] | None = None,
        autotune_max_samples: int = 100,
    ):
        if autotune:
            tuned = _do_autotune(model_name, provider,
                                 texts=autotune_texts,
                                 max_sample_size=autotune_max_samples)
            n_workers = tuned.workers
            threads_per_worker = tuned.threads
            batch_size = tuned.batch_size
            print(f"Autotune: {n_workers} workers x {threads_per_worker} threads, batch={batch_size}")
        else:
            n_workers = workers
            if n_workers <= 0:
                n_workers = min(os.cpu_count() or 4, 8)

        self._n_workers = n_workers
        self._dim = 0

        self._workers: list[TextEmbedding] = []
        for i in range(n_workers):
            worker = TextEmbedding(
                model_name,
                provider=provider,
                threads=threads_per_worker,
                batch_size=batch_size,
                offline=offline,
                show_download_progress=(show_download_progress and i == 0),
                cache_dir=cache_dir,
                max_length=max_length,
                dim=dim,
                pooling=pooling,
            )
            self._workers.append(worker)

        self._dim = self._workers[0].dim

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    @property
    def num_workers(self) -> int:
        """Number of worker sessions."""
        return self._n_workers

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using all workers."""
        n = len(texts)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        per_worker = (n + self._n_workers - 1) // self._n_workers
        chunks = []
        for i in range(self._n_workers):
            start = i * per_worker
            end = min(start + per_worker, n)
            if start < end:
                chunks.append(texts[start:end])

        results: list[np.ndarray] = [None] * len(chunks)

        def embed_chunk(idx: int, chunk: list[str]) -> None:
            results[idx] = self._workers[idx].embed(chunk)

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            futures = {
                executor.submit(embed_chunk, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                future.result()

        return np.concatenate(results, axis=0)

    def close(self) -> None:
        """Release all worker sessions."""
        for worker in self._workers:
            worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"TextEmbeddingPool(workers={self._n_workers}, dim={self._dim})"
