"""Corpus sampling utilities for autotune benchmarking."""

from __future__ import annotations

import math
import random


def _sample_corpus(texts: list[str], max_size: int = 100) -> list[str]:
    """Sample a representative subset of texts for autotune benchmarking.

    Uses stratified sampling by text length to ensure the sample includes
    a mix of short, medium, and long texts.

    Args:
        texts: Full corpus (can be millions of texts)
        max_size: Maximum number of texts to sample (default 100)

    Returns:
        Sampled texts representative of the corpus distribution
    """
    n = len(texts)
    if n <= max_size:
        return texts

    n_buckets = 10
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(n_buckets)]

    for index, text in enumerate(texts):
        word_count = len(text.split())
        if word_count <= 0:
            bucket_idx = 0
        else:
            log_count = math.log10(word_count)
            bucket_idx = min(int(log_count * 2.5), n_buckets - 1)
        buckets[bucket_idx].append((index, text))

    sampled = []
    sampled_indices = set()
    per_bucket = max(1, max_size // n_buckets)

    for bucket in buckets:
        if not bucket:
            continue
        if len(bucket) <= per_bucket:
            for index, text in bucket:
                if index not in sampled_indices:
                    sampled.append(text)
                    sampled_indices.add(index)
        else:
            step = len(bucket) / per_bucket
            for i in range(per_bucket):
                index, text = bucket[int(i * step)]
                if index not in sampled_indices:
                    sampled.append(text)
                    sampled_indices.add(index)

    if len(sampled) < max_size:
        remaining_indices = [i for i in range(n) if i not in sampled_indices]
        for index in random.sample(
            remaining_indices, min(max_size - len(sampled), len(remaining_indices))
        ):
            sampled.append(texts[index])
            sampled_indices.add(index)

    return sampled[:max_size]
