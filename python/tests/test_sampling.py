"""Unit tests for corpus sampling utilities."""

import pytest

from libembedding.sampling import _sample_corpus


def test_sample_corpus_smaller_than_max():
    texts = ["short", "a bit longer text", "tiny"]
    result = _sample_corpus(texts, max_size=100)
    assert result == texts


def test_sample_corpus_exact_max():
    texts = [f"text {i}" for i in range(50)]
    result = _sample_corpus(texts, max_size=50)
    assert result == texts


def test_sample_corpus_larger_than_max():
    texts = [f"text {i}" for i in range(200)]
    result = _sample_corpus(texts, max_size=50)
    assert len(result) == 50
    assert all(t in texts for t in result)


def test_sample_corpus_empty():
    result = _sample_corpus([], max_size=10)
    assert result == []


def test_sample_corpus_single():
    result = _sample_corpus(["only one"], max_size=10)
    assert result == ["only one"]


def test_sample_corpus_stratified():
    short = ["a"] * 50
    medium = [" ".join(["word"] * 20)] * 50
    long = [" ".join(["word"] * 100)] * 50
    texts = short + medium + long
    result = _sample_corpus(texts, max_size=30)
    assert len(result) == 30
    # Should contain mix of lengths
    lengths = [len(t.split()) for t in result]
    assert min(lengths) < 10
    assert max(lengths) > 50


def test_sample_corpus_no_duplicates():
    texts = [f"unique text {i}" for i in range(200)]
    result = _sample_corpus(texts, max_size=50)
    assert len(result) == len(set(result))


def test_sample_corpus_max_size_one():
    texts = [f"text {i}" for i in range(10)]
    result = _sample_corpus(texts, max_size=1)
    assert len(result) == 1
