"""Unit tests for libembedding similarity functions."""

import numpy as np
import pytest


def test_cosine_similarity_identical():
    from libembedding.similarity import cosine_similarity

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    from libembedding.similarity import cosine_similarity

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 0.0) < 1e-5


def test_cosine_similarity_opposite():
    from libembedding.similarity import cosine_similarity

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-5


def test_cosine_similarity_shape_mismatch():
    from libembedding.similarity import cosine_similarity

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_cosine_similarity_2d_raises():
    from libembedding.similarity import cosine_similarity

    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_dot_product_basic():
    from libembedding.similarity import dot_product

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    assert abs(dot_product(a, b) - 32.0) < 1e-5


def test_dot_product_negative():
    from libembedding.similarity import dot_product

    a = np.array([1.0, -1.0], dtype=np.float32)
    b = np.array([1.0, -1.0], dtype=np.float32)
    assert abs(dot_product(a, b) - 2.0) < 1e-5


def test_dot_product_shape_mismatch():
    from libembedding.similarity import dot_product

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        dot_product(a, b)


def test_euclidean_distance_identical():
    from libembedding.similarity import euclidean_distance

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(euclidean_distance(a, b) - 0.0) < 1e-5


def test_euclidean_distance_unit():
    from libembedding.similarity import euclidean_distance

    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(euclidean_distance(a, b) - 1.0) < 1e-5


def test_euclidean_distance_diagonal():
    from libembedding.similarity import euclidean_distance

    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    assert abs(euclidean_distance(a, b) - np.sqrt(2)) < 1e-5


def test_euclidean_distance_shape_mismatch():
    from libembedding.similarity import euclidean_distance

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError):
        euclidean_distance(a, b)


def test_similarity_non_contiguous_raises_or_converts():
    from libembedding.similarity import cosine_similarity

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).T
    b = np.array([1.0, 2.0], dtype=np.float32)
    # np.ascontiguousarray should handle this, but shape mismatch takes precedence
    with pytest.raises(ValueError):
        cosine_similarity(a, b)
