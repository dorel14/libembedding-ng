"""Integration tests for ImageEmbedding.

These tests require either a cached model or network access.
They are skipped if the model cannot be loaded.
"""

import os
import tempfile

import numpy as np
import pytest


def _image_model(**kwargs):
    """Create an image embedding model, skipping if download fails."""
    from libembedding import ImageEmbedding
    from libembedding.exceptions import DownloadError

    kwargs.setdefault("show_download_progress", False)
    try:
        return ImageEmbedding("Qdrant/clip-ViT-B-32-vision", **kwargs)
    except DownloadError:
        pytest.skip("image model download unavailable")


def _create_test_image(path: str) -> None:
    """Create a minimal valid PNG file for testing."""
    # Minimal valid 1x1 PNG (8 bytes IHDR + 13 bytes IDAT + 12 bytes IEND)
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE,  # 8-bit RGB
        0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54,  # IDAT chunk
        0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01,  # data
        0xE2, 0x21, 0xBC, 0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44,  # IEND
        0xAE, 0x42, 0x60, 0x82,
    ])
    with open(path, "wb") as f:
        f.write(png_data)


def test_image_list_supported_models():
    from libembedding import ImageEmbedding

    models = ImageEmbedding.list_supported_models()
    assert len(models) > 0
    for m in models:
        assert m.dim > 0
        assert m.model_code


def test_image_embed_files():
    model = _image_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _create_test_image(img_path)
        result = model.embed_files([img_path])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, model.dim)
        assert result.dtype == np.float32
    model.close()


def test_image_embed_files_multiple():
    model = _image_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(3):
            img_path = os.path.join(tmpdir, f"test_{i}.png")
            _create_test_image(img_path)
            paths.append(img_path)
        result = model.embed_files(paths)
        assert result.shape == (3, model.dim)
    model.close()


def test_image_embed_files_empty():
    model = _image_model()
    result = model.embed_files([])
    assert result.shape == (0, model.dim)
    model.close()


def test_image_embed_bytes():
    model = _image_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _create_test_image(img_path)
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        result = model.embed_bytes([img_bytes])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, model.dim)
    model.close()


def test_image_embed_bytes_empty():
    model = _image_model()
    result = model.embed_bytes([])
    assert result.shape == (0, model.dim)
    model.close()


def test_image_info():
    model = _image_model()
    info = model.info()
    assert info.dimension > 0
    assert info.max_length > 0
    assert info.batch_size > 0
    model.close()


def test_image_name():
    model = _image_model()
    assert "clip" in model.name.lower() or "vision" in model.name.lower()
    model.close()


def test_image_stats():
    model = _image_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _create_test_image(img_path)
        model.embed_files([img_path])
    stats = model.stats()
    assert stats.texts_embedded >= 1
    assert stats.batches_run >= 1
    assert stats.avg_latency_ms > 0
    model.close()


def test_image_context_manager():
    try:
        with _image_model() as model:
            assert model.dim > 0
    except pytest.skip.Exception:
        pass


def test_image_batch_size_override():
    model = _image_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        _create_test_image(img_path)
        result = model.embed_files([img_path], batch_size=1)
        assert result.shape == (1, model.dim)
    model.close()
