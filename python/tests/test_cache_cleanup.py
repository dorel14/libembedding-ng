# SPDX-License-Identifier: MIT
"""cleanup_model_cache / cleanup_model_cache_except on a fake cache, no network.

The fake cache follows the downloader's layout: models-- + repository with
'/' replaced by '-'. BAAI/bge-small-en-v1.5 is downloaded from several
repositories (Qdrant/..., Xenova/...); keeping it by its usual name must keep
all of them.
"""

from pathlib import Path

from libembedding import cleanup_model_cache, cleanup_model_cache_except

BGE_QDRANT = "models--Qdrant-bge-small-en-v1.5-onnx-Q"
BGE_XENOVA = "models--Xenova-bge-small-en-v1.5"
OTHER = "models--org-other"


def _fake_cache(root: Path) -> Path:
    for name, size in ((BGE_QDRANT, 100), (BGE_XENOVA, 50), (OTHER, 30)):
        (root / name).mkdir(parents=True)
        (root / name / "model.onnx").write_bytes(b"x" * size)
    (root / "tune_cache.json").write_text("{}")
    (root / "notes").mkdir()
    return root


def _names(root: Path) -> set[str]:
    return {p.name for p in root.iterdir()}


def test_keep_by_usual_name_keeps_every_download(tmp_path):
    cache = _fake_cache(tmp_path)
    result = cleanup_model_cache(["BAAI/bge-small-en-v1.5"], cache_dir=str(cache))
    assert result["deleted_count"] == 1
    assert result["freed_bytes"] == 30
    assert _names(cache) == {BGE_QDRANT, BGE_XENOVA, "tune_cache.json", "notes"}


def test_dry_run_reports_without_deleting(tmp_path):
    cache = _fake_cache(tmp_path)
    result = cleanup_model_cache(["org/other"], cache_dir=str(cache), dry_run=True)
    assert result["deleted_count"] == 2
    assert result["freed_bytes"] == 150
    assert _names(cache) == {BGE_QDRANT, BGE_XENOVA, OTHER, "tune_cache.json", "notes"}


def test_none_keeps_everything(tmp_path):
    # Documented as "all models are kept"; it used to delete every model.
    cache = _fake_cache(tmp_path)
    result = cleanup_model_cache(None, cache_dir=str(cache))
    assert result == {"deleted_count": 0, "freed_bytes": 0, "details": []}
    assert OTHER in _names(cache) and BGE_QDRANT in _names(cache)


def test_empty_list_deletes_every_model(tmp_path):
    cache = _fake_cache(tmp_path)
    result = cleanup_model_cache([], cache_dir=str(cache))
    assert result["deleted_count"] == 3
    assert _names(cache) == {"tune_cache.json", "notes"}


def test_except_variant(tmp_path):
    cache = _fake_cache(tmp_path)
    result = cleanup_model_cache_except("BAAI/bge-small-en-v1.5", cache_dir=str(cache))
    assert result["deleted_count"] == 1
    assert OTHER not in _names(cache)
    assert BGE_QDRANT in _names(cache) and BGE_XENOVA in _names(cache)


def test_missing_cache_directory(tmp_path):
    result = cleanup_model_cache(["org/other"], cache_dir=str(tmp_path / "absent"))
    assert result["deleted_count"] == 0 and result["freed_bytes"] == 0
