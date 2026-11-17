"""Per-model reference validation (T25).

Auteur: David Orel
Version: 1.14.0

SPDX-License-Identifier: MIT

Every registry model that is in the local cache is compared with the SAME ONNX
file run by Python onnxruntime on HuggingFace ``tokenizers`` ids (see
``model_reference.py``). Nothing is downloaded: a model that is not cached, or a
machine without onnxruntime/tokenizers, skips.

The registry has known tokenizer gaps (Unigram/SentencePiece, byte-level BPE,
non-Latin normalisation, words over 100 characters), so the assertions encode
a baseline, not perfection:

* WordPiece models must match on English (``en_core_min`` >= 0.999);
* no model may fall below its committed baseline (``data/model_baselines.json``)
  by more than ``SLACK``;
* a cached model absent from the baseline fails: measure it and add it.

The committed ``data/reference_embeddings.json`` lets three light models be
checked without onnxruntime.
"""

import json
from pathlib import Path

import model_reference as ref
import pytest

pytestmark = pytest.mark.network

DATA = Path(__file__).resolve().parent / "data"
BASELINES = json.loads((DATA / "model_baselines.json").read_text(encoding="utf-8"))["models"]
FIXTURES = json.loads((DATA / "reference_embeddings.json").read_text(encoding="utf-8"))["models"]

SLACK = 0.02  # allowed drop of a baseline statistic
EN_CORE_FLOOR = {ref.WORDPIECE: 0.999}  # families whose English must be exact
MAX_MB = 560  # same bound as the report


def _entry(kind: str, name: str, model_file: str):
    for idx, info in ref.registry_entries(kind):
        if info.model_name == name and info.model_file == model_file:
            return idx, info
    pytest.fail(f"{kind} {name} {model_file} is no longer in the registry")


def _cached(kind: str, idx: int, info) -> Path:
    directory = ref.model_dir(kind, idx)
    if directory is None or not (directory / info.model_file).is_file():
        pytest.skip(f"{info.model_name} ({info.model_file}) is not in the local cache")
    return directory


def _need_reference():
    if not ref.have_reference_deps():
        pytest.skip("onnxruntime and tokenizers are required for the reference")


def _keys(kind: str):
    return [k for k in BASELINES if k.startswith(kind + "|")]


@pytest.mark.parametrize("key", _keys("text"))
def test_dense_matches_reference(key):
    _need_reference()
    _, name, model_file = key.split("|")
    idx, info = _entry("text", name, model_file)
    directory = _cached("text", idx, info)
    base = BASELINES[key]
    assert ref.tokenizer_family(directory / "tokenizer.json") == base["family"]

    res = ref.measure_dense(idx, info, directory, ref.load_corpus())
    core = ref.english_core(res["per_text"])
    floor = EN_CORE_FLOOR.get(base["family"])
    if floor is not None:
        assert core >= floor, f"{name}: English cosine {core:.4f} < {floor}"
    assert res["all"]["min"] >= base["min"] - SLACK
    assert res["all"]["mean"] >= base["mean"] - SLACK


@pytest.mark.parametrize("key", _keys("sparse"))
def test_sparse_matches_reference(key):
    _need_reference()
    _, name, model_file = key.split("|")
    idx, info = _entry("sparse", name, model_file)
    directory = _cached("sparse", idx, info)
    base = BASELINES[key]
    res = ref.measure_sparse(idx, info, directory, ref.load_corpus())
    floor = EN_CORE_FLOOR.get(base["family"])
    if floor is not None:
        assert ref.english_core(res["per_text"]) >= floor
    assert res["all"]["mean"] >= base["mean"] - SLACK


@pytest.mark.parametrize("key", _keys("reranker"))
def test_reranker_matches_reference(key):
    _need_reference()
    _, name, model_file = key.split("|")
    idx, info = _entry("reranker", name, model_file)
    directory = _cached("reranker", idx, info)
    res = ref.measure_reranker(idx, info, directory)
    assert res["pearson_vs_logit"] >= BASELINES[key]["pearson"] - SLACK


def test_cached_models_have_a_baseline():
    """A cached model with no baseline has never been validated."""
    missing = []
    for kind in ("text", "sparse", "reranker"):
        for idx, info in ref.registry_entries(kind):
            if info.model_file.endswith(".gguf") or info.download_size_mb > MAX_MB:
                continue
            directory = ref.model_dir(kind, idx)
            if directory is None or not (directory / info.model_file).is_file():
                continue
            if f"{kind}|{info.model_name}|{info.model_file}" not in BASELINES:
                missing.append(f"{kind}|{info.model_name}|{info.model_file}")
    assert not missing, (
        "cached but never validated; run python/tests/model_reference.py --json "
        f"then --baselines: {missing}"
    )


def _fixture_model(key: str):
    name, quant = key.split("|")
    for idx, info in ref.registry_entries("text"):
        if info.model_name == name and info.quantization == quant:
            return idx, info
    pytest.fail(f"{key} is no longer in the registry")


def _embedder(info):
    from libembedding import TextEmbedding

    return TextEmbedding(
        info.model_name,
        quantization=info.quantization,
        max_length=min(info.max_tokens, ref.MAX_LEN_CAP),
        show_download_progress=False,
    )


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_committed_reference_vectors(key):
    """Compares with vectors committed from onnxruntime + HF tokenizers.

    Needs the model cache but neither onnxruntime nor tokenizers.
    """
    idx, info = _fixture_model(key)
    _cached("text", idx, info)
    fixture = FIXTURES[key]
    ids = sorted(fixture["vectors"])
    texts = {t["id"]: t["text"] for t in ref.load_corpus()}
    emb = _embedder(info)
    try:
        ours = emb.embed([texts[i] for i in ids])
    finally:
        emb.close()
    for tid, vec in zip(ids, ours):
        assert ref.cosine(fixture["vectors"][tid], vec) >= 0.9999, f"{key} {tid}"


@pytest.mark.parametrize("key", sorted(FIXTURES))
def test_deterministic_and_batch_equals_single(key):
    idx, info = _fixture_model(key)
    _cached("text", idx, info)
    texts = [t["text"] for t in ref.load_corpus() if t["script"] == "latin-ascii"]
    texts.append(ref.long_text())
    emb = _embedder(info)
    try:
        first = emb.embed(texts)
        second = emb.embed(texts)
        singles = [emb.embed([t])[0] for t in texts]
    finally:
        emb.close()
    for i, text in enumerate(texts):
        assert (first[i] == second[i]).all(), f"two runs differ on text {i}"
        assert ref.cosine(first[i], singles[i]) >= 0.9999, (
            f"batch != single on {text[:30]!r}"
        )
