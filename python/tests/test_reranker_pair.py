"""Reranker fidelity: sequence-pair encoding vs onnxruntime + HF tokenizers.

The ONNX reranker used to tokenise the literal text ``query [SEP] doc`` as ONE
sequence (no pair template, no segment ids). The reference below encodes a real
pair through ``tokenizers`` (tokenizer.json post_processor) and runs the same
ONNX file with onnxruntime, so the scores must agree.

Author: David Orel
SPDX-License-Identifier: MIT
"""

from pathlib import Path

import numpy as np
import pytest

CACHE = Path.home() / ".cache" / "libembedding"

QUERY = "how do I reset my password"
DOCS = [
    "To reset your password, open Settings, choose Security and click Reset password.",
    "The quick brown fox jumps over the lazy dog.",
    "Password resets require access to the e-mail address attached to the account.",
    "Our office is closed on public holidays and on the first Monday of each month.",
    "Two-factor authentication adds a second step when you sign in to your account.",
    "Recipe: mix flour, sugar and eggs, then bake for forty minutes at 180 degrees.",
    "If you forgot your login credentials, use the recovery link sent by e-mail. " * 20,
    "Quarterly revenue grew by twelve percent, driven by subscription renewals.",
]


def _reference_scores(model_dir: Path, onnx_name: str, *, literal: bool = False) -> np.ndarray:
    """HF reference scores; literal=True is the old buggy single-sequence input."""
    ort = pytest.importorskip("onnxruntime")
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=512)
    sess = ort.InferenceSession(
        str(model_dir / "onnx" / onnx_name), providers=["CPUExecutionProvider"]
    )
    wanted = {i.name for i in sess.get_inputs()}
    scores = []
    for doc in DOCS:
        enc = tok.encode(QUERY + " [SEP] " + doc) if literal else tok.encode(QUERY, doc)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        scores.append(float(sess.run(None, {k: v for k, v in feed.items() if k in wanted})[0].ravel()[0]))
    return np.array(scores)


@pytest.mark.network
@pytest.mark.parametrize(
    "repo,onnx_name,quant",
    [("jinaai/jina-reranker-v1-turbo-en", "model.onnx", "none")],
)
def test_reranker_matches_pair_reference(repo, onnx_name, quant):
    from libembedding import Reranker

    model_dir = CACHE / ("models--" + repo.replace("/", "-"))
    if not (model_dir / "onnx" / onnx_name).exists():
        pytest.skip(f"{repo} not cached")
    try:
        reranker = Reranker(repo, offline=True, show_download_progress=False, quantization=quant)
    except Exception as exc:  # noqa: BLE001 - any load failure means "not available"
        pytest.skip(f"{repo} unavailable: {exc}")
    try:
        ours = np.zeros(len(DOCS))
        for r in reranker.rerank(QUERY, DOCS):
            ours[r.index] = r.score
    finally:
        reranker.close()

    ref = _reference_scores(model_dir, onnx_name)
    bad = _reference_scores(model_dir, onnx_name, literal=True)

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    corr, corr_bad, corr_refs = pearson(ours, ref), pearson(ours, bad), pearson(ref, bad)
    print(
        f"\nPearson ours~pair={corr:.6f} ours~literal[SEP]={corr_bad:.6f} "
        f"pair~literal={corr_refs:.6f} max|pair-literal|={np.abs(ref - bad).max():.4f}"
    )
    # Negative control: the two references must differ, else the test proves nothing.
    assert corr_refs < 0.99 or np.abs(ref - bad).max() > 0.05, "test cannot discriminate"
    assert corr >= 0.999, f"Pearson {corr:.4f} vs HF pair reference"
    assert list(np.argsort(-ours)) == list(np.argsort(-ref)), "ranking differs"
