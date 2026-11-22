# SPDX-License-Identifier: MIT
# Auteur: David Orel
# Version: 1.16.0
"""Generates answerai_small_golden.json: reference ColBERT vectors from pylate.

Not run in CI (needs torch + pylate + the PyTorch weights). The fixture it
writes is committed; tests/test_multivector_fidelity.cpp compares the C
runtime against it.

    pip install pylate
    python gen_golden.py [model_id] [output.json]
"""
import json
import sys

import numpy as np
from pylate import models

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "lightonai/answerai-colbert-small-v1"
OUT = sys.argv[2] if len(sys.argv) > 2 else "answerai_small_golden.json"

QUERIES = [
    "what is the capital of France?",
    "how do bananas grow",
    "Who wrote Hamlet?",
]
DOCS = [
    "Paris is the capital of France.",
    "Bananas grow in tropical climates, (mostly) near the equator.",
    "Hamlet, a tragedy, was written by William Shakespeare!",
]


def vectors(arrs):
    # 3 decimals: the cosine error this adds is ~1e-5, far below the 0.999 target.
    # Rounded in float64, otherwise tolist() prints float32 noise digits.
    return [np.round(np.asarray(a, dtype=np.float64), 3).tolist() for a in arrs]


model = models.ColBERT(model_name_or_path=MODEL_ID)
model.eval()
q = model.encode(QUERIES, is_query=True, show_progress_bar=False)
d = model.encode(DOCS, is_query=False, show_progress_bar=False)

out = {
    "model": MODEL_ID,
    "reference": "pylate",
    "queries": [{"text": t, "vectors": v} for t, v in zip(QUERIES, vectors(q))],
    "docs": [{"text": t, "vectors": v} for t, v in zip(DOCS, vectors(d))],
}
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, separators=(",", ":"))
print({"queries": [len(x) for x in q], "docs": [len(x) for x in d], "dim": len(q[0][0])})
