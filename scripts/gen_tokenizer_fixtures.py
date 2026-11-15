#!/usr/bin/env python3
# libembedding - scripts/gen_tokenizer_fixtures.py
# Generates the HuggingFace `tokenizers` reference fixtures for the tokenizer
# fidelity test (tests/test_tokenizer_fidelity.cpp, roadmap T26).
#
# Auteur: David Orel
# Version: 1.14.0
#
# SPDX-License-Identifier: MIT
"""Produce reference token ids from HF ``tokenizers`` for the fidelity test.

The C++ test never imports ``tokenizers``: it compares libembedding's own
tokenizer against the JSON written here, so CI needs neither Python nor the
HF package. Only this generator does. Run it after editing the corpus or when
a registry model's tokenizer.json changes::

    pip install tokenizers huggingface_hub
    python scripts/gen_tokenizer_fixtures.py

Three fixtures are written to ``tests/data/tokenizer/``:

``hf_normalizer.json``
    Per-codepoint output of ``BertNormalizer`` (strip accents + lowercase,
    no clean_text, no CJK padding) for every codepoint whose output is not
    itself. Hermetic: needs no model.
``hf_pretokenized.json``
    For each corpus text, the words produced by the full ``BertNormalizer``
    followed by ``BertPreTokenizer``, uncased and cased. Hermetic too.
``hf_ids.json``
    For each tokenizer family of the registry, the ids HF produces for each
    corpus text (special tokens on, no truncation, no padding), with the
    SHA-256 of the tokenizer.json used. The C++ test only checks a family when
    the same tokenizer.json is present in the local model cache.

tokenizer.json files are read from the libembedding cache first
(``$LIBEMBEDDING_CACHE_DIR`` or ``~/.cache/libembedding``) and downloaded from
the Hugging Face Hub otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data" / "tokenizer"

# One representative per tokenizer family of the registry. `cache_dir` is the
# folder name libembedding's downloader uses under its cache root, `repo` the
# Hub repository the registry downloads tokenizer.json from.
FAMILIES = [
    {
        "family": "wordpiece-bert-uncased",
        "description": "WordPiece + BertNormalizer(lowercase) + BertPreTokenizer "
        "(MiniLM, bge-*-en, snowflake, gte-en, nomic, mxbai, SPLADE)",
        "cache_dir": "models--Xenova-bge-small-en-v1.5",
        "repo": "Xenova/bge-small-en-v1.5",
    },
    {
        "family": "wordpiece-bert-chinese",
        "description": "WordPiece + BertNormalizer(lowercase) + BertPreTokenizer, "
        "Chinese vocabulary (bge-*-zh)",
        "cache_dir": "models--Xenova-bge-small-zh-v1.5",
        "repo": "Xenova/bge-small-zh-v1.5",
    },
    {
        "family": "wordpiece-mpnet",
        "description": "WordPiece + BertNormalizer(lowercase) + BertPreTokenizer, "
        "<s>/</s> specials (all-mpnet-base-v2)",
        "cache_dir": "models--Xenova-all-mpnet-base-v2",
        "repo": "Xenova/all-mpnet-base-v2",
    },
    {
        "family": "unigram-xlmr",
        "description": "Unigram (SentencePiece) + Precompiled nmt_nfkc + Metaspace "
        "(multilingual-e5, paraphrase-multilingual, bge-m3, bge-reranker)",
        "cache_dir": "models--intfloat-multilingual-e5-small",
        "repo": "intfloat/multilingual-e5-small",
    },
    {
        "family": "bpe-bytelevel-clip",
        "description": "BPE + NFC/Lowercase + ByteLevel (CLIP text)",
        "cache_dir": "models--Qdrant-clip-ViT-B-32-text",
        "repo": "Qdrant/clip-ViT-B-32-text",
    },
    {
        "family": "bpe-bytelevel-modernbert",
        "description": "BPE + ByteLevel, GPT-NeoX style (modernbert-embed-large)",
        "cache_dir": "models--lightonai-modernbert-embed-large",
        "repo": "lightonai/modernbert-embed-large",
    },
]

# Hangul syllables decompose algorithmically into conjoining jamo; storing all
# 11 172 would dominate the fixture, so only every HANGUL_STEP-th one is kept.
HANGUL_FIRST, HANGUL_LAST, HANGUL_STEP = 0xAC00, 0xD7A3, 97


def cache_root() -> Path:
    env = os.environ.get("LIBEMBEDDING_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "libembedding"


def locate_tokenizer_json(fam: dict) -> Path | None:
    local = cache_root() / fam["cache_dir"] / "tokenizer.json"
    if local.is_file():
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(fam["repo"], "tokenizer.json"))
    except Exception as exc:  # noqa: BLE001 - report and keep going
        print(f"  skip {fam['family']}: {exc}", file=sys.stderr)
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gen_normalizer(out: Path) -> None:
    from tokenizers.normalizers import BertNormalizer

    norm = BertNormalizer(
        clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=True
    )
    entries: dict[str, str] = {}
    for cp in range(0x80, 0x30000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if HANGUL_FIRST <= cp <= HANGUL_LAST and (cp - HANGUL_FIRST) % HANGUL_STEP:
            continue
        s = chr(cp)
        r = norm.normalize_str(s)
        if r != s:
            entries[f"{cp:04X}"] = r
    doc = {
        "_comment": "Generated by scripts/gen_tokenizer_fixtures.py - do not edit.",
        "normalizer": "BertNormalizer(clean_text=False, handle_chinese_chars=False, "
        "strip_accents=True, lowercase=True)",
        "range": ["0080", "2FFFF"],
        "hangul_step": HANGUL_STEP,
        "changed": entries,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n", encoding="utf-8", newline="\n")
    print(f"  {out.name}: {len(entries)} codepoints change")


def gen_pretokenized(texts: list[dict], out: Path) -> None:
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer

    pre = BertPreTokenizer()
    variants = {"uncased": BertNormalizer(lowercase=True), "cased": BertNormalizer(lowercase=False)}
    doc: dict = {"_comment": "Generated by scripts/gen_tokenizer_fixtures.py - do not edit."}
    for name, norm in variants.items():
        doc[name] = {
            t["id"]: [w for w, _ in pre.pre_tokenize_str(norm.normalize_str(t["text"]))] for t in texts
        }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"  {out.name}: {len(texts)} texts x {len(variants)} variants")


def gen_ids(texts: list[dict], out: Path) -> None:
    import tokenizers
    from tokenizers import Tokenizer

    families = []
    for fam in FAMILIES:
        path = locate_tokenizer_json(fam)
        if path is None:
            continue
        tok = Tokenizer.from_file(str(path))
        tok.no_truncation()
        tok.no_padding()
        ids = {t["id"]: tok.encode(t["text"], add_special_tokens=True).ids for t in texts}
        families.append({**fam, "tokenizer_sha256": sha256(path), "ids": ids})
        print(f"  {fam['family']}: {path}")
    # Hand-rolled layout, one line per text: json.dumps(indent=...) would put
    # every id on its own line and make a regeneration diff unreadable.
    def dump(v: object) -> str:
        return json.dumps(v, ensure_ascii=False)

    parts = [
        "{",
        f' "_comment": {dump("Generated by scripts/gen_tokenizer_fixtures.py - do not edit.")},',
        f' "tokenizers_version": {dump(tokenizers.__version__)},',
        ' "families": [',
    ]
    for fi, fam in enumerate(families):
        parts.append("  {")
        for key in ("family", "description", "cache_dir", "repo", "tokenizer_sha256"):
            parts.append(f"   {dump(key)}: {dump(fam[key])},")
        parts.append('   "ids": {')
        items = list(fam["ids"].items())
        for ti, (tid, ids) in enumerate(items):
            sep = "," if ti + 1 < len(items) else ""
            parts.append(f"    {dump(tid)}: {json.dumps(ids, separators=(',', ':'))}{sep}")
        parts.append("   }")
        parts.append("  }" + ("," if fi + 1 < len(families) else ""))
    parts += [" ]", "}"]
    out.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("error: `pip install tokenizers huggingface_hub` first", file=sys.stderr)
        return 1
    corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    texts = corpus["texts"]
    print(f"corpus: {len(texts)} texts")
    gen_normalizer(DATA_DIR / "hf_normalizer.json")
    gen_pretokenized(texts, DATA_DIR / "hf_pretokenized.json")
    gen_ids(texts, DATA_DIR / "hf_ids.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
