"""Module de génération des fichiers llm.txt et llm_full.txt.

Ce module génère deux fichiers à la racine du dépôt, consommés par les LLM
et les robots d'indexation :

- ``llm.txt`` — un index concis décrivant le projet, avec des liens vers
  chaque page de documentation (en anglais et en français) et des extraits
  de code clés.
- ``llm_full.txt`` — le contenu complet de tous les documents Markdown de
  la documentation (anglais + français), nettoyé du *front-matter* YAML,
  auxquels s'ajoutent le README et les exemples de code.

Conventions suivies (llm.txt standard, https://llm-txt.org/) :
  * ``llm.txt``  — résumé et table des matières
  * ``llm_full.txt`` — corpus complet

Auteur: David Orel
Version: 1.0.0
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# ── Configuration ---------------------------------------------------------

# Repository root (parent de scripts/)
ROOT = Path(__file__).resolve().parent.parent

# Documentation directories
DOCS_EN_DIR = ROOT / "docs" / "en"        # English docs
DOCS_FR_DIR = ROOT / "docs"                # French docs (root of docs/)
README_FILE = ROOT / "README.md"           # Root README
EXAMPLES_DIR = ROOT / "examples"           # Code examples

# Output files (generated at repository root)
OUTPUT_INDEX = ROOT / "llm.txt"
OUTPUT_FULL = ROOT / "llm_full.txt"

# Directories to exclude — archived docs and Jekyll scaffolding
ARCHIVE_DIR = ROOT / "docs" / "archive"
JEKYLL_DIRS = {"_data", "_includes", "_site", ".jekyll-cache", "assets"}

# GitHub Pages base URL (Jekyll / just-the-docs)
BASE_URL = "https://dorel14.github.io/libembedding-ng"

# GitHub repository URL (for raw source links)
GH_REPO = "https://github.com/dorel14/libembedding"

# ── Navigation order ------------------------------------------------------
# Mirrors docs/_data/en_nav.yml and docs/_data/fr_nav.yml to present
# documents in the intended reading order.

# English doc order (relative to docs/en/)
EN_DOC_ORDER: list[str] = [
    "getting_started.md",
    "api_reference.md",
    "models.md",
    "performance_tuning.md",
    "advanced_usage.md",
    "similarity.md",
    "python/cache.md",
    "python/benchmark.md",
    "python/backend.md",
    "python/stats.md",
    "c_api/similarity.md",
    "c_api/embedding_mode.md",
    "c_api/embedding_cache.md",
    "c_api/autotune_cache.md",
    "c_api/downloader.md",
    "c_api/llamacpp_backend.md",
    "c_api/error.md",
    "c_api/config.md",
    "c_api/model_selector.md",
    "c_api/autotuner.md",
    "c_api/worker_autotune.md",
    "c_api/embedding_benchmark.md",
    "c_api/gguf_registry.md",
]

# French doc order (relative to docs/)
FR_DOC_ORDER: list[str] = [
    "getting_started.md",
    "api_reference.md",
    "models.md",
    "performance_tuning.md",
    "advanced_usage.md",
    "similarity.md",
    "python/cache.md",
    "python/benchmark.md",
    "python/backend.md",
    "python/stats.md",
    "c_api/similarity.md",
    "c_api/embedding_mode.md",
    "c_api/embedding_cache.md",
    "c_api/autotune_cache.md",
    "c_api/downloader.md",
    "c_api/llamacpp_backend.md",
    "c_api/error.md",
    "c_api/config.md",
    "c_api/model_selector.md",
    "c_api/autotuner.md",
    "c_api/worker_autotune.md",
    "c_api/embedding_benchmark.md",
    "c_api/gguf_registry.md",
]


# ── Utility functions ------------------------------------------------------


def clean_front_matter(content: str) -> str:
    """Supprimer le *front-matter* YAML et les attributs kramdown du Markdown.

    Les fichiers Jekyll commencent souvent par un bloc YAML entre deux
    lignes ``---`` contenant des métadonnées (titre, nav_order, slug, …).
    Cette fonction les retire pour ne garder que le contenu réel.
    """
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
    # Remove kramdown class attributes {: .class }
    content = re.sub(r"\{:.*?\}", "", content)
    return content.strip()


def read_version() -> str:
    """Extraire la version du projet depuis ``python/pyproject.toml``.

    Falls back to ``include/libembedding/config.h`` if pyproject.toml
    is not available.
    """
    pyproject = ROOT / "python" / "pyproject.toml"
    if pyproject.exists():
        match = re.search(r'^version\s*=\s*"([0-9][0-9.]*)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            return match.group(1)

    config_h = ROOT / "include" / "libembedding" / "config.h"
    if config_h.exists():
        match = re.search(r'LIBEMBEDDING_VERSION_STRING "([0-9][0-9.]*)"', config_h.read_text(encoding="utf-8"))
        if match:
            return match.group(1)

    return "unknown"


def title_from_content(content: str) -> str:
    """Extract the first H1 title from markdown content.

    Falls back to the filename stem if no H1 is found.
    """
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "Untitled"


def title_from_filename(path: Path) -> str:
    """Generate a human-readable title from a file path."""
    return path.stem.replace("-", " ").replace("_", " ").title()


def doc_url(rel_path: str, locale: str = "en") -> str:
    """Build the GitHub Pages HTML URL for a documentation file.

    Parameters
    ----------
    rel_path:
        Path relative to the docs directory (e.g. ``"c_api/similarity.md"``).
    locale:
        ``"en"`` for English docs, ``"fr"`` for French docs.

    Returns
    -------
    str
        Full URL to the rendered HTML page on GitHub Pages.
    """
    stem = rel_path.replace(".md", "").replace("\\", "/")
    if locale == "en":
        return f"{BASE_URL}/en/{stem}.html"
    return f"{BASE_URL}/{stem}.html"


def gh_source_url(rel_path: str, locale: str = "en") -> str:
    """Build a GitHub source URL for a documentation markdown file."""
    if locale == "en":
        return f"{GH_REPO}/blob/main/docs/en/{rel_path}"
    return f"{GH_REPO}/blob/main/docs/{rel_path}"


def collect_docs(docs_dir: Path, doc_order: list[str]) -> list[tuple[str, Path, str]]:
    """Collect documentation files in the specified order.

    Returns a list of ``(rel_path, abs_path, title)`` tuples. Any markdown
    files in the directory that are not in ``doc_order`` are appended at
    the end (so they are not silently dropped).
    """
    result: list[tuple[str, Path, str]] = []
    seen: set[str] = set()

    for rel in doc_order:
        abs_path = docs_dir / rel
        if abs_path.exists() and abs_path.is_file():
            content = abs_path.read_text(encoding="utf-8")
            title = title_from_content(content)
            result.append((rel, abs_path, title))
            seen.add(rel.replace("\\", "/"))

    # Catch any docs not listed in the navigation order
    if docs_dir.exists():
        for md_file in sorted(docs_dir.rglob("*.md")):
            rel = md_file.relative_to(docs_dir).as_posix()
            # Skip index.md (entry page) — covered by README
            if rel == "index.md":
                continue
            # Skip the English sub-directory when collecting a top-level docs dir
            if "en" in md_file.parts:
                continue
            # Skip Jekyll directories
            if any(part in JEKYLL_DIRS for part in md_file.parts):
                continue
            # Skip anything inside archive (shouldn't happen for en/, but be safe)
            if "archive" in md_file.parts:
                continue
            if rel not in seen:
                content = md_file.read_text(encoding="utf-8")
                title = title_from_content(content)
                result.append((rel, md_file, title))
                seen.add(rel)

    return result


# ── Generation functions ----------------------------------------------------


def generate_index(version: str) -> str:
    """Generate the ``llm.txt`` summary file content.

    Produces a concise project overview with installation instructions,
    quick-start code snippets, and a table of contents linking to every
    documentation page (English + French).
    """
    lines: list[str] = []

    # Header
    lines.append("# libembedding-ng")
    lines.append("")
    lines.append(f"> Version {version} — C/C++ and Python library for generating dense, sparse, and image embeddings from ONNX and GGUF (llama.cpp) models.")
    lines.append("")
    lines.append("**Local-first embedding and reranking engine for C/C++ and Python.**")
    lines.append("")
    lines.append("libembedding provides a unified runtime for ONNX Runtime and llama.cpp models,")
    lines.append("supporting dense embeddings, sparse embeddings, image embeddings and reranking")
    lines.append("through a single API. It is published on PyPI as `libembedding-ng` and is a")
    lines.append("fork of [pacifio/libembedding](https://github.com/pacifio/libembedding).")
    lines.append("")
    lines.append("## Key features")
    lines.append("")
    lines.append("- **Text embeddings** — 44 text models (dense), ONNX + GGUF/llama.cpp")
    lines.append("- **Sparse embeddings** — SPLADE++, BGE-M3 sparse")
    lines.append("- **Image embeddings** — CLIP, ResNet, Unicom, Nomic Vision")
    lines.append("- **Reranking** — BGE, Jina rerankers")
    lines.append("- **Auto-tuning** — find optimal workers, threads, batch size")
    lines.append("- **LRU Cache** — thread-safe cache for frequent embeddings")
    lines.append("- **Multi-backend** — ONNX Runtime + llama.cpp")
    lines.append("- **Similarity** — cosine, dot product, Euclidean")
    lines.append("- **Production helpers** — session pooling, streaming, benchmarking")
    lines.append("")

    # Installation
    lines.append("## Installation (Python)")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install libembedding-ng")
    lines.append("```")
    lines.append("")
    lines.append("The PyPI wheel bundles ONNX Runtime, libcurl, and (optionally) llama.cpp")
    lines.append("as a self-contained shared library.")
    lines.append("")

    # Quick start
    lines.append("## Quick start (Python)")
    lines.append("")
    lines.append("```python")
    lines.append("from libembedding import TextEmbedding, SparseTextEmbedding, Reranker")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("# Dense text embeddings")
    lines.append('model = TextEmbedding("BAAI/bge-small-en-v1.5")')
    lines.append('embeddings = model.embed(["The cat sat on the mat", "A kitten on a rug"])')
    lines.append("similarity = np.dot(embeddings[0], embeddings[1])  # 0.82")
    lines.append("")
    lines.append("# Sparse embeddings (SPLADE)")
    lines.append("sparse = SparseTextEmbedding()")
    lines.append('results = sparse.embed(["machine learning"])')
    lines.append("")
    lines.append("# Reranking")
    lines.append('reranker = Reranker("BAAI/bge-reranker-base")')
    lines.append('ranked = reranker.rerank("What is deep learning?", ["Deep learning uses neural networks", "The weather is sunny today"])')
    lines.append("```")
    lines.append("")

    # Quick start C
    lines.append("## Quick start (C)")
    lines.append("")
    lines.append("```c")
    lines.append("#include <libembedding/text_embedding.h>")
    lines.append("")
    lines.append("lembed_text_options_t opts = lembed_text_options_default();")
    lines.append("lembed_text_embedding_t* embedder = NULL;")
    lines.append("lembed_text_embedding_create(&opts, &embedder);")
    lines.append("")
    lines.append('const char* texts[] = { "Hello world", "How are you?" };')
    lines.append("lembed_embeddings_t result = {0};")
    lines.append("lembed_text_embedding_embed(embedder, texts, 2, 0, &result);")
    lines.append("// result.data = float[2][384], L2-normalized")
    lines.append("")
    lines.append("lembed_embeddings_free(&result);")
    lines.append("lembed_text_embedding_free(embedder);")
    lines.append("```")
    lines.append("")

    # English documentation TOC
    lines.append("## English documentation")
    lines.append("")
    en_docs = collect_docs(DOCS_EN_DIR, EN_DOC_ORDER)
    for rel, _, title in en_docs:
        url = doc_url(rel, "en")
        src = gh_source_url(rel, "en")
        lines.append(f"- [{title}]({url}) — [source]({src})")
    lines.append("")

    # French documentation TOC
    lines.append("## Documentation française")
    lines.append("")
    fr_docs = collect_docs(DOCS_FR_DIR, FR_DOC_ORDER)
    for rel, _, title in fr_docs:
        url = doc_url(rel, "fr")
        src = gh_source_url(rel, "fr")
        lines.append(f"- [{title}]({url}) — [source]({src})")
    lines.append("")

    # Code examples
    lines.append("## Code examples")
    lines.append("")
    py_examples = sorted((EXAMPLES_DIR / "python").glob("*.py")) if (EXAMPLES_DIR / "python").exists() else []
    cpp_examples = sorted((EXAMPLES_DIR / "cpp").glob("*.cpp")) if (EXAMPLES_DIR / "cpp").exists() else []
    for ex in py_examples:
        url = f"{GH_REPO}/blob/main/examples/python/{ex.name}"
        lines.append(f"- [Python: {ex.name}]({url})")
    for ex in cpp_examples:
        url = f"{GH_REPO}/blob/main/examples/cpp/{ex.name}"
        lines.append(f"- [C++: {ex.name}]({url})")
    lines.append("")

    # Repository links
    lines.append("## Repository")
    lines.append("")
    lines.append(f"- **GitHub**: [{GH_REPO}]({GH_REPO})")
    lines.append(f"- **PyPI**: [https://pypi.org/project/libembedding-ng/](https://pypi.org/project/libembedding-ng/)")
    lines.append(f"- **Documentation**: [{BASE_URL}/]({BASE_URL}/)")
    lines.append(f"- **License**: MIT (fork of pacifio/libembedding)")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*This file is auto-generated by `scripts/generate_llm_docs.py`. "
                 f"The full documentation is in [`llm_full.txt`](llm_full.txt).*")
    lines.append("")

    return "\n".join(lines)


def generate_full_text(version: str) -> str:
    """Generate the ``llm_full.txt`` complete documentation corpus.

    Concatenates the README and all documentation pages (English + French)
    into a single plain-text file with clear section separators. YAML
    front matter is stripped.
    """
    sections: list[str] = []

    # Header
    sections.append(f"# libembedding-ng — Full Technical Documentation")
    sections.append(f"")
    sections.append(f"Version: {version}")
    sections.append(f"Source: {GH_REPO}")
    sections.append(f"Site: {BASE_URL}/")
    sections.append(f"")

    # README
    if README_FILE.exists():
        readme_content = clean_front_matter(README_FILE.read_text(encoding="utf-8"))
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")
        sections.append(f"## DOCUMENT: README.md (Project overview)")
        sections.append(f"")
        sections.append(readme_content)

    # English documentation
    en_docs = collect_docs(DOCS_EN_DIR, EN_DOC_ORDER)
    for rel, path, title in en_docs:
        content = clean_front_matter(path.read_text(encoding="utf-8"))
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")
        sections.append(f"## DOCUMENT (EN): {title}")
        sections.append(f"Source: {gh_source_url(rel, 'en')}")
        sections.append(f"Rendered: {doc_url(rel, 'en')}")
        sections.append(f"")
        sections.append(content)

    # French documentation
    fr_docs = collect_docs(DOCS_FR_DIR, FR_DOC_ORDER)
    for rel, path, title in fr_docs:
        content = clean_front_matter(path.read_text(encoding="utf-8"))
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")
        sections.append(f"## DOCUMENT (FR): {title}")
        sections.append(f"Source: {gh_source_url(rel, 'fr')}")
        sections.append(f"Rendu: {doc_url(rel, 'fr')}")
        sections.append(f"")
        sections.append(content)

    # Code examples
    sections.append(f"")
    sections.append(f"---")
    sections.append(f"")
    sections.append(f"# Code Examples")
    sections.append(f"")

    py_examples = sorted((EXAMPLES_DIR / "python").glob("*.py")) if (EXAMPLES_DIR / "python").exists() else []
    cpp_examples = sorted((EXAMPLES_DIR / "cpp").glob("*.cpp")) if (EXAMPLES_DIR / "cpp").exists() else []

    for ex in py_examples:
        code = ex.read_text(encoding="utf-8")
        url = f"{GH_REPO}/blob/main/examples/python/{ex.name}"
        sections.append(f"## Example (Python): {ex.name}")
        sections.append(f"Source: {url}")
        sections.append(f"")
        sections.append(f"```python")
        sections.append(code)
        sections.append(f"```")

    for ex in cpp_examples:
        code = ex.read_text(encoding="utf-8")
        url = f"{GH_REPO}/blob/main/examples/cpp/{ex.name}"
        sections.append(f"## Example (C++): {ex.name}")
        sections.append(f"Source: {url}")
        sections.append(f"")
        sections.append(f"```cpp")
        sections.append(code)
        sections.append(f"```")

    return "\n".join(sections)


def generate() -> None:
    """Generate ``llm.txt`` (index) and ``llm_full.txt`` (full corpus)."""
    version = read_version()
    index_content = generate_index(version)
    full_content = generate_full_text(version)

    OUTPUT_INDEX.write_text(index_content, encoding="utf-8")
    OUTPUT_FULL.write_text(full_content, encoding="utf-8")

    # Report how many docs were included
    en_count = len(collect_docs(DOCS_EN_DIR, EN_DOC_ORDER))
    fr_count = len(collect_docs(DOCS_FR_DIR, FR_DOC_ORDER))
    print(f"Version: {version}")
    print(f"English docs: {en_count}, French docs: {fr_count}")
    print(f"Generated: {OUTPUT_INDEX.name} ({len(index_content)} bytes)")
    print(f"Generated: {OUTPUT_FULL.name} ({len(full_content)} bytes)")


if __name__ == "__main__":
    generate()
