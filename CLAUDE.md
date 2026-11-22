# CLAUDE.md — libembedding-ng

Bibliothèque d'embeddings (dense, sparse, image, reranking) en C/C++ header-only avec
bindings Python (cffi), deux backends : ONNX Runtime et llama.cpp (GGUF).
Package PyPI : `libembedding-ng`, importé sous `libembedding`. Licence MIT.

Les règles générales du dépôt (structure, conventions, options CMake, archivage) sont
dans AGENTS.md, importé ici :

@AGENTS.md

Ce fichier ajoute ce qui sert à Claude Code au quotidien : manière de travailler,
commandes de validation, pièges déjà payés, et état de la roadmap.

---

## 1. Manière de travailler (non négociable)

- **Environnement : Windows + PowerShell.** Toutes les commandes proposées sont en PowerShell.
- **Modifications de code uniquement par édition directe des fichiers** (outil d'édition).
  Jamais de modification via script batch, Python, `sed`, ou autre outil de réécriture en masse.
- **Builds uniquement dans `build/`** (ou `build-<variante>/` pour un audit isolé en worktree).
  Nettoyer les artefacts dès qu'ils ne servent plus.
- **Vérifier, ne pas supposer.** Avant d'affirmer un fait sur le code, le lire (fichier:ligne).
  Les plans du projet distinguent toujours « vérifié dans le dépôt » de « supposé » : faire pareil.
- **Plan d'abord pour tout changement non trivial.** Format des plans du projet :
  Contexte → Décisions actées → Tâches ordonnées → Validation → Risques → Hors périmètre.
  Les plans vivent dans `.kilo/plans/`.
- **Si la validation de base est rouge, s'arrêter** : corriger d'abord, ne pas empiler les phases.
- **Rayon d'action** : après chaque tâche, `git diff --name-only` ne doit lister que les fichiers
  prévus. `git diff` ne doit montrer que des changements sémantiques (LF imposé par
  `.gitattributes`, pas de BOM UTF-8).
- **Gros changements de `third_party/`** (bump llama.cpp, ONNX Runtime…) : dans un
  **worktree Git séparé**, baseline conservée pour comparaison, puis `git worktree remove`.
- Rendre compte avant de continuer si un échec sort du périmètre du plan en cours.

## 2. Version et commits

- **Aucun bump manuel de version.** semantic-release calcule la version à partir des commits
  conventionnels sur `main`. Ne pas éditer `config.h` / `python/pyproject.toml` pour la version.
- `python scripts/sync_version.py --check` doit rester vert.
- **Ne jamais éditer `CHANGELOG.md` à la main.**
- Commits conventionnels (`feat:`, `fix(scope):`, `test:`, `docs:`, `perf:`, `refactor:`),
  une feature testée = un commit.

## 3. Commandes de validation (protocole V0→V5)

Depuis la racine du dépôt, en PowerShell :

| Étape | Commande | Critère |
|---|---|---|
| V0 | relire les fichiers concernés avant d'éditer | — |
| V2 | `cmake -S . -B build` puis `cmake --build build --config Debug` | 0 erreur, 0 nouveau warning projet (ignorer `third_party/`) |
| V3 | `ctest --test-dir build -C Debug --timeout 120` | tous verts, **sans manipuler `PATH`** |
| V4 | `cd python` puis `pytest tests/ -q` (interpréteur `.venv`) | 0 échec ; relever le baseline réel, ne pas réutiliser un ancien chiffre |
| V4b | `pytest tests/ -q -m "not network"` | passe sans réseau |
| V5 | `ruff check python/src python/tests` ; `pyright python/src` ; `python scripts/sync_version.py --check` | tout vert |
| Docs | `python scripts/generate_llm_docs.py` | si `README.md`, `docs/`, `examples/` ou le script ont changé |

Raccourci : `.\run_tests.ps1` (build Debug + ctest).
La CI wheel exige une couverture Python ≥ 40 % (`--cov-fail-under=40`).

## 4. Pièges déjà rencontrés

- **`ctest` « Timeout » = souvent une DLL manquante.** Le process meurt avec `0xC0000135`
  (`STATUS_DLL_NOT_FOUND`). Lancer l'exe directement et lire `$LASTEXITCODE`.
- **DLL libcurl Windows** : jamais commitée. `pwsh -File scripts/fetch_windows_libcurl.ps1`
  la place dans `third_party/curl/bin/`. Sans elle, la configure échoue (voulu).
- **llama.cpp vendorisé = snapshot élagué v0.6.0 (ggml 0.26.0)**. Après tout remplacement :
  `pwsh -File scripts/prune_llama_cpp.ps1`. Les options forcées à `OFF` dans le
  `CMakeLists.txt` racine ne doivent jamais passer à `ON` (dossiers supprimés).
- **MSVC Debug** : `/O2` et `/RTC1` sont incompatibles ; les flags d'optimisation
  (`/O2 /Oi /Ot /arch:AVX2`) vont en Release/RelWithDebInfo seulement. `/GL` reste interdit.
- **Toute modification de l'API C publique** doit être reportée dans
  `python/src/libembedding/_cdefs.h` **et** `libembedding.def`, sinon `AttributeError` côté Python.
- **Cache LRU** : `lembed_cache_get()` rend un pointeur emprunté (déprécié) ;
  utiliser `lembed_cache_get_copy()` (copie sous verrou).
- **llama.cpp et BERT** : aucune logit MLM n'est exposée pour l'architecture BERT
  (`res->t_logits` jamais assigné). Le sparse GGUF passe donc par un runtime ggml maison,
  pas par `llama_get_logits_ith()`.
- **Tenseurs quantifiés (Q8_0)** : ne jamais lire `n * sizeof(float)` brut, ça dépasse
  `ggml_nbytes` et fait avorter le process. Lectures déquant-sûres uniquement.
- **Dense llama.cpp** : `llama_encode()` + `llama_get_embeddings_seq(ctx, 0)`,
  `batch.logits[i] = 0`. Utiliser `llama_model_load_from_file` / `llama_model_free`.
- **macOS** : les dylibs vendorisées doivent passer par `install_name_tool`
  (`@loader_path`) puis `delocate-wheel`, aucun chemin `build/` ou `/Users/runner/` dans `otool -L`.
- **Benchmarks** : un sous-process par backend (le pic RSS est un high-water mark par process),
  warmup séparé du run chronométré, médiane retenue.

## 5. Garde-fous de périmètre

- **Pas de recherche vectorielle** : ni HNSW, IVF, PQ, sharding, top-K nearest neighbors.
- **Le backend se choisit par extension / point d'entrée** (`.gguf` → llama.cpp, `.onnx` /
  identifiant HF / inconnu → ONNX Runtime). Exception actée (2026-11-21, option A) : le champ
  `backend` (`ONNX` / `LLAMACPP` / `AUTO`) est une option officielle **du reranker et du texte
  dense** (forcer un backend, repli ONNX → GGUF en `AUTO` si les fichiers ONNX sont
  introuvables). Sparse et image n'ont pas ce champ (changer la struct = rupture d'ABI : sparse
  demanderait `lembed_sparse_options_v2`). Ne pas étendre `backend=` à d'autres modalités sans
  décision.
- Ne pas transformer la lib en framework ML généraliste.
- **Modèles du registry** : licences compatibles MIT uniquement (pas de CC-BY-NC) ;
  ajouter peu de modèles, chacun validé (fidélité vs ONNX/PyTorch, Recall@K).
- GPU (CUDA, DirectML, CoreML, TensorRT, Vulkan, Metal) : hors périmètre jusqu'après 1.x.
- Ne jamais présumer qu'une quantification est plus rapide : mesurer.

## 6. État de la roadmap (au 2026-11-22)

Fait : API C/C++/Python stable, autotune (texte, reranker, sparse, quantization auto),
cache LRU, backend llama.cpp dense + reranker, sparse ONNX avec `top_k`/`min_weight`,
benchmark 4 backends (colonne Sparse GGUF encore `unavailable`), audit llama.cpp v0.6.0
(merge conditionnel : sparse +8,6 % débit, dense légère régression dans le bruit).

En cours / ouvert :
- **P2.6 — CPU dispatch runtime** (baseline/AVX2/AVX-512/NEON/SVE) : XL, phases 0→6.
  Phase 0 close (`.kilo/plans/rfc-cpu-dispatch.md`, décisions actées 2026-11-21) : ggml
  seulement (SIMD maison si > 3 % mesuré), score CPUID ggml + override
  `LEMBED_GGML_CPU_VARIANT`, une wheel par plateforme avec toutes les variantes, baseline
  x86-64-v1 / armv7 NEON-VFPv4 / aarch64 ARMv8.0 / macOS M1, ordre Windows → Linux →
  macOS/ARM. Pas de matériel Apple Silicon / AMD : validation par la CI (runners, Intel
  SDE pour la correction par ISA, QEMU armv7l) ; la perf hors runners se corrige sur
  retours utilisateurs. Phase 1 close le 2026-11-22 (5 tours entrelacés, i7-1065G7) :
  le build dispatch égale `build-vnni` à moins de 1,5 % (requête, docs courts et longs),
  le natif reste 7 à 9 % derrière ; +22 ms au chargement, avantage ×12,7 sur ONNX.
  Suivant : phase 2 (Linux x64), validée par la CI.
- **Fix chargement `libembedding.dylib` sur macOS** : implémenté, validation sur runner
  `macos-14` et doc `python/README.md` restantes.
- **LE-8.8 / LE-1.11** : wheels multi-plateformes complètes.
- **LE-11** : benchmarks concurrentiels vs SentenceTransformers et FastEmbed.
- **T14 ColBERT** : RFC `.kilo/plans/rfc-colbert.md`, décisions en attente.

Fait depuis le 2026-11-01 : ONNX Runtime 1.31.0, sparse GGUF (LE-9.4) avec entrée
registry, tests Linux en CI (LE-1.10), tokenizers HF-exacts (WordPiece, Unigram, BPE
byte-level et `Whitespace`), matrice de compatibilité entièrement validée sur les
modèles en cache, smoke-test de perf en CI (ratios, non bloquant).