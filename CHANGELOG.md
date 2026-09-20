# CHANGELOG


## v1.7.0 (2026-09-19)

### Bug Fixes

- **docs**: Fix GitHub Pages baseurl and nav URLs for libembedding-ng repo
  ([`d1f3486`](https://github.com/dorel14/libembedding-ng/commit/d1f3486333703a75412abc0ebe3f2cfeca957529))

- Change baseurl from /libembedding to /libembedding-ng (matches repo name) - Fix nav URLs to be
  relative (no leading /) for proper baseurl resolution - Fix English index link to French index
  (../index.html)

### Chores

- **tests**: Refactor CMakeLists.txt for test organization
  ([`a7ede63`](https://github.com/dorel14/libembedding-ng/commit/a7ede63fc053b71cd37f49be853ee7012f4ac485))

- Consolidated unit and integration test sections for clarity. - Ensured all test executables are
  properly linked to `libembedding`. - Maintained conditional logic for Windows and ONNXRuntime
  library handling. - Preserved functionality for local loading tests without network dependencies.

### Features

- Ajout de structures de données versionnées pour les modèles et options d'embedding
  ([`b502b24`](https://github.com/dorel14/libembedding-ng/commit/b502b24b966303beeffd389eb05614d4bac3dfdd))

* Ajout de `lembed_model_desc_v2_t`, `lembed_text_options_v2_t`, et `lembed_reranker_options_v2_t`
  pour étendre les fonctionnalités avec des informations de quantification et de cache. *
  Amélioration de la gestion des versions pour une meilleure compatibilité et extensibilité.

- **docs**: Add documentation for similarity functions and backend detection
  ([`b3a8c4d`](https://github.com/dorel14/libembedding-ng/commit/b3a8c4d5d41f57b6264d9778f885f4c35b22e8a5))

- Introduced `similarity.md` detailing native functions for comparing embedding vectors. - Added
  `backend.md` to explain auto-detection of backends based on model names or file paths. - Created
  `benchmark.md` for unified benchmarking of ONNX and llama.cpp backends. - Documented `cache.md`
  for LRU caching of embeddings, including usage and integration with `TextEmbedding`. - Added
  `stats.md` to provide runtime statistics for embedding contexts. - Updated French documentation
  for similarity functions and backend detection. - Implemented a new LRU cache class in `cache.py`
  for efficient embedding storage. - Added a script for syncing version numbers across header files.


## v1.6.0 (2026-09-15)

### Bug Fixes

- Include llama.cpp runtime DLLs in Windows wheel
  ([`620a85a`](https://github.com/dorel14/libembedding-ng/commit/620a85a153e3857d575fdf0b809c8cc51daee6da))

### Chores

- Bump version to 1.6.0 for PyPI release
  ([`ab8be9a`](https://github.com/dorel14/libembedding-ng/commit/ab8be9a12616bbeb58cf53ecd251d5246f52303f))

This fixes the PyPI deployment issue where semantic-release was confused by the large gap between
  last PyPI release (1.0.2) and current code (1.5.8+). Version 1.6.0 is higher than all existing
  tags and PyPI releases.

### Features

- **benchmarks**: 🚀 add comprehensive autotune tests and benchmarks
  ([`052a69b`](https://github.com/dorel14/libembedding-ng/commit/052a69b84eb7aeba463f7b46828010e3303ed356))

- Introduced multiple benchmark scripts for autotuning with various corpus sizes and configurations.
  - Added tests for: - Custom user corpus in `test_autotune_corpus.py` - Python autotune from script
  in `test_autotune_python.py` - C++ autotuner with constraints and configurable weights in
  `test_autotuner.cpp` - GGUF model registry in `test_gguf_registry.cpp` - Large corpus simulation
  with stratified sampling in `test_large_corpus.py` - Unified backend benchmark comparing ONNX and
  llama.cpp in `test_unified_benchmark.cpp` - Tokenization vs inference profiling in
  `token_profile.cpp` - Debug trace test for embedding creation in `trace_test.cpp` - Ultra minimal
  embedding test in `ultra_min.cpp` - Verification of reproducibility in `verify_bench.cpp` -
  Windows API integration test in `win_test.cpp`

These additions enhance the benchmarking capabilities and provide a more robust framework for
  evaluating model performance across different scenarios.

### Testing

- 🧪 add comprehensive unit and integration tests for libembedding
  ([`b4d6683`](https://github.com/dorel14/libembedding-ng/commit/b4d66838c6420d7d0570f0431fc796f537595349))

- Introduced unit tests for exception hierarchy in `libembedding.exceptions`. - Added integration
  tests for `ImageEmbedding`, `SparseTextEmbedding`, and `TextEmbeddingPool`. - Implemented smoke
  tests for package imports and public API exports. - Created tests for model registry and
  resolution, ensuring model availability and properties. - Developed tests for similarity functions
  and corpus sampling utilities. - Included tests for various dataclasses in `libembedding.types`. -
  Added a script to sync version across C/Python headers, ensuring consistency in versioning.

These changes enhance the test coverage and reliability of the `libembedding` library, facilitating
  easier maintenance and future development.


## v1.5.8 (2026-09-05)

### Bug Fixes

- Set TextEmbedding default batch_size to 256 to match C API
  ([`8bf437a`](https://github.com/dorel14/libembedding-ng/commit/8bf437adf6b5ce2326757a9de46449406fdf1917))


## v1.5.7 (2026-09-05)

### Bug Fixes

- Add missing TextEmbedding properties and correct embed/stats methods
  ([`ae22c3c`](https://github.com/dorel14/libembedding-ng/commit/ae22c3caa5650d7737042ec0ce5bcc3ff5f2145d))


## v1.5.6 (2026-09-05)

### Bug Fixes

- Correct lembed_free_string usage and add num_threads deprecation in TextEmbedding
  ([`40947d3`](https://github.com/dorel14/libembedding-ng/commit/40947d340fff1fa47dbb76944b01099f2a228f3a))


## v1.5.5 (2026-09-05)


## v1.5.4 (2026-09-05)

### Bug Fixes

- Correct lembed_model_selection_t fields to double and remove duplicate
  ([`9cacf39`](https://github.com/dorel14/libembedding-ng/commit/9cacf392b9f517b34860bb10e731139d8fa1f9a4))


## v1.5.3 (2026-09-05)

### Bug Fixes

- Remove duplicate lembed_model_selection_t definition in _cdefs.h
  ([`ec76bcf`](https://github.com/dorel14/libembedding-ng/commit/ec76bcfb876d252b11fdcee770f4e5b890af0fd8))


## v1.5.2 (2026-09-05)

### Bug Fixes

- Add missing lembed_sparse_options_t definition to _cdefs.h
  ([`21f7b3a`](https://github.com/dorel14/libembedding-ng/commit/21f7b3a3440fa3956a79984e8ac053fafe608f7d))

- Correct include path for model_registry.h in embedding_mode_impl.hpp
  ([`b4c7306`](https://github.com/dorel14/libembedding-ng/commit/b4c73061e1ebce7f52cfa88d315a99481120d57b))

The detail/embedding_mode_impl.hpp file incorrectly included model_registry.h without the ../
  prefix, causing a build failure on all platforms when compiling the shared library.

- Use correct target name libembedding_stb for macOS/Linux image linking
  ([`301157b`](https://github.com/dorel14/libembedding-ng/commit/301157b609fa0f8a5d0ff218b329b6125cffe349))


## v1.5.1 (2026-09-05)

### Bug Fixes

- Limit cmake parallel build on Linux/macOS and fix Windows encoding
  ([`dbfad2b`](https://github.com/dorel14/libembedding-ng/commit/dbfad2b2fa794ca753a9419713d4a584ea4eafec))

- Limit cmake --build parallel jobs to 2 on Linux/macOS to avoid OOM when compiling the bundled
  llama.cpp with all models. - Add encoding=utf-8 to _cdefs_path.read_text() in _binding.py to fix
  UnicodeDecodeError on Windows where default encoding is cp1252.


## v1.5.0 (2026-09-05)


## v1.4.0 (2026-09-05)

### Bug Fixes

- Remove tracked CMake build artifacts and update .gitignore
  ([`054df71`](https://github.com/dorel14/libembedding-ng/commit/054df71cc706c7f7bca9b40db97358d0a66f8e4b))

CMakeCache.txt, CMakeFiles/, and related artifacts were accidentally committed with local Windows
  paths, breaking cross-platform CI builds.

This removes them from the repository and prevents future inclusion.

### Chores

- Finalize llama.cpp bundling and remaining integration cleanup
  ([`b18cc91`](https://github.com/dorel14/libembedding-ng/commit/b18cc91e5c885266c541c07a5d9dd1c55a9de0c0))

### Documentation

- Update README and GitHub Pages for v1.4.0
  ([`d1cffad`](https://github.com/dorel14/libembedding-ng/commit/d1cffad4ba8a73bec29cbf796bd99f52fe590f41))

### Features

- Add LlamaRerankerProvider with GGUF auto-routing
  ([`650cda8`](https://github.com/dorel14/libembedding-ng/commit/650cda851579e214165c6fb49c1a787539ee2b24))

- Integrate llama.cpp v0.3.0 as always-on backend
  ([`daf69ea`](https://github.com/dorel14/libembedding-ng/commit/daf69eaf5af3d5d902457a49655b11b11b24afba))

### Testing

- Add llama.cpp integration tests
  ([`d932cde`](https://github.com/dorel14/libembedding-ng/commit/d932cde55991d51ea8b46c80bfd42a05ffcaeb49))


## v1.3.0 (2026-08-29)

### Bug Fixes

- Code review cleanup and improvements
  ([`1248445`](https://github.com/dorel14/libembedding-ng/commit/12484457efc284aa4a10b94b52894649a39cce8d))

- Remove DEBUG fprintf from production code

- Fix unified autotuner to support IMAGE task

- Add sparse autotune wrapper in extern C block

- Add SparseTuningResult and ImageTuningResult types

- Expose sparse_autotune and image_autotune in Python

- Expose all autotune enums (objectives, tasks, profiles)

- Remove duplicate struct definitions

- Expose autotune enums in Python bindings
  ([`3e5c6b7`](https://github.com/dorel14/libembedding-ng/commit/3e5c6b77b6ce690a5ba073067f615ce603271ec2))

- Add LEMBED_OBJECTIVE_* (latency, throughput, balanced, memory)

- Add LEMBED_TASK_* (embedding, reranking, image, sparse)

- Add LEMBED_PROFILE_* (interactive, balanced, quality)

- Verify all auto-tuner functions work from Python

- Expose sparse and image autotuners in Python
  ([`dc4f943`](https://github.com/dorel14/libembedding-ng/commit/dc4f94342cee27393676a1ecc032930e37b75f5f))

- Add sparse_autotune() wrapper

- Add image_autotune() wrapper

- Add SparseTuningResult and ImageTuningResult types

- Fix image autotuner with BMP-based benchmark (no zlib needed)

- All 4 autotuners now work from Python

- Remove duplicate declarations in _cdefs.h
  ([`bdd42bc`](https://github.com/dorel14/libembedding-ng/commit/bdd42bcd605298dff6b871636f867fa7da83dec1))

- Remove old lembed_autotune with int return type

- Remove duplicate lembed_autotune_mode_t definition

- Remove duplicate lembed_tuning_result_t definition

- Keep only lembed_status_t return type versions

- Resolve remaining structural issues (S2, S4, S6)
  ([`a9771da`](https://github.com/dorel14/libembedding-ng/commit/a9771dadf47d2828d501a1099ef037ed87dcde46))

- S2: Remove duplicate lembed_reranker_autotune declarations in _cdefs.h

- S4: Delete model_selector_impl.hpp (duplication)

- S6: Extract check_status() helper, remove 4x check_or_throw()

- R4: Extract is_optional_file() helper in downloader_impl.hpp

### Documentation

- Detect language from URL prefix /en/ (fix switch-link misclassification)
  ([`37c90b6`](https://github.com/dorel14/libembedding-ng/commit/37c90b6b8932ac02a7fa6ca642df8ea059cee000))

- The cross-language switch links made URL-membership detection misclassify the French home
  (/index.html) as English (en_nav's 'Documentation française' entry also points to /index.html). -
  Use page.url prefix '/en/' instead: only pages under /en/ are English, so the French home
  correctly gets the French sidebar.

- Document pacifio/libembedding fork, Windows DLL, and GitHub Pages setup
  ([`60c31dd`](https://github.com/dorel14/libembedding-ng/commit/60c31dd783e4a77387395783d2d0a68b1212bf28))

- Mark the project as a fork of pacifio/libembedding, published on PyPI as libembedding-ng - Drop
  the inaccurate 'header-only' claim: Windows now builds a native DLL and a compiled shared library
  is bundled for the Python bindings - Fix PyPI install name (libembedding-ng) and stale version/URL
  references (yourorg/libembedding, 0.3.0, py3-none-any wheels) - Add docs/_config.yml (Jekyll) so
  the docs/ folder publishes via GitHub Pages 'Deploy from a branch'; fix internal .md links so they
  resolve as .html

- Fix sidebar language detection
  ([`e0c0217`](https://github.com/dorel14/libembedding-ng/commit/e0c0217ad4d7979c83b5d327298ec987a18ac115))

- Replace fragile 'page.path contains en/' check (misclassified French pages as English) with
  URL-membership detection: a page is English only if its URL matches one of the English nav
  entries. This guarantees the sidebar language matches the page.

- Image quantization methodology and storage analysis
  ([`7dff953`](https://github.com/dorel14/libembedding-ng/commit/7dff953c607eda110c56a5d61fbd0a14d183e1b8))

- Document CLIP INT8/UINT8/FP16 quantized models from Xenova

- Storage cost: CLIP 512-dim = 2KB vs ResNet 2048-dim = 8KB per vector

- At 1M images: CLIP FP32 = 2GB, CLIP INT8 = 0.5GB

- Image storage cost analysis
  ([`10f4750`](https://github.com/dorel14/libembedding-ng/commit/10f47500dc260d7aac95d2dc1529520b9bc752d4))

- CLIP INT8: 0.5 KB/vector, ResNet FP32: 8 KB/vector

- At 1M images: CLIP INT8 = 0.51 GB, ResNet FP32 = 8.19 GB

- Make sidebar language-aware (FR/EN switch)
  ([`53cca0f`](https://github.com/dorel14/libembedding-ng/commit/53cca0f0a402e95218714174fc2f5947d3b88306))

- Override components/site_nav.html to render a French nav on French pages and an English nav on
  English pages, so the sidebar language matches the page being read (fixes English pages showing a
  French sidebar) - Define nav entries in _data/fr_nav.yml and _data/en_nav.yml - Drop the
  now-redundant 'Documentation anglaise' aux link (switch is in the nav)

- Order sidebar nav and separate FR/EN via front matter
  ([`61de320`](https://github.com/dorel14/libembedding-ng/commit/61de32009cfcdacd981eab1265fee3ba2f42609e))

- Remove the config nav: block (it silently fell back to the auto-generated nav, mixing FR/EN pages
  and ignoring order) - Set nav_order (1-6) on the French pages for a coherent order: Accueil ->
  Demarrage -> API Python -> Modeles -> Performance -> Usage avance - Set nav_exclude: true on the
  English pages so they no longer clutter the sidebar; a 'Documentation anglaise' aux link provides
  access - Enable nav_enabled: true

- Override whole sidebar.html instead of cached site_nav
  ([`a472344`](https://github.com/dorel14/libembedding-ng/commit/a4723444e8e4d48d040564500ff755fb7ea4cf25))

- The theme pulls site_nav via include_cached, which can ignore project overrides on GitHub Pages,
  so the language-aware nav was not applied (French pages kept showing the English sidebar). -
  Override components/sidebar.html directly (plain include from the layout) so the FR/EN nav switch
  is guaranteed to render. Drop the unused site_nav override.

- Switch GitHub Pages theme to just-the-docs
  ([`cc9aaf8`](https://github.com/dorel14/libembedding-ng/commit/cc9aaf8468fd32b0e26f2e2f02c77bf998c7bed0))

- Replace jekyll-theme-minimal with remote_theme: just-the-docs/just-the-docs for a modern,
  searchable docs site with a sidebar navigation - Add a defaults block applying the theme layout to
  all pages (the theme's own default only targets a docs/ subfolder, but our Pages source root is
  docs/) - Define an explicit sidebar nav, aux links, and correct 'Edit this page' repo - Add custom
  accent SCSS (plain CSS overrides, build-safe) for a modern indigo look

### Features

- Autotuner with objectives and constraints
  ([`3362431`](https://github.com/dorel14/libembedding-ng/commit/33624316b9db1202f0030f04d56420499c6ea359))

- Add optimization objectives: latency, throughput, balanced, memory

- Add constrained autotuner (min_tokens, max_latency_ms)

- Image quality benchmark: CLIP > ResNet for retrieval

- Image profiling: ONNX dominates (3x-500x preprocessing)

- Fix sparse autotuner signature mismatch

- 4 specialized autotuners: text, sparse, reranker, image

- Clip INT8 quantized model integration
  ([`6ca4201`](https://github.com/dorel14/libembedding-ng/commit/6ca4201169143a3fb58d4d23fd2cebaf458fcb00))

- Add CLIP INT8 model from Xenova (88.6 MB)

- INT8 1.21x faster than FP32 (68.2 vs 82.8 ms/image)

- INT8 3.95x less RAM (92 vs 364 MB)

- INT8 3.13x faster load (316 vs 988 ms)

- Image embedding profiling and pipeline optimization
  ([`b8f8ce1`](https://github.com/dorel14/libembedding-ng/commit/b8f8ce17f9f9581464ed74ad0f2a8a65bc370519))

- Profile preprocessing vs inference: ONNX dominates (3x-500x)

- Zero-copy preprocessing (eliminate ImageTensor intermediate copy)

- Optimized RGB→CHW loop (cache-friendly pixel access)

- Add image auto-tuner (threads × batch_size)

- Profiling shows batching gives 22-27% gain for transformers

- ResNet-50 fastest (19.2 img/s), CLIP better for multimodal

- Image quantization and storage analysis
  ([`0561478`](https://github.com/dorel14/libembedding-ng/commit/05614786c49490b2715e1e67bb50dbbbaa46f6e4))

- CLIP INT8 model integrated (1.21x faster, 3.95x less RAM)

- Storage analysis: CLIP INT8 = 0.5GB/1M images vs ResNet FP32 = 8GB

- Autotuner with objectives: latency, throughput, balanced, memory

- Constrained autotuner: min_tokens, max_latency_ms

- Intégrer ONNX Runtime et libcurl dans la distribution globale
  ([`3a46d50`](https://github.com/dorel14/libembedding-ng/commit/3a46d50b6ef283806704b6334bf421bf31805f5d))

- Ajout de la fonction CMake copy_runtime_libs pour copier automatiquement les .so/.dylib/.dll
  d'ONNX Runtime et libcurl à côté des exécutables sur toutes les plateformes (Linux, macOS,
  Windows). - Mise à jour du workflow GitHub Actions pour inclure les DLLs ONNX Runtime et libcurl
  dans les wheels PyPI. - Application de copy_runtime_libs aux cibles examples et tests. - Mise à
  jour de la documentation (README et getting_started) pour refléter le bundling automatique au
  runtime.

- Reranker optimization with auto-tuner and INT8 quantization
  ([`60e28c1`](https://github.com/dorel14/libembedding-ng/commit/60e28c101bc10bbfb88b91581c40a700550d6b39))

- Add Jina v1 turbo INT8 quantized model (5x faster than BGE-base, 4.2x less RAM)

- Implement reranker auto-tuner (threads x batch_size x max_tokens)

- Add unified auto-tuner API (lembed_autotune_unified) for all task types

- Add profile-based auto-config: fast/balanced/quality

- Make tokenizer.json optional for vision models in downloader

- 50-query quality benchmark: INT8 NDCG@10 -0.28% vs FP32 (recommended default)

- Comprehensive benchmarks: latency, throughput, batching, memory, workers, top_k

- Auto-tuner cache per (model, CPU cores) in JSON

- Sparse embedding with auto-tuner
  ([`5dcb01c`](https://github.com/dorel14/libembedding-ng/commit/5dcb01cbc9e566a4665f4b0416797c6727cbf51a))

- Fix sparse embed function signature (6 params with sparse_opts)

- Add sparse auto-tuner (top_k x min_weight x storage_format)

- Fix file_exists to use std::filesystem (fixes BGE-M3 offline mode)

- Make tokenizer.json optional for vision models

- Add sparse benchmark: SPLADE++ 44.5 docs/s, BGE-M3 19.9 docs/s

### Refactoring

- Condense downloader_impl.hpp
  ([`a69bc01`](https://github.com/dorel14/libembedding-ng/commit/a69bc01c17376c44106c56372a5332d85248a4a7))

- Merge common functions (get_cache_dir, file_exists, read_file_to_string, mkdirs)

- Extract repo_to_dirname() helper

- Clean up stub version (remove redundant casts)

- Reduce from 331 to ~200 lines

- Factorize image_preprocess.hpp
  ([`4f20cf8`](https://github.com/dorel14/libembedding-ng/commit/4f20cf82eeb8cf3f29b6c513f8e414e3ce452e7b))

- Extract rgb_to_chw_float() helper (shared by all 4 functions)

- Use shared IMAGE_NET_MEAN/STD constants

- Fix misleading comment about cache locality

- Add nullptr validation for mean/std_dev parameters

- Reduce code duplication from ~250 to ~150 lines

- Split autotuner_impl.hpp into focused modules
  ([`5bc9524`](https://github.com/dorel14/libembedding-ng/commit/5bc952414a9944b60c760231bce8a86ef65a770e))

- autotune_cache.hpp: cache system (CPU detection, cache I/O)

- autotune_bench_text.hpp: text embedding auto-tuner

- autotune_bench_reranker.hpp: reranker auto-tuner

- autotune_bench_sparse.hpp: sparse embedding auto-tuner

- autotune_bench_image.hpp: image embedding auto-tuner

- autotuner_impl.hpp: unified orchestrator only

- Fix linkage issues with extern C wrappers


## v1.2.1 (2026-08-28)

### Bug Fixes

- **autotune**: Redefinition de variable 'f' dans autotuner_impl.hpp
  ([`6067233`](https://github.com/dorel14/libembedding-ng/commit/6067233bd752c1f8036a60a987b21b6a926f5a33))


## v1.2.0 (2026-08-28)

### Bug Fixes

- **ci**: Make version job output explicit dependency for publish job
  ([`121f796`](https://github.com/dorel14/libembedding-ng/commit/121f79682815f49277207b04e26da628bde48d46))

### Features

- **autotune**: Implémenter l'auto-tuning et la sélection automatique de modèle
  ([`d178ff2`](https://github.com/dorel14/libembedding-ng/commit/d178ff2d68368ad9070c6f52d94e073eff6ea60d))

Ajout du système complet d'auto-tuning C++/Python pour l'optimisation automatique des paramètres de
  performance (workers, threads, batch_size).

Le tuning s'appuie sur un cache persistant clé par configuration matérielle (cœurs CPU, modèle,
  version de la bibliothèque), évitant de refaire le benchmark à chaque exécution. Deux modes sont
  proposés : - QUICK (5-15s) : benchmark synthétique rapide - FULL (30-120s) : benchmark exhaustif

La sélection automatique de modèle (`auto_select_model`) évalue les modèles disponibles selon un
  use-case ("speed", "quality", "balanced") et retourne le meilleur candidat avec sa configuration
  optimale.

Les nouveaux composants : - API C `lembed_autotune`, `lembed_autotune_custom`,
  `lembed_auto_select_model`, `lembed_autotune_clear_cache` - Types Python `TuningResult` et
  `ModelSelectionResult` - Fonctions Python `autotune()`, `auto_select_model()`,
  `clear_autotune_cache()` - Documentation de performance (FR et EN) avec guide d'utilisation -
  Suite de tests complète (C++ et Python) incluant le cache, le corpus personnalisé et la sélection
  de modèle - Support Windows amélioré pour le répertoire de cache (`USERPROFILE` puis
  `LOCALAPPDATA`)


## v1.1.1 (2026-08-28)

### Bug Fixes

- **ci**: Skip model download tests when network is unavailable
  ([`1eb6443`](https://github.com/dorel14/libembedding-ng/commit/1eb6443a7570848a152729fa53a8f68ab4a3d0a6))


## v1.1.0 (2026-08-28)

### Chores

- **release**: Bump version 1.0.3 pour nouvelle release PyPI
  ([`73d9607`](https://github.com/dorel14/libembedding-ng/commit/73d960717e49b844da6332ac5eb6048f6f6c66a6))

- **release**: Test semantic-release config with gitpython pin
  ([`45b4759`](https://github.com/dorel14/libembedding-ng/commit/45b4759a98c8fbf70edd20f846cd5b0a5a6e96b7))

### Features

- **python**: Ajouter le pool d'embeddings multi-workers
  ([`28dc639`](https://github.com/dorel14/libembedding-ng/commit/28dc639f45a10341062d758ae071bfca18c53eb4))

Ajout de la classe `TextEmbeddingPool` dans l'API Python, permettant la parallélisation au niveau
  des requêtes via plusieurs sessions ONNX distinctes. Cette approche est plus efficace que le
  parallélisme intra-op d'ORT pour les petits transformeurs, offrant jusqu'à 4x le débit d'une
  session unique.

Les composants associés ont également été ajoutés : - En-têtes C++ pour l'autotuneur, le sélecteur
  de modèle et le pool d'embeddings (`autotuner.h`, `embedding_pool.hpp`, `model_selector.h`) -
  Suite de benchmarks complète (CMakeLists.txt + 40+ fichiers) - Section "Performance
  Characteristics" dans le README documentant les résultats d'optimisation CPU - Correction de la
  compatibilité Windows pour le répertoire de cache (`USERPROFILE` avant `HOME`) - Fichier
  `CHANGELOG.md` et mise à jour de `.gitignore` (`.venv/`)


## v1.0.2 (2026-08-26)

### Continuous Integration

- Bump version 1.0.2 pour nouvelle release PyPI
  ([`117bab5`](https://github.com/dorel14/libembedding-ng/commit/117bab5b2748b97155adea17a7e12adc6b35c7bb))


## v1.0.1 (2026-08-26)

### Bug Fixes

- Auditwheel step ne doit pas corrompre le nom de la wheel
  ([`dcfb8f9`](https://github.com/dorel14/libembedding-ng/commit/dcfb8f907aa97ccfd18fe1dfc3be092efea34229))

- Enlever le pin pydantic v1, laisser semantic-release gerer ses deps
  ([`377402c`](https://github.com/dorel14/libembedding-ng/commit/377402c540099858e06327c4b85effe7015764dc))

- Installer la wheel reparee aussi sur Windows (win_amd64)
  ([`8d63935`](https://github.com/dorel14/libembedding-ng/commit/8d6393582b3edc35484117d7eade5b1563b5ff57))

- Installer la wheel reparee sur Linux (manylinux) et macOS (macosx)
  ([`76e1d33`](https://github.com/dorel14/libembedding-ng/commit/76e1d339d3556f9124f6f90e842871bd49033d35))

- Installer seulement la wheel reparee (manylinux) sur Linux
  ([`017e820`](https://github.com/dorel14/libembedding-ng/commit/017e820caf0bb15693b63265efb8730ea1a33b6a))

- List wheel contents avec plusieurs wheels apres auditwheel
  ([`8ad88da`](https://github.com/dorel14/libembedding-ng/commit/8ad88da53d6f42e05a9ada1dcb5b14372b4daf8f))

- Trouver la wheel reparee avec ls au lieu du glob
  ([`a1cfd75`](https://github.com/dorel14/libembedding-ng/commit/a1cfd75b16538d4b51d2475048ecc53987a8f46b))

- Uploader seulement les wheels reparees (manylinux/macosx/win)
  ([`59b1356`](https://github.com/dorel14/libembedding-ng/commit/59b1356ca46e2518b4e78c0be1f6610ca7b9e3dd))

### Continuous Integration

- Epingler pydantic v1 pour compatibilite semantic-release 9.4.0
  ([`407362f`](https://github.com/dorel14/libembedding-ng/commit/407362f495bea063a0f89c26e5c57262e08527a9))

- Epingler semantic-release a 9.4.0 (stable)
  ([`821a08b`](https://github.com/dorel14/libembedding-ng/commit/821a08bf98a2d843a061f22790661f4512a7b483))

- Installer patchelf 0.18 depuis les sources pour auditwheel
  ([`b64ea2e`](https://github.com/dorel14/libembedding-ng/commit/b64ea2e216a98e86cc37fcb56787cdcfbe57b1b8))

- Passer a semantic-release 10.1.0 (pydantic v2 compatible)
  ([`b9c80ab`](https://github.com/dorel14/libembedding-ng/commit/b9c80ab461949e1600c2277b7200864fe679c0f0))

- Revenir a un job release simple sans semantic-release (version depuis pyproject.toml + PYPI_TOKEN)
  ([`8f287ea`](https://github.com/dorel14/libembedding-ng/commit/8f287eaf0c4c94986215f508126a649c430d1967))

- Revenir au workflow release simple qui marche + bump version 1.0.1
  ([`752adda`](https://github.com/dorel14/libembedding-ng/commit/752addaa3ebb5131334d8ddde112d1cd403d6609))

- Semantic-release 9.4.0 avec pydantic v1 + PYPI_TOKEN pour publication
  ([`ce93a23`](https://github.com/dorel14/libembedding-ng/commit/ce93a23105932eb6f2e3ee46355f0b3f0f6e6aed))

- Utiliser latest semantic-release pour compat pydantic v2
  ([`7ab04bd`](https://github.com/dorel14/libembedding-ng/commit/7ab04bd2eda2560f3d61150916d26aae746d0a4c))


## v1.0.0 (2026-08-25)

### Continuous Integration

- Build_command doit etre une chaine, utiliser un no-op
  ([`efaa9e7`](https://github.com/dorel14/libembedding-ng/commit/efaa9e7032f10f9029c6629dd87a0664c49ddf17))

- Desactiver le build dans semantic-release (wheels rebuilt separement)
  ([`bd7c364`](https://github.com/dorel14/libembedding-ng/commit/bd7c364d580be0a91ada71ca51733c497ffb57b7))

- Revenir a semantic-release avec PYPI_TOKEN et rebuild des wheels apres le bump de version
  ([`eb0ac57`](https://github.com/dorel14/libembedding-ng/commit/eb0ac57b42a18f93ecf813b9ab7dd9d2027012fc))


## v0.1.0 (2026-08-25)

### Bug Fixes

- Forcer la decouverte du package libembedding dans le wheel
  ([`5b99d0e`](https://github.com/dorel14/libembedding-ng/commit/5b99d0e38b23edb287cc43d1daee79a6ed057aa3))

- Ne pas declarencher build.yml sur push vers main pour eviter les doublons avec release.yml
  ([`240833f`](https://github.com/dorel14/libembedding-ng/commit/240833f4e911dead32de6a82ae1f8510826ce17d))

- Retirer la declaration duplicate de packages dans pyproject.toml
  ([`94c9729`](https://github.com/dorel14/libembedding-ng/commit/94c9729d4d750f611a5da91876e91aff31792ca6))

- Vendor curl DLL on Windows and rename PyPI package to libembedding-ng
  ([`02c65ee`](https://github.com/dorel14/libembedding-ng/commit/02c65eeed15c6e6c88f51f4b22793439cbd3e3ea))

- python/pyproject.toml: rename PyPI package from libembedding to libembedding-ng - CMakeLists.txt:
  make bundled third_party paths overridable via cache variables (CURL_INCLUDE_DIR, CURL_LIBRARY,
  ONNXRuntime_INCLUDE_DIR, ONNXRuntime_LIBRARY) so CI can use vcpkg's MSVC-compatible curl instead
  of the MinGW .dll.a - build-wheel.yml: on Windows, install curl via vcpkg, override curl paths to
  vcpkg's MSVC-compatible import lib, and vendor ALL third-party DLLs from vcpkg's bin (libcurl.dll,
  zlib.dll, etc.) into the wheel so it's self-contained

### Build System

- Corriger la localisation des DLL ONNX Runtime et simplifier la découverte CMake
  ([`3ccdb38`](https://github.com/dorel14/libembedding-ng/commit/3ccdb38748cab512d0b6c447be5b0dbdbbe9fbe4))

Les DLL Windows sont désormais copiés depuis le répertoire `bin` au lieu de `lib`, correspondant à
  la structure des archives officielles ONNX Runtime. Le workflow télécharge désormais l'archive
  ONNX Runtime pour Windows au lieu d'utiliser le répertoire `third_party` empaqueté.

Le module CMake `FindONNXRuntime` inclut un fallback pour les bibliothèques partagées versionnées
  (par ex. `libonnxruntime.so.1.20.1`), et `CMakeLists.txt` utilise désormais exclusivement ce
  module personnalisé en supprimant la logique de détection via la configuration officielle.

- Documenter l'incompatibilité LTCG avec WINDOWS_EXPORT_ALL_SYMBOLS
  ([`a2be5f5`](https://github.com/dorel14/libembedding-ng/commit/a2be5f5caadbe650258969a8eb2cc909b9c53e5b))

Une note explicative a été ajoutée dans la section des optimisations CPU du fichier CMakeLists.txt
  pour justifier l'absence des options `/GL` (optimisation whole-program) et `/LTCG` sous MSVC. Ces
  dernières sont volontairement omises car elles sont incompatibles avec
  `WINDOWS_EXPORT_ALL_SYMBOLS`, qui nécessite que CMake puisse analyser les fichiers objets afin de
  générer automatiquement le fichier définition d'exportation. Une alternative est proposée :
  fournir un fichier `.def` explicite pour activer LTCG.

### Chores

- Add python-semantic-release for automated versioning
  ([`2106fd7`](https://github.com/dorel14/libembedding-ng/commit/2106fd7944ed371a229cd5460ab5ec17bb58d6c3))

- pyproject.toml (root): semantic-release config reading version from python/pyproject.toml, angular
  commit parser, major_on_zero for 0.x, changelog generation, GitHub release creation (no direct
  PyPI publish) - .github/workflows/semantic-release.yml: runs on push to main, uses PAT (not
  GITHUB_TOKEN) to trigger downstream publish.yml via release event -
  python/src/libembedding/__init__.py: read version from importlib.metadata instead of hardcoding to
  prevent drift from pyproject.toml

- Remove unnecessary debug symbols and unused static libraries from third_party
  ([`1a8426d`](https://github.com/dorel14/libembedding-ng/commit/1a8426dc4a723d6d1a46de81f3ae3ad5ddea5587))

- Remove onnxruntime.pdb (389 MB debug symbols) - Remove onnxruntime_providers_shared.pdb (0.4 MB
  debug symbols) - Remove 13 unused .a static libraries from curl/lib (not referenced in
  CMakeLists.txt; only libcurl.dll.a is used for linking on Windows) - *.pdb already added to
  .gitignore in feat(build) commit

### Continuous Integration

- Add GitHub Actions workflows for building and publishing Python wheels
  ([`0b35a4d`](https://github.com/dorel14/libembedding-ng/commit/0b35a4d5b6300f58088fcefe68069f13bfe80a14))

- build-wheel.yml: reusable workflow that builds a wheel for one platform (Linux/macOS/Windows),
  vendors ONNX Runtime via auditwheel/delocate, runs tests, and uploads the wheel as an artifact -
  build.yml: CI workflow triggered on push/PR to main, builds wheels on ubuntu-22.04, macos-14,
  macos-13, windows-2022 - publish.yml: release workflow triggered on GitHub release, builds all
  wheels and publishes to PyPI via OIDC trusted publishing - python/setup.py: minimal setup.py to
  force platform-specific wheel tags (setuptools treats .so/.dll as data files by default, producing
  py3-none-any)

- Ajouter initial_version et remettre la version a 0.1.0 pour bootstrap
  ([`35c0a90`](https://github.com/dorel14/libembedding-ng/commit/35c0a90a9190302d4cf97099bcfdc3ef4330210f))

- Ajouter shell:bash sur l'etape de listage des roues Windows
  ([`35a8275`](https://github.com/dorel14/libembedding-ng/commit/35a82750ee9cfbf56d865b31a6b30e2ad938111b))

- Ajouter shell:bash sur l'etape de normalisation des roues Windows
  ([`34390a4`](https://github.com/dorel14/libembedding-ng/commit/34390a4e5215dd38c5b690d07e8535b11981c3ca))

- Corriger auditwheel/delocate et la detection ONNX Runtime sur Windows
  ([`ca5a95a`](https://github.com/dorel14/libembedding-ng/commit/ca5a95a02245a3e534e144de01bb5eb86ec2be44))

- Corriger la configuration du workflow de publication sémantique
  ([`cd6e9e6`](https://github.com/dorel14/libembedding-ng/commit/cd6e9e6e780c548e63b7836fac05eb41a29ade2b))

Le workflow GitHub Actions pour python-semantic-release a été mis à jour pour utiliser
  `secrets.GH_TOKEN` au lieu de `secrets.SEMANTIC_RELEASE_PAT` et ajoute une étape de configuration
  git pour le bot github-actions. Les URLs du projet dans `pyproject.toml` ont également été
  corrigées pour pointer vers le dépôt `dorel14/libembedding`.

Les fichiers README.md et python/README.md ont été mis à jour pour documenter les nouvelles
  fonctionnalités (chargement local de modèles, streaming, statistiques, fonctions de similarité) et
  les nouveaux champs d'options.

- Corriger le chemin ONNX Runtime et rétablir la version 0.1.0
  ([`f480ed4`](https://github.com/dorel14/libembedding-ng/commit/f480ed45815b09ce603dc56f8e0636d4c01e4a99))

Le chemin `ONNXRUNTIME_ROOT` dans le workflow de construction utilise maintenant un séparateur de
  chemin cohérent (`/`), évitant les problèmes de résolution de chemin sous Windows. La version du
  projet est rétablie à `0.1.0` conformément aux versions précédentes publiées.

- Corriger le telechargement ONNX Runtime Windows et les chemins CMake
  ([`74f8fef`](https://github.com/dorel14/libembedding-ng/commit/74f8fef6aac01ac0a85730e3be1038b43e2f62d0))

- Desactiver commit_version_number pour utiliser la version de pyproject.toml
  ([`11b4a40`](https://github.com/dorel14/libembedding-ng/commit/11b4a4097db41c8d442008fc592f230315026660))

- Desactiver upload_to_pypi dans semantic-release et supprimer le tag v0.2.0
  ([`5f33c5f`](https://github.com/dorel14/libembedding-ng/commit/5f33c5f3243e21bb789c3c605546e939cfdc5687))

- Exporter GH_TOKEN pour gh et garder la publication pyPI conditionnee au nouveau tag
  ([`a827657`](https://github.com/dorel14/libembedding-ng/commit/a827657f91f73b445d7123ef342074c13e9bff2c))

- Mettre a jour la version vers 0.2.0 pour declarer la release
  ([`a665deb`](https://github.com/dorel14/libembedding-ng/commit/a665deb3fe551309c17e870ad821aaa4016d8778))

- Rechercher les DLL ONNX Runtime recursivement sur Windows
  ([`cb36e9a`](https://github.com/dorel14/libembedding-ng/commit/cb36e9a3e0b2620bbe9eede7cc3b25df81b86f2c))

- Remplacer semantic-release par un job de release simple (tag + github release + pypi OIDC)
  ([`492c8b9`](https://github.com/dorel14/libembedding-ng/commit/492c8b980bfb01cbe3ff22ba3bdc87fd7d4194d7))

- Retirer macos-13 de la matrice pour debloquer la release
  ([`14b1ef1`](https://github.com/dorel14/libembedding-ng/commit/14b1ef1c49e79b7a2c44785e5407ff9e81c30cd4))

- Utiliser libembedding.dll sur Windows pour l'export des symboles
  ([`b05b83e`](https://github.com/dorel14/libembedding-ng/commit/b05b83e71b7aa18d32f315a155a0bf8ec2a62f43))

- Utiliser shell:bash pour l'installation du wheel sur Windows
  ([`79d251d`](https://github.com/dorel14/libembedding-ng/commit/79d251d0ed63e701cd7873d5041ebe83fa377556))

- **release**: Transmettre les paramètres de matrice au workflow réutilisable
  ([`4869c7b`](https://github.com/dorel14/libembedding-ng/commit/4869c7b8a21b2b6a7a09409c2ed6c52f17e18ffd))

Les paramètres `os` et `python` définis dans la matrice sont désormais explicitement passés au
  workflow réutilisable `build-wheel.yml` via la clause `with`, ce qui permet au job de construction
  de recevoir correctement les valeurs de la stratégie matricielle.

- **workflows**: Réorganiser les workflows de publication et de construction
  ([`3937f45`](https://github.com/dorel14/libembedding-ng/commit/3937f45ea9929619f357c156316b43411b015341))

Les workflows `publish.yml` et `semantic-release.yml` ont été supprimés et remplacés par un nouveau
  workflow `release.yml` unifié. Le workflow `build-wheel.yml` a été mis à jour pour installer
  `curl` sur macOS, spécifier `shell: bash` pour l'étape de construction et rendre les arguments
  CURL optionnels dans la configuration CMake. Le fichier `CMakeLists.txt` a également été modifié
  pour supprimer l'appel à `find_package(ONNXRuntime REQUIRED)` dans la branche non-Windows, la
  dépendance ONNX Runtime étant désormais gérée différemment.

### Documentation

- Ajout de la documentation Python en français et en anglais
  ([`8fc972c`](https://github.com/dorel14/libembedding-ng/commit/8fc972c12a9fe5ad4c7d2e6d091bb6db7069acd2))

### Features

- **api**: Ajouter le chargement local de modèles et l'introspection en temps réel
  ([`3874b62`](https://github.com/dorel14/libembedding-ng/commit/3874b62e139f14fb0a4511a75e903c7c82d726c3))

Ajoute la capacité de charger des modèles ONNX locaux à partir de répertoires contenant model.onnx +
  tokenizer.json, pour tous les types d'embeddings (texte, sparse, image, reranker) :

- Ajoute lembed_*_create_from_path() dans l'API C pour chaque type de contexte, lisant les fichiers
  via read_file_to_string() et chargeant directement en mémoire sans téléchargement - Ajoute
  lembed_model_desc_t et lembed_*_desc() pour l'introspection en temps réel : nom, dimension,
  max_length, pooling, threads, batch_size, provider et device_id - Ajoute lembed_*_model_name() et
  lembed_*_max_length() comme accesseurs simplifiés - Ajoute le paramètre offline aux options et à
  lembed_ensure_*_model() pour forcer l'utilisation du cache uniquement - Ajoute batch_size et
  pooling aux options texte/image pour les modèles locaux sans config.json - Remplace num_threads
  par threads dans l'API Python (avec deprecation warning) et ajoute batch_size, offline, dim,
  pooling - Ajoute list_supported_models() et info() aux classes Python - Met à jour la version à
  0.2.0 et le nom du paquet PyPI en libembedding-ng

BREAKING CHANGE: le paramètre num_threads est remplacé par threads dans l'API Python TextEmbedding,
  ImageEmbedding, SparseTextEmbedding et Reranker. Le paramètre batch_size dans embed() accepte
  désormais None au lieu de 0.

- **build**: Ajouter le support Windows et les dépendances tierces
  ([`457f2fc`](https://github.com/dorel14/libembedding-ng/commit/457f2fcf02ed2af0021642614d1497618f2c23d0))

Ajoute le support complet de Windows dans le système de build CMake : - Définit la politique CMP0144
  et NOMINMAX pour éviter les conflits min/max de l'API Windows - Configure ONNX Runtime et CURL
  comme dépendances tierces locales pour Windows, avec création de cibles importées - Passe la
  bibliothèque en mode SHARED (DLL) sous Windows pour le support Python cffi, tout en conservant le
  mode INTERFACE pour Linux/macOS - Corrige S_ISREG, ssize_t et le stockage thread-local pour la
  compatibilité MSVC dans les en-têtes de la bibliothèque - Convertit les chemins UTF-8 en UTF-16
  pour l'API CreateSession d'ONNX Runtime sous Windows

Ajoute curl et onnxruntime comme dépendances tierces dans third_party/ et ajoute la méthode
  list_supported_models() dans l'API Python TextEmbedding.

- **stats**: Ajouter les statistiques d'exécution et les fonctions de similarité
  ([`9873f18`](https://github.com/dorel14/libembedding-ng/commit/9873f182b2bfcf618d3849300eb9214ab0fca86b))

Ajoute un système complet de statistiques en temps réel pour tous les contextes d'embedding (texte,
  sparse, image, reranker) ainsi que des fonctions de similarité vectorielle exposées dans l'API
  Python :

- Ajoute lembed_stats_t dans types.h avec texts_embedded, batches_run et avg_latency_ms - Ajoute
  lembed_*_stats() dans l'API C pour chaque type de contexte, avec chronométrage via std::chrono
  dans les fonctions embed/rerank - Ajoute lembed_version() dans config.h pour exposer la version à
  l'API C et Python - Ajoute lembed_text_embedding_embed_stream() et la méthode Python
  embed_stream() pour un traitement par lots avec rendu incrémental - Ajoute similarity.h avec
  cosine_similarity, dot_product et euclidean_distance, exposées en Python via similarity.py -
  Ajoute la classe Stats et les accesseurs stats() dans toutes les classes Python (TextEmbedding,
  SparseTextEmbedding, ImageEmbedding, Reranker) - Ajoute list_supported_models() et info() dans
  l'API Python - Ajoute des tests pour les statistiques, la similarité et max_length - Met à jour
  CMakeLists.txt pour compiler test_similarity et copier les DLLs runtime sur Windows

BREAKING CHANGE: la méthode embed() de TextEmbedding accepte désormais batch_size=None au lieu de 0
  pour utiliser la valeur par défaut du contexte.

### Breaking Changes

- **stats**: La méthode embed() de TextEmbedding accepte désormais batch_size=None au lieu de 0 pour
  utiliser la valeur par défaut du contexte.
