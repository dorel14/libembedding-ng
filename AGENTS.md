# AGENTS.md — Guide des agents pour libembedding

> **Fork** : [dorel14/libembedding](https://github.com/dorel14/libembedding) (fork de [pacifio/libembedding](https://github.com/pacifio/libembedding))  
> **License** : MIT  
> **Version courante** : 1.4.0  
> **Dernière mise à jour** : 2026-09-04

---

## 1. Rôle des agents

Les agents travaillant sur ce dépôt doivent :

- **Maintenir la documentation synchronisée** avec le code (README, docs/, docstrings, commentaires).
- **Archiver** dans `docs/archive/` tout document qui n’a plus à être publié (épuration).
- **Respecter les conventions de code, de build et de packaging** décrites ci-dessous.
- **Ne pas introduire de régression** sur Windows (DLL native), macOS ou Linux.

---

## 2. Structure du dépôt

```
libembedding/
├── include/libembedding/          # API publique C + en-têtes C++ internes
│   ├── libembedding.h             # En-tête parapluie (umbrella header)
│   ├── text_embedding.h           # API C texte dense
│   ├── sparse_text_embedding.h    # API C sparse
│   ├── image_embedding.h          # API C image
│   ├── reranker.h                 # API C reranking
│   ├── similarity.h               # Fonctions de similarité C
│   ├── model_registry.h           # Registre des modèles
│   ├── downloader.h               # Téléchargement de modèles
│   ├── gguf_registry.h            # Registre GGUF (llama.cpp)
│   ├── autotuner.h                # Auto-tuner C
│   ├── unified_benchmark.h        # Benchmark unifié C
│   ├── config.h                   # Version et macros de configuration
│   ├── error.h                    # Codes d’erreur et thread-local error
│   ├── types.h                    # Types opaques, enums, options structs
│   ├── detail/                    # Implémentations C++ internes (include-only)
│   └── cpp/                       # Wrappers C++ (provider, models, pool)
├── python/src/libembedding/       # Bindings Python (cffi)
├── tests/                         # Tests unitaires C++
├── examples/                      # Exemples C++ et Python
├── benchmarks/                    # Benchmarks C++ et Python
├── docs/                          # Documentation GitHub Pages (FR/EN)
│   └── archive/                   # Documentation obsolète à archiver
├── third_party/                   # Dépendances embarquées
│   ├── onnxruntime/               # ONNX Runtime (Windows)
│   ├── stb/                       # stb_image / stb_image_resize
│   ├── cJSON/                     # Parseur JSON
│   └── curl/                      # libcurl (Windows)
├── CMakeLists.txt                 # Build système principal
├── pyproject.toml                 # Config semantic-release (racine)
├── python/pyproject.toml          # Package Python (libembedding-ng)
├── build.sh                       # Script de build Unix
├── run_tests.sh                   # Script de tests Unix
├── run_examples.sh                # Script d’exemples Unix
└── README.md                      # Documentation principale
```

---

## 3. Contraintes de build

### 3.1 Outils requis

| Outil | Version minimale | Usage |
|---|---|---|
| CMake | >= 3.18 | Build C/C++ |
| Compilateur C++ | C++17 (GCC 7+, Clang 5+, MSVC 2017+) | Core |
| Compilateur C | C99 | Core |
| Python | >= 3.9 | Bindings |

Attention nous travaillons sous windows donc utilisation  de powershell 
Les corrections de code doivent être faites en  mode 'edit', je ne tolère aucune modification  via batch ou  python  ou  autre script

### 3.2 Distribution duale (contrainte majeure)

- **Linux / macOS** : bibliothèque **header-only INTERFACE**. L’utilisateur définit `LIBEMBEDDING_IMPLEMENTATION` dans **exactement un** `.cpp` de son projet.
- **Windows** : bibliothèque **SHARED (DLL)** construite par CMake. Les bindings Python utilisent `cffi` pour charger `libembedding.dll`.
- **Python** : la wheel PyPI (`libembedding-ng`) embarque la DLL/so/dylib compilée et toutes les dépendances runtime (ONNX Runtime, libcurl).

### 3.3 Dépendances

| Dépendance | Requise | Notes |
|---|---|---|
| ONNX Runtime >= 1.16 | Oui | Bundlé dans les wheels et `third_party/onnxruntime/` sur Windows |
| llama.cpp | Oui | Backend GGUF. Récupéré via CMake `FetchContent` (v0.3.0). |
| libcurl >= 7.0 | Non | Téléchargement de modèles. Désactivée avec `-DLIBEMBEDDING_NO_DOWNLOAD=ON` |
| cJSON | Oui | Bundlé dans `third_party/` |
| stb_image | Non | Optionnel. Désactivé avec `-DLIBEMBEDDING_NO_IMAGE=ON` |

### 3.4 Options CMake

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBEMBEDDING_BUILD_TESTS=ON \
  -DLIBEMBEDDING_BUILD_EXAMPLES=ON \
  -DLIBEMBEDDING_BUILD_SHARED=OFF \
  -DLIBEMBEDDING_NO_DOWNLOAD=OFF \
  -DLIBEMBEDDING_NO_IMAGE=OFF
```

### 3.5 Contraintes Windows spécifiques

- `NOMINMAX` doit être défini pour éviter les conflits `min`/`max`.
- `/GL` (LTCG) **est volontairement omis** : incompatible avec `WINDOWS_EXPORT_ALL_SYMBOLS`.
- Les DLLs runtime (ONNX Runtime, libcurl, libembedding) sont copiées automatiquement à côté des exécutables via `copy_runtime_dlls()`.
- Les chemins ONNX Runtime peuvent être surchargés via `ONNXRUNTIME_ROOT` ou les variables CMake `ONNXRuntime_INCLUDE_DIR` / `ONNXRuntime_LIBRARY`.

---

## 4. Conventions de code

### 4.1 Style général

- **Langue** : commentaires en anglais ou français (le projet est bilingue). Les agents doivent maintenir la cohérence avec le fichier modifié.
- **License** : chaque fichier source doit contenir `SPDX-License-Identifier: MIT`.
- **Auteur** : les nouveaux fichiers doivent inclure `Auteur: David Orel`.
- **Version** : chaque module `.h` / `.hpp` doit comporter un en-tête avec `Version: X.Y.Z`.

Exemple d’en-tête standard :

```c
/*
 * libembedding - <module>
 * <description courte>
 *
 * Auteur: SoniqueBay Team
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */
```

### 4.2 API C (public)

- **Préfixe** : `lembed_` pour toutes les fonctions et types publics.
- **Opaques handles** : `typedef struct lembed_<type> lembed_<type>_t;`
- **Options structs** : toujours fournir une fonction `lembed_<type>_options_default()`.
- **Create/Destroy** : `lembed_<type>_create()` / `lembed_<type>_free()`.
- **Résultats** : structures allouées par la bibliothèque, libérées par l’appelant via `lembed_*_free()`.
- **Gestion d’erreur** : retourner `lembed_status_t`. Sur échec, appeler `lembed_last_error()` pour le détail thread-local.
- **`extern "C"`** : toutes les déclarations publiques sont encapsulées dans `extern "C" { ... }`.

### 4.3 Implémentations C++ (internes)

- **Inclure dans `detail/`** : les implémentations C++ sont des en-têtes **include-only** inclus quand `LIBEMBEDDING_IMPLEMENTATION` est défini.
- **Guard macro** : chaque implémentation doit avoir sa propre guard (`LIBEMBEDDING_<MODULE>_IMPL_HPP`).
- **Namespaces** : le code interne C++ vit dans `namespace lembed { namespace detail { ... } }`.
- **Inclure le minimum** : les implémentations ne doivent pas inclure d’en-têtes système superflus.

### 4.4 Python

- **Nom de package** : `libembedding-ng` sur PyPI, importé sous `libembedding`.
- **Binding** : `cffi` via `_binding.py`. Le fichier `_cdefs.h` contient les déclarations CFFI.
- **Classes Python** : `TextEmbedding`, `SparseTextEmbedding`, `ImageEmbedding`, `Reranker`, `TextEmbeddingPool`.
- **Exceptions** : `LembedError` (générique), `LlamaError` (llama.cpp).
- **Version** : lue depuis `importlib.metadata` dans `__init__.py`.

### 4.5 Naming

| Élément | Convention |
|---|---|
| Fonctions C | `lembed_<module>_<action>()` |
| Types C | `lembed_<module>_t` ou `lembed_<action>_t` |
| Enums C | `LEMBED_<MODULE>_<VALUE>` |
| Options C | `lembed_<module>_options_t` |
| Headers C | `<module>.h` |
| Implémentations | `detail/<module>_impl.hpp` |
| Python | `PascalCase` pour les classes, `snake_case` pour les fonctions |
| CMake targets | `libembedding` (ALIAS : `libembedding::libembedding`) |

---

## 5. Gestion de la version

- **C API** : `include/libembedding/config.h` définit `LIBEMBEDDING_VERSION_MAJOR/MINOR/PATCH` et `LIBEMBEDDING_VERSION_STRING`.
- **Python** : `python/pyproject.toml` définit `project.version`.
- **semantic-release** : configuré dans le `pyproject.toml` racine. Les releases sont créées automatiquement à partir des commits conventionnels sur `main`.
- **Règle** : toute modification fonctionnelle, de build ou de packaging doit faire évoluer la version selon les règles de semver (ou `major_on_zero` pour 0.x).

---

## 6. Documentation

### 6.1 README.md

- Document principal du projet. Doit rester synchronisé avec les fonctionnalités disponibles.
- Contient : installation, quick start (Python + C/C++), API référence, exemples, benchmarks, architecture.

### 6.2 docs/ (GitHub Pages)

- **Format** : Markdown avec Jekyll + thème `just-the-docs`.
- **Bilingue** : pages FR dans `docs/` et `docs/en/` pour l’anglais.
- **Navigation** : `docs/_data/fr_nav.yml` et `docs/_data/en_nav.yml`.
- **Archivage** : déplacer les docs obsolètes dans `docs/archive/` (ne pas supprimer).

### 6.3 Commentaires de code

- Les headers publics doivent être documentés avec des commentaires Doxygen-compatibles.
- Les fonctions internes complexes doivent avoir des commentaires expliquant le « pourquoi ».

---

## 7. Tests

### 7.1 Exécution

```bash
# Build + tests unitaires (pas de réseau)
./run_tests.sh

# Build + tests unitaires + intégration (téléchargement de modèles)
./run_tests.sh --integration
```

### 7.2 Catégories

| Catégorie | Réseau | Exemples |
|---|---|---|
| Unitaires | Non requis | `test_model_registry`, `test_pooling`, `test_similarity`, `test_local_loading` |
| Intégration | Requis | `test_text_embedding`, `test_sparse_embedding`, `test_reranker`, `test_image_embedding`, `test_introspection` |
| llama.cpp | Requis (si activé) | `test_llama_embedding` |

### 7.3 Contraintes

- Les tests unitaires doivent pouvoir compiler avec `LIBEMBEDDING_NO_DOWNLOAD=ON` quand cela est possible.
- Les tests d’intégration sont désactivés par défaut (`LIBEMBEDDING_INTEGRATION_TESTS=OFF`).
- Sur Windows, les DLLs runtime sont copiées automatiquement à côté des exécutables de test.

---

## 8. Intégration continue et packaging

### 8.1 Python

- **Package** : `libembedding-ng`
- **Build** : setuptools avec `python/pyproject.toml`
- **Wheels** : construites par GitHub Actions (`build-wheel.yml`) sur ubuntu-22.04, macos-14, macos-13, windows-2022.
- **Audit** : `auditwheel` (Linux), `delocate` (macOS), vérification manuelle (Windows).
- **Runtime** : les wheels embarquent ONNX Runtime et libcurl.

### 8.2 GitHub Actions

- Les workflows se trouvent dans `.github/workflows/`.
- Ne pas commiter de secrets. Utiliser `secrets.PYPI_TOKEN` pour la publication.
- Les releases sont créées automatiquement par `semantic-release` ou le workflow `release.yml`.

---

## 9. Règles d’archivage (épuration de la doc)

Les agents doivent :

1. **Déplacer** vers `docs/archive/` tout document qui :
   - Décrit une ancienne architecture ou un ancien workflow.
   - Contient des résultats de benchmark obsolètes.
   - Décrit une fonctionnalité supprimée ou remplacée.
2. **Ne jamais supprimer** de documentation sans la archiver (traçabilité).
3. **Mettre à jour** les liens internes de la documentation principale quand un document est archivé.
4. **Ne pas archiver** le README.md racine ni les fichiers de navigation (`_data/*.yml`).

---

## 10. Pièges à éviter

| Piège | Solution |
|---|---|
| Oublier `LIBEMBEDDING_IMPLEMENTATION` dans un `.cpp` | Sur Linux/macOS, exactement un fichier doit la définir. Sur Windows, la DLL est précompilée. |
| Modifier `config.h` sans mettre à jour `python/pyproject.toml` | Les deux versions doivent être identiques. |
| Ajouter un header sans `extern "C"` | L’API publique doit être compatible C. |
| Désactiver `WINDOWS_EXPORT_ALL_SYMBOLS` sans `.def` | Impossible d’utiliser LTCG sans fichier de définition explicite. |
| Oublier `SPDX-License-Identifier: MIT` | Requis dans tous les fichiers source du projet. |
| Casser la compatibilité `cffi` | Les bindings Python dépendent de symboles exacts. Toute modification de l’API C doit être reflétée dans `python/src/libembedding/_cdefs.h`. |
| Archiver un document toujours référencé | Vérifier les liens avant archivage. |

---

## 11. Checklist avant commit

- [ ] Le code compile en Release et Debug sur au moins une plateforme.
- [ ] Les tests unitaires passent (`./run_tests.sh`).
- [ ] La documentation (README, docs/) est à jour.
- [ ] Les commentaires de code sont présents pour les nouvelles fonctions publiques.
- [ ] `SPDX-License-Identifier: MIT` est présent dans tous les fichiers modifiés.
- [ ] Les versions C (`config.h`) et Python (`python/pyproject.toml`) sont synchronisées.
- [ ] Aucun secret ou fichier sensible n’est ajouté.
- [ ] Les benchmarks obsolètes sont archivés dans `docs/archive/` si nécessaire.

---

## 12. Contacts et ressources

- **Dépôt** : https://github.com/dorel14/libembedding-ng
- **Issues** : https://github.com/dorel14/libembedding-ng/issues
- **PyPI** : https://pypi.org/project/libembedding-ng/
