---
title: Plateformes
nav_order: 7
---

# Plateformes supportées

> **English:** [Platforms](en/platforms.html)

Cette page décrit où libembedding tourne, ce qui est testé en CI, et ce qui
reste à valider sur un appareil physique. Les nombres et noms cités viennent du
code et des workflows ; ce qui n'a pas été exécuté sur du matériel réel est dit
comme tel.

## Matrice

| Plateforme | Livrable | Testé en CI | Jamais exécuté sur matériel physique ? |
|---|---|---|---|
| Windows x64 | wheel PyPI | oui (`windows-2022`) | — |
| Linux x86_64 | wheel PyPI | oui (`ubuntu-22.04`) | — |
| Linux aarch64 | wheel PyPI `manylinux_2_31_aarch64` | oui (runner ARM `ubuntu-22.04-arm`) | oui (Raspberry Pi) |
| macOS arm64 (Apple Silicon, macOS 14+) | wheel PyPI | oui (`macos-15`) | — |
| Android arm64-v8a | wheel `android_24_arm64_v8a` (CPython 3.13) | construit ; tests natifs sur émulateur x86_64 uniquement | oui (appareil arm64) |
| Android x86_64 | wheel `android_24_x86_64` (CPython 3.13) | oui (émulateur) | — |
| iOS 15.1+ (arm64, appareil et simulateur) | bibliothèque native `libembedding.xcframework` (pas de wheel Python) | oui (simulateur) | oui (iPhone) |

Sources : `.github/workflows/ci.yml` (matrice `python-tests`),
`.github/workflows/tests.yml`, `.github/workflows/build-wheel.yml`,
`.github/workflows/build-wheel-android.yml`, `.github/workflows/android.yml`,
`.github/workflows/ios.yml`.

Sur macOS et iOS, ggml est compilé sans Metal (`-DGGML_METAL=OFF` dans
`.github/workflows/build-wheel.yml` et `scripts/build_ios.sh`) : le GPU est
hors périmètre avant la 1.x.

## Linux aarch64 et Raspberry Pi

La wheel `manylinux_2_31_aarch64` s'installe sur un Linux ARM **64 bits** avec
glibc 2.31 ou plus : Raspberry Pi 3/4/5 sous Raspberry Pi OS 64 bits
(bullseye, bookworm), Ubuntu 20.04+, AWS Graviton, Ampere. Le plancher
`manylinux_2_31` est imposé par `scripts/build_wheel_manylinux.sh`
(« above the manylinux_2_31 floor »).

- ggml embarque huit bibliothèques CPU ARM (`ggml-cpu-armv8.0_1` …
  `ggml-cpu-armv9.2_2`) et charge la meilleure à l'exécution ;
  `libembedding.ggml_cpu_features()` indique laquelle.
- Raspberry Pi OS **32 bits (armv7l) n'est pas supporté**.
- La CI construit la wheel dans un conteneur `manylinux_2_28` et la teste sur
  les runners ARM de GitHub. Elle n'a **pas** été testée sur un Raspberry Pi
  physique.
- Les wheels Linux (x86_64 et aarch64) téléchargent par une libcurl liée
  statiquement sur mbedTLS (`scripts/build_curl_mbedtls.sh`, HTTP seulement) :
  elles n'embarquent ni libcurl ni bibliothèque TLS du système.
- Les téléchargements HTTPS utilisent le bundle de certificats du système
  (`/etc/ssl/certs/ca-certificates.crt`, `/etc/pki/tls/certs/ca-bundle.crt`,
  `/etc/ssl/ca-bundle.pem`, `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`,
  `/etc/ssl/cert.pem`, premier trouvé). `SSL_CERT_FILE` ou `CURL_CA_BUNDLE`
  les remplacent (`include/libembedding/detail/downloader_impl.hpp`,
  `find_ca_source()`).

## Android

Wheels `android_24_arm64_v8a` et `android_24_x86_64` pour **CPython 3.13**,
Android 7.0 (API 24) et plus : les builds PEP 738 utilisés par BeeWare et
Chaquopy. PyPI ne fournit ni cffi ni numpy pour Android ; ils viennent de
l'index de Chaquopy :

```bash
pip install --extra-index-url https://chaquo.com/pypi-13.1 libembedding-ng
```

(`PIP_EXTRA_INDEX_URL=https://chaquo.com/pypi-13.1` dans
`.github/workflows/build-wheel-android.yml`.)

**Build.** `scripts/build_android_libs.sh` (étape before-build de
cibuildwheel) construit, par ABI, libembedding, ggml/llama et une libcurl
**statique** au-dessus de **mbedTLS** (`scripts/build_android_curl.sh` →
`scripts/build_curl_mbedtls.sh`) ; ONNX Runtime vient de l'AAR officiel
`onnxruntime-android` 1.31.0 (variable `ONNXRUNTIME_VERSION`). Le niveau d'API
est `ANDROID_API_LEVEL` (24), qui fixe aussi le tag `android_24_*`.

**Téléchargement de modèles.** Android n'a pas de libcurl pour le code natif :
le téléchargement passe par la libcurl statique ci-dessus. Les certificats sont
ceux du système, dans `/system/etc/security/cacerts` (`CURLOPT_CAPATH`),
sauf si `SSL_CERT_FILE` ou `CURL_CA_BUNDLE` désigne un fichier existant.
Sans `HOME`, le cache est dans `$TMPDIR/libembedding`, ou à défaut
`/data/local/tmp/libembedding` (`get_cache_dir()`).

**Tests.**

- `android.yml` construit les bibliothèques natives et les tests pour
  `arm64-v8a` et `x86_64`, puis exécute les tests unitaires sur un
  **émulateur x86_64** (API 30) via `scripts/android_run_tests.sh`, y compris
  `test_text_embedding` (téléchargement réel d'un modèle) et
  `test_embedding_parity`.
- `build-wheel-android.yml` teste la wheel x86_64 sur l'émulateur de
  cibuildwheel (`pytest -m "not network or download_smoke"`). La wheel arm64
  est **construite mais pas exécutée** : un runner x86_64 n'a pas d'émulateur
  arm64 rapide.
- Aucune exécution sur un appareil Android arm64 physique n'a eu lieu.

## iOS

iOS (15.1 et plus, arm64) est livré comme **bibliothèque native**, pas comme
wheel Python.

### Construire

```bash
bash scripts/build_ios.sh <iphoneos|iphonesimulator> <work-dir> [tests]
```

- `iphoneos` : appareil (arm64). `iphonesimulator` : simulateur, arm64
  uniquement (`CMAKE_OSX_ARCHITECTURES=arm64`).
- `tests` (optionnel) : construit aussi les tests unitaires
  (`test_similarity`, `test_pooling`, `test_tokenizer`, `test_model_registry`,
  `test_struct_sizes`, `test_sparse_gguf`, `test_embedding_parity`), à lancer
  sur un simulateur avec `xcrun simctl spawn`.
- Cible de déploiement : `IOS_DEPLOYMENT_TARGET`, **15.1** par défaut (le
  minimum du pod ONNX Runtime).
- ONNX Runtime : archive du pod `onnxruntime-c` **1.30.0**, la plus récente
  publiée pour iOS (1.31.0 ne l'est pas), vérifiée par SHA-256. Les en-têtes
  1.30 sont ceux utilisés à la compilation.
- Le résultat est un `libembedding.dylib` autonome : ONNX Runtime (statique),
  ggml/llama (statique, `BUILD_SHARED_LIBS=OFF`) et libcurl/mbedTLS (statique)
  y sont liés ; seules des bibliothèques et frameworks système restent
  dépendances (vérifié par `otool -L` en fin de script). Le dispatch CPU de
  ggml (`LIBEMBEDDING_CPU_DISPATCH=OFF`) et Metal sont désactivés.
- Une application iOS n'embarque une bibliothèque dynamique que dans un
  framework : le script enveloppe donc le dylib dans
  `<work-dir>/framework-<sdk>/libembedding.framework`. Ce framework contient
  le binaire (install name `@rpath/libembedding.framework/libembedding`), les
  en-têtes `include/libembedding` dans `Headers/` et un module map C dans
  `Modules/` (module `libembedding`, umbrella `libembedding.h`). Il contient
  aussi un `Info.plist` et les textes de licence. Il est signé ad hoc : Xcode le
  re-signe avec l'identité de l'application quand il l'embarque.

### xcframework

Le workflow `.github/workflows/ios.yml` (sur pull request vers `dev`/`main` ou
lancement manuel, runner `macos-15`) construit les deux slices et exécute les
tests unitaires sur un simulateur iPhone. Il assemble ensuite
`libembedding.xcframework` à partir des deux `libembedding.framework`, lance
les tests du paquet Swift `examples/ios` (voir ci-dessous) et publie le
xcframework comme artifact **`libembedding-ios-xcframework`**
(`libembedding-ios.xcframework.zip`). Chaque framework contient les textes de
licence de tout ce qui est lié : `LICENSE`, `LICENSE.onnxruntime`,
`LICENSE.llama.cpp`, `LICENSE.curl`, `LICENSE.mbedtls`, et
`NOTICE.ca-certificates` (bundle de certificats Mozilla, MPL 2.0).

### Utiliser depuis Swift

`examples/ios/` est un paquet SwiftPM qui utilise le xcframework :

- `binaryTarget(name: "libembedding", path: "libembedding.xcframework")` : le
  xcframework n'est pas versionné, copiez-le à côté de `Package.swift` ;
- `import libembedding` donne accès à l'API C (`lembed_*`) ;
- `LibEmbeddingSwift` est une petite enveloppe Swift (`TextEmbedder` : création,
  `embed`, `cosine`, libération dans `deinit`) ;
- `LibEmbeddingSmokeTests` (XCTest) télécharge bge-small, vérifie dimension,
  normalisation et classement, compare au JSON de référence x86_64 et vérifie
  que bge-large est refusé par la limite mobile.

La CI lance `xcodebuild test` sur un simulateur. Dans une application Xcode,
ajoutez `libembedding.xcframework` aux « Frameworks, Libraries, and Embedded
Content » de la cible, en mode **Embed & Sign**. Ce chemin n'a été validé que
par le paquet SwiftPM sur simulateur, pas dans une application signée sur un
iPhone.

### Cache et certificats

- Cache des modèles : `$HOME/Library/Caches/libembedding` (le système peut le
  purger ; un modèle purgé est retéléchargé). `LIBEMBEDDING_CACHE_DIR` et
  `FASTEMBED_CACHE_DIR` priment.
- iOS n'expose aucun fichier de certificats, et la vérification native
  (SecTrust) n'est pas disponible avec mbedTLS. Le bundle CA de Mozilla
  (`cacert-2026-11-05.pem` de curl.se/ca, épinglé par SHA-256 dans
  `scripts/build_ios.sh`) est donc **compilé dans le dylib** :
  `-DLIBEMBEDDING_CA_BUNDLE_EMBED=<pem>` (`CMakeLists.txt`) définit
  `LIBEMBEDDING_EMBEDDED_CA_BUNDLE`, et `downloader_impl.hpp` le donne à
  libcurl par `CURLOPT_CAINFO_BLOB`.
- `SSL_CERT_FILE` ou `CURL_CA_BUNDLE`, s'ils désignent un fichier existant,
  remplacent ce bundle.
- Le bundle embarqué se périme : il faut reconstruire pour en changer la date.

### Tests

La CI exécute les tests unitaires sur un simulateur, dont
`test_embedding_parity`, qui télécharge `bge-small` : cela valide de bout en
bout libcurl statique, mbedTLS et le bundle CA embarqué. Rien n'a été exécuté
sur un iPhone physique.

## Limite de taille des modèles sur mobile

Sur Android et iOS, la mémoire du téléphone est la contrainte : un modèle du
registre dont les fichiers dépassent **150 Mo** est refusé **par son nom, avant
tout téléchargement**, avec une erreur qui liste les modèles légers du même
type.

| Élément | Détail |
|---|---|
| Constante C | `LEMBED_MOBILE_MODEL_LIMIT_MB` (150), `include/libembedding/model_registry.h` |
| Limite active | `int lembed_mobile_model_limit_mb(void)` : 150 sur Android et iOS, 0 (aucune limite) ailleurs |
| Taille d'un modèle | `int lembed_model_download_size_mb(model_code, model_file)` : fichier du modèle plus poids externes `.onnx_data`, en Mo ; `-1` si le couple n'est pas au registre |
| Python | `libembedding.mobile_model_limit_mb()` ; `ModelInfo.download_size_mb` (`-1` si inconnu) |
| Erreur | `LEMBED_ERROR_UNSUPPORTED` côté C, `UnsupportedError` côté Python |

Comportement :

- Le contrôle s'applique aux fonctions `lembed_ensure_text_model`,
  `lembed_ensure_sparse_model`, `lembed_ensure_image_model`,
  `lembed_ensure_reranker_model` et aux constructeurs qui les utilisent
  (`include/libembedding/detail/download_manager.hpp`,
  `check_mobile_model_limit`), et à `lembed_ensure_gguf_model` pour les GGUF du
  registre (`include/libembedding/downloader.h`). Une paire dépôt/fichier hors
  registre n'a pas de taille connue et n'est pas contrôlée.
- Un **chemin local** n'est jamais vérifié.
- `list_text_models()`, `list_sparse_models()`, `list_image_models()` et
  `list_reranker_models()` ne renvoient que les modèles ≤ 150 Mo quand la
  limite est active (`python/src/libembedding/models.py`, `_list_models`).
- `LIBEMBEDDING_ALLOW_LARGE_MODELS=1` lève la limite (priorité sur la
  suivante).
- `LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT=1` applique la limite sur un poste de
  bureau, pour tester le comportement d'une application. Seule la valeur `1`
  compte pour ces deux variables.

Modèles légers, d'après la table de tailles de `model_registry.h`
(`lembed__model_sizes`) : texte, entre autres `all-MiniLM-L6-v2` (90 Mo,
23 Mo quantifié), `all-MiniLM-L12-v2` (133 Mo, 34 Mo quantifié),
`bge-small-en-v1.5` (133 Mo, variantes quantifiées 34 et 66 Mo),
`snowflake-arctic-embed-xs` et `-s`, `bge-small-zh-v1.5` (95 Mo) et les
variantes quantifiées de plusieurs modèles de base (`nomic-embed-text-v1.5`
137 Mo, `gte-base-en-v1.5` 147 Mo, `snowflake-arctic-embed-m` 110 Mo) ; image :
ResNet-50 (94 Mo) et CLIP ViT-B-32 INT8 (89 Mo) ; reranker : Jina Reranker V1
Turbo quantifié (38 Mo). **Aucun modèle sparse n'est léger** (SPLADE++ :
532 Mo). Pour la liste exacte et à jour, appeler `list_*_models()` sur la
plateforme visée ou lire `ModelInfo.download_size_mb`.

```python
import libembedding

print(libembedding.mobile_model_limit_mb())  # 150 sur Android/iOS, 0 ailleurs
for m in libembedding.list_text_models():
    print(m.model_name, m.download_size_mb)
```

## Parité numérique entre plateformes

Une référence calculée sur x86_64 (Windows, ONNX Runtime 1.31.0, poids fp32)
sert de contrôle : `python/tests/data/parity_bge_small_en_v15.json` contient les
embeddings `bge-small-en-v1.5` (dimension 384) de trois textes, avec
`"min_cosine": 0.999`. Chaque plateforme doit donner la même dimension et un
cosinus **≥ 0,999** par texte par rapport à cette référence : le compilateur,
les noyaux SIMD, la version d'ONNX Runtime ou le tokenizer peuvent déplacer les
derniers chiffres, pas le vecteur. Le seuil est lu dans le JSON.

Deux tests lisent ce fichier :

- `tests/test_embedding_parity.cpp` (C++, test d'intégration : il télécharge le
  modèle ; chemin de la référence : `argv[1]`, sinon `LEMBED_PARITY_REF`,
  sinon le chemin compilé) ; exécuté en CI sur l'émulateur Android
  (`scripts/android_run_tests.sh`) et le simulateur iOS (`ios.yml`) ;
- `python/tests/test_parity.py` (pytest, marqueurs `network` et
  `download_smoke`) ; exécuté dans le job de test des wheels sur Linux x64,
  Linux aarch64, macOS et Windows (`ci.yml` → `build-wheel.yml`), et sur
  l'émulateur Android via `download_smoke`.

## Licences des composants embarqués

Chaque wheel contient, dans `.dist-info/licenses/`, la licence MIT du projet
et celles du code natif qu'elle embarque (PEP 639, `license-files` dans
`python/pyproject.toml`) :

- toutes : ONNX Runtime (`LICENSE` et `ThirdPartyNotices.txt` de la version
  embarquée), llama.cpp/ggml, cJSON, stb ;
- Linux et Android : curl et mbedTLS (liés statiquement) ;
- Windows : curl et zlib (DLL de vcpkg) ;
- Linux : libgomp (runtime OpenMP de GCC, GPL-3.0 avec l'exception GCC Runtime
  Library 3.1, qui autorise sa redistribution).

`python/licenses/THIRD_PARTY_NOTICES.md` détaille la répartition. Les textes
stables sont versionnés dans `python/licenses/`. Les autres sont ajoutés au
build par `scripts/collect_wheel_licenses.sh`. La CI vérifie que chaque wheel
les contient (`build-wheel.yml`, étape « Check licence texts »).

Le xcframework iOS porte les siennes dans chaque `libembedding.framework`
(voir [xcframework](#xcframework)).

## Voir aussi

- [Démarrage](getting_started.html) — installation
- [C API — Téléchargement](c_api/downloader.html) — certificats, limite mobile
- [Modèles](models.html) — registre
