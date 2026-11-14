# third_party — dépendances embarquées

Ce dossier contient les dépendances **vendorisées** : elles sont versionnées
comme des fichiers ordinaires du dépôt (aucun sous-module git), afin que le
build n'ait besoin d'aucun téléchargement à la configure.

| Dossier | Rôle | Contenu minimal | Licence |
|---|---|---|---|
| `cJSON.c` / `cJSON.h` / `cJSON/` | Parsing JSON pour la lecture des modèles ONNX | 3 fichiers | MIT |
| `curl/` | Téléchargement des modèles via libcurl. Headers + bibliotheque d'import uniquement ; la DLL runtime n'est **pas** versionnée (`bin/` est gitignoré) | `include/`, `lib/` | curl |
| `stb/` | Décodage et redimensionnement d'images (`LIBEMBEDDING_NO_IMAGE=ON` pour désactiver) | headers | MIT / public domain |
| `onnxruntime/` | Backend ONNX (include + lib) | `include/`, `lib/` | MIT |
| `llama.cpp/` | Backend GGUF : cible `llama` + `ggml`, toujours activée | snapshot **élagué**, voir ci-dessous | MIT |

## llama.cpp : snapshot élagué

llama.cpp n'est vendorisé que pour ce que la cible `libembedding` lie :
`llama` et `ggml` (voir `CMakeLists.txt`, section « llama.cpp »).

Le dossier conserve exactement :

```
CMakeLists.txt  LICENSE  AUTHORS  README.md
cmake/  ggml/  include/  src/  licenses/  vendor/
```

Tout le reste est retiré : outillage Python amont (`gguf-py/`, `conversion/`,
`convert_*.py`), tests, serveur et outils CLI (`tools/`, `app/`, `common/`),
docs et images (`docs/`, `media/`), fixtures de test (`models/*.gguf`,
~77 Mo), CI amont (`.github/`, `ci/`, `.devops/`) et consignes d'agents
amont (`AGENTS.md`, `CLAUDE.md`) — ces dernières n'ont rien à faire dans ce
dépôt et étaient chargées comme instructions de projet.

Deux générateurs sont **conservés** car llama.cpp les invoque via CMake :

```
ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py
ggml/src/ggml-opencl/kernels/embed_kernel.py
```

Ils ne servent que si `GGML_WEBGPU` ou `GGML_OPENCL` est activé ; ces
options sont épinglées à `OFF` dans le `CMakeLists.txt` racine, comme les
autres entrées pointant vers un dossier retiré (`LLAMA_BUILD_COMMON`,
`LLAMA_BUILD_TOOLS`, `LLAMA_BUILD_APP`, `LLAMA_BUILD_MTMD`,
`GGML_BUILD_TESTS`, `GGML_BUILD_EXAMPLES`, `GGML_VIRTGPU`). Une suppression
de ces deux scripts casserait la configure si quelqu'un réactivait un
backend optionnel.

## Mettre à jour llama.cpp

1. Remplacer `third_party/llama.cpp/` par le nouveau tag amont
   (`ggml-org/llama.cpp`, tag `bNNNN`) — l'historique git est conservé par
   `git add -A`, pas par un sous-module.
2. **Réappliquer l'élagage** :
   ```powershell
   pwsh -File scripts/prune_llama_cpp.ps1
   ```
   Le script est idempotent : les fichiers déjà absents sont ignorés, et il
   échoue si `third_party/llama.cpp/CMakeLists.txt` n'est pas au bon endroit.
3. Si le nouveau tag exige une option llama.cpp supplémentaire, l'ajouter
   au bloc d'options du `CMakeLists.txt` racine avec son commentaire.
4. Vérifier : configure + build + tests.
   ```powershell
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIBEMBEDDING_BUILD_TESTS=ON
   cmake --build build --config Release
   ctest --test-dir build -C Release --output-on-failure
   ```
   Une configure réussie prouve qu'aucun `add_subdirectory()` ne pointait
   vers un dossier retiré : CMake échoue bruyamment sur un dossier manquant.
5. Documenter le bump : `README.md` (table des dépendances), `AGENTS.md`
   (§3.3) et régénérer `llm.txt` / `llm_full.txt`.

## Android, iOS et wheels Linux : libcurl et mbedTLS compilés depuis les sources

Ni Android (code NDK) ni le SDK iOS ne fournissent libcurl. Sous Linux, le
libcurl du système ferait greffer par auditwheel une trentaine de bibliothèques
(OpenSSL, krb5, LDAP, libssh, plusieurs sous LGPL). Rien n'est vendorisé ici :
`scripts/build_curl_mbedtls.sh` télécharge les sources officielles, vérifie
leur SHA-256 épinglé, puis les compile en bibliothèques statiques liées dans
`libembedding.so` / `libembedding.dylib`. Il est appelé :

- par `scripts/build_android_curl.sh` (Android : `android.yml`,
  `scripts/build_android_libs.sh`) ;
- par `scripts/build_ios.sh` (`ios.yml`) ;
- en natif pour les wheels Linux (`build-wheel.yml`,
  `scripts/build_wheel_manylinux.sh`) ;
- en cross-compilation armv7l (`scripts/build_wheel_armv7l.sh`, toolchain
  `cmake/toolchains/armv7l-linux-gnueabihf.cmake`).

## ONNX Runtime pour armv7l : compilé depuis les sources

ONNX Runtime ne publie aucun build armv7l et ne prend en charge ARM 32 bits
qu'en cross-compilation. `build-wheel-armv7l.yml` clone le tag de la version
embarquée (1.31.0), le compile avec la toolchain ci-dessus et `--no_telemetry`
(depuis 1.31, un build depuis les sources active la télémétrie 1DS par
défaut), et met le résultat en cache par version et toolchain. Rien n'est
versionné ici.

Les deux textes de licence sont installés dans
`curl-prefix/share/licenses/{curl,mbedtls}/`, d'où
`scripts/collect_wheel_licenses.sh` les copie dans les wheels.

| Composant | Version | Source | Licence |
|---|---|---|---|
| curl (HTTP/HTTPS seulement) | 8.21.0, celle des en-têtes de `curl/` | `curl.se/download`, archive signée par Daniel Stenberg | curl (type MIT) |
| mbedTLS (branche LTS) | 3.6.7 | releases GitHub `Mbed-TLS/mbedtls`, somme SHA-256 publiée | Apache-2.0 (double licence avec GPL-2.0-or-later, utilisé sous Apache-2.0) |

Certificats :

- **Android** : magasin système `/system/etc/security/cacerts`, via
  `CURLOPT_CAPATH` (voir `detail/downloader_impl.hpp`).
- **iOS** : aucun fichier de certificats n'est exposé, et curl ne sait utiliser
  SecTrust qu'avec OpenSSL ou GnuTLS. Le bundle CA de Mozilla publié par
  `curl.se/ca` (fichier daté, SHA-256 épinglé dans `build_ios.sh`, licence
  MPL-2.0) est donc compilé dans la bibliothèque : l'option CMake
  `LIBEMBEDDING_CA_BUNDLE_EMBED=<pem>` génère `lembed_embedded_ca_bundle.h`,
  passé à curl via `CURLOPT_CAINFO_BLOB`. Pour le mettre à jour, changer
  `CA_BUNDLE_DATE` et `CA_BUNDLE_SHA256` dans `build_ios.sh` (la date est
  aussi reprise dans `NOTICE.ca-certificates`, écrit par le même script).

## Attribution

`llama.cpp/LICENSE`, `llama.cpp/AUTHORS` et `llama.cpp/licenses/` sont
conservés : l'attribution MIT fait partie de la redistribution.

---

Auteur: David Orel — SPDX-License-Identifier: MIT
