---
title: Dépannage
nav_order: 9
---

# Dépannage

> **English:** [Troubleshooting](en/troubleshooting.html)

Symptômes fréquents, cause, correction. Chaque entrée est tirée du code ; quand le code ne permet pas d'en dire plus, la page s'arrête là.

## Lire l'erreur

### C

Toute fonction qui échoue renvoie un `lembed_status_t` (`include/libembedding/error.h`). Deux fonctions donnent le détail :

- `lembed_status_message(status)` : libellé fixe du code (par exemple `"Unsupported operation"`).
- `lembed_last_error()` : détail **thread-local** posé par le dernier appel en échec (tampon de 1024 octets). À lire dans le même thread, juste après l'échec.

```c
lembed_status_t s = lembed_text_embedding_create(&opts, &ctx);
if (s != LEMBED_OK) {
    fprintf(stderr, "%s: %s\n", lembed_status_message(s), lembed_last_error());
}
```

Codes : `LEMBED_ERROR_INVALID_ARGUMENT`, `OUT_OF_MEMORY`, `ONNX_RUNTIME`, `TOKENIZER`, `DOWNLOAD`, `IO`, `MODEL_NOT_FOUND`, `UNSUPPORTED`, `BATCH_SIZE` (réservé, jamais renvoyé), `LLAMA`, `CACHE_MISS`, `GGUF`.

### Python

`check_status()` lève une sous-classe de `LembedError` avec `status_code`, `message` et `detail` (`python/src/libembedding/_status.py`, `exceptions.py`). Le message est `"<message>: <detail>"`.

| Code C | Exception Python |
|---|---|
| `INVALID_ARGUMENT` | `InvalidArgumentError` |
| `OUT_OF_MEMORY` | `OutOfMemoryError` |
| `ONNX_RUNTIME` | `OnnxRuntimeError` |
| `TOKENIZER` | `TokenizerError` |
| `DOWNLOAD` | `DownloadError` |
| `IO` | `IOError` (celle du package, pas la builtin) |
| `MODEL_NOT_FOUND` | `ModelNotFoundError` |
| `UNSUPPORTED` | `UnsupportedError` |
| `BATCH_SIZE` | `BatchSizeError` |
| `LLAMA` | `LlamaError` |
| `CACHE_MISS`, `GGUF` | `LembedError` (pas de sous-classe dédiée dans `_STATUS_MAP`) ; lire `status_code` |

```python
from libembedding import LembedError
try:
    ...
except LembedError as e:
    print(e.status_code, e.message, e.detail)
```

Pour voir ce que la bibliothèque journalise, définir `LIBEMBEDDING_VERBOSE=1` (valeurs vraies : `1`, `true`, `yes`, `on`) **avant** de lancer le programme : la variable est lue une seule fois par processus et les lignes vont sur stderr (`detail/log.hpp`).

## Windows

### `ctest` affiche « Timeout », ou le programme meurt au démarrage avec `0xC0000135`

- **Cause** : `STATUS_DLL_NOT_FOUND`. Une DLL d'exécution manque à côté de l'exécutable ; `ctest` rapporte le process mort comme un timeout.
- **Correction** : lancer l'exécutable directement et lire son code de sortie (`$LASTEXITCODE` sous PowerShell) pour confirmer. Les DLL runtime (ONNX Runtime, libcurl, libembedding) sont copiées à côté des exécutables par `copy_runtime_dlls()` ; vérifier qu'elles y sont (`AGENTS.md` §3.5).

### `ONNX Runtime API version N is not supported by the loaded onnxruntime library (version X)`

- **Cause** : la bibliothèque a été compilée contre des en-têtes ONNX Runtime plus récents que le `onnxruntime.dll` effectivement chargé. Le commentaire du code cite le cas d'une copie de `System32` qui masque la nôtre (`detail/onnx_session_impl.hpp`, `ort_api()`).
- **Correction** : faire charger le `onnxruntime.dll` fourni avec la bibliothèque (celui à côté de `libembedding.dll`) plutôt que celui du système. Le message indique les deux versions (`N` = version attendue, `X` = version chargée).
- En Python, `_binding.py` précharge les DLL dépendantes (dont `onnxruntime.dll`) depuis les répertoires de la bibliothèque et enregistre ces répertoires pour la résolution des DLL. Si l'erreur apparaît quand même, vérifier quelle bibliothèque est chargée (`LIBEMBEDDING_LIBRARY`, ci-dessous).

### Le CMake s'arrête avec `libcurl runtime DLL not found`

- **Cause** : sous Windows, `libembedding.dll` est liée à libcurl et la DLL (`libcurl-x64.dll`) n'est jamais versionnée dans le dépôt.
- **Correction** (message de `CMakeLists.txt`) : une des options suivantes.
  - `pwsh -File scripts/fetch_windows_libcurl.ps1` (installe le runtime dans `third_party/curl/bin/`) ;
  - `-DCURL_LIBRARY=<vcpkg>\installed\x64-windows\lib\libcurl.lib` ;
  - `-DCURL_RUNTIME_DLL=<chemin>\libcurl-x64.dll` ;
  - `-DLIBEMBEDDING_NO_DOWNLOAD=ON` (aucun code de téléchargement) ;
  - `-DLIBEMBEDDING_CURL_STATIC=ON` (runtime géré par l'appelant).

## Python : quelle bibliothèque native est chargée ?

- `LIBEMBEDDING_LIBRARY=<chemin>` force le fichier à charger ; si le chemin n'existe pas, `OSError: LIBEMBEDDING_LIBRARY points to a missing file: ...` (`_binding.py`).
- Dans un dépôt source qui contient plusieurs builds, un avertissement `Loading <lib>, which is older than the build output <lib2>` signale qu'une copie plus récente existe ; définir `LIBEMBEDDING_LIBRARY` ou supprimer la copie périmée.
- Sous macOS, `_binding.py` ajoute les répertoires de la bibliothèque à `DYLD_LIBRARY_PATH` avant le `dlopen` (`LD_LIBRARY_PATH` sous Linux). Si le chargement de `libembedding.dylib` échoue, commencer par définir `LIBEMBEDDING_LIBRARY` vers le fichier voulu pour isoler le problème.

## Modèles et réseau

### `Model file not in cache (offline mode): <chemin>`

- **Cause** : `offline=True` (Python) / `offline = 1` (C) interdit tout téléchargement et le fichier n'est pas dans le cache. Le statut est `LEMBED_ERROR_DOWNLOAD` (`DownloadError` en Python) (`detail/downloader_impl.hpp`, `downloader.h`).
- **Correction** : charger le modèle une fois avec le réseau, ou pointer vers le bon cache. Ordre de résolution du répertoire de cache : argument `cache_dir` explicite, puis `LIBEMBEDDING_CACHE_DIR`, puis `FASTEMBED_CACHE_DIR`, puis `~/.cache/libembedding` (`detail/downloader_impl.hpp`, `get_cache_dir`). Une build avec `-DLIBEMBEDDING_NO_DOWNLOAD=ON` ne sait lire que le cache existant.
- `LEMBED_ERROR_CACHE_MISS` n'a rien à voir avec ce cas : il concerne le **cache d'autotune** (`autotune_cache.h`).

### `LEMBED_ERROR_UNSUPPORTED` et un message sur la licence

- **Cause** : certains modèles du registre sont sous une licence non compatible MIT (`jinaai/jina-reranker-v2-base-multilingual`, `onnx-community/embeddinggemma-300m-ONNX`) ; leur chargement est refusé par défaut (`detail/download_manager.hpp`).
- **Correction** : le message l'indique. Définir `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1` après avoir lu la licence, ou passer un chemin local de modèle.

### `ModelNotFoundError` : `Unknown model: <nom>` (Python)

- **Cause** : le nom n'est ni un chemin local, ni un fichier `.gguf`, ni un modèle du registre (`text_embedding.py`, branche ONNX). Côté Python, un identifiant inconnu est rejeté avant tout chargement.
- **Correction** : vérifier le nom avec `list_text_models()` ; voir [Routage des modèles](model_routing.html).

## Fournisseur d'exécution (GPU, etc.)

### `Execution provider X was requested but is not available: ...` (`LEMBED_ERROR_UNSUPPORTED`)

- **Cause** : un fournisseur autre que CPU a été demandé alors que cette build ne peut pas l'attacher (`detail/onnx_session_impl.hpp`, `provider_never_available`) :
  - CUDA : build sans `USE_CUDA` ;
  - CoreML : uniquement sur plateformes Apple ;
  - DirectML : uniquement Windows avec `USE_DML` ;
  - TensorRT : build sans `USE_TENSORRT` ;
  - `llama` / `llamacpp` (valeur 5) : ce n'est pas un fournisseur ONNX Runtime ; pour le backend llama.cpp, charger un fichier `.gguf` ;
  - toute autre valeur : `unknown lembed_execution_provider_t value`.
- **Correction** : utiliser `LEMBED_PROVIDER_CPU` (`provider="cpu"`), comme le dit le message. La vérification se fait avant toute lecture de fichier, sans exception.
- Un fournisseur compilé mais qui échoue à s'attacher à l'exécution lève une erreur au chargement (`apply_provider_config()`). Pour savoir ce qui tourne réellement, voir [Routage des modèles](model_routing.html).

## GGUF refusé au chargement

Le sparse GGUF passe par le runtime ggml maison, qui refuse un fichier avec une raison précise (`LEMBED_ERROR_UNSUPPORTED`, détail dans `lembed_last_error()` ; `UnsupportedError` en Python). Messages (`detail/gguf/gguf_weights.hpp`) :

| Message | Cause | Correction |
|---|---|---|
| `missing tensor '<nom>'` | Un tenseur requis est absent du fichier (voir la [convention GGUF sparse](gguf_sparse_convention.html)) | Utiliser un GGUF qui suit la convention |
| `tensor '<nom>' has type <t>; this runtime supports f32, f16 and q8_0` | Type de tenseur non testé, refusé par son nom | Reconvertir en F32, F16 ou Q8_0 |
| `tensor '<nom>' is a vector of type <t>; vectors must be f32` | Biais / LayerNorm quantifiés | Garder les tenseurs 1-D en F32 |
| `tensor '<nom>' has shape [...], expected [...]` | Forme incompatible avec le graphe | Fichier incorrect pour ce modèle |
| `token_type_embd.weight is not F32` | Table de types de tokens quantifiée | La garder en F32 |

Détails sur les types : [Quantification](quantization.html). Une exception levée pendant le chargement est rapportée en `LEMBED_ERROR_GGUF`.

Autre cas : `lembed_sparse_text_embedding_set_max_batch_tokens()` appelée sur un contexte ONNX renvoie `LEMBED_ERROR_UNSUPPORTED` avec `max_batch_tokens applies to the GGUF runtime only` (`detail/sparse_text_embedding_gguf_impl.hpp`).

En Python, un nom de modèle se terminant par `.gguf` qui n'est pas un fichier existant et ne contient pas de `/` lève `FileNotFoundError: GGUF model not found: '<nom>'` ; avec un `/`, il est interprété comme `<repo>/<fichier>` à télécharger (`text_embedding.py`).
