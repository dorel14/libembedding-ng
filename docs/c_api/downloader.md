---
title: Téléchargement de modèles — C API
nav_order: 13
---

# Téléchargement de modèles C API

Ce module gère le téléchargement et la résolution des modèles (ONNX et GGUF) depuis HuggingFace ou le cache local.

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_ensure_text_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | S'assurer qu'un modèle texte est en cache |
| `lembed_ensure_sparse_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Idem pour un modèle sparse |
| `lembed_ensure_image_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Idem pour un modèle image |
| `lembed_ensure_reranker_model(model_idx, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Idem pour un reranker |
| `lembed_ensure_gguf_model(repo, filename, cache_dir, progress, offline, out_path)` | `lembed_status_t` | Idem pour un modèle GGUF |
| `lembed_resolve_gguf_path(name_or_path, cache_dir, offline, model_path_out)` | `lembed_status_t` | Résoudre le chemin d'un `.gguf` (chemin, URL ou nom de registre) |
| `lembed_cleanup_model_cache(cache_dir, keep_models, dry_run, deleted_count, freed_bytes)` | `lembed_status_t` | Purger le cache, en gardant une liste de modèles |
| `lembed_cleanup_model_cache_except(cache_dir, active_model_dir, dry_run, deleted_count, freed_bytes)` | `lembed_status_t` | Purger le cache sauf le modèle actif |
| `lembed_free_string(s)` | `void` | Libérer une chaîne allouée par la bibliothèque |

> Il n'existe **pas** de `lembed_download_model()` : le téléchargement passe
> toujours par une fonction `lembed_ensure_*`. Le préfixe `ensure` signifie
> « téléchargement si absent, sinon no-op ».
>
> `lembed_resolve_gguf_path()` est déclarée dans cet en-tête et implémentée dans
> le bloc `LIBEMBEDDING_IMPLEMENTATION`. Elle est bien exportée par
> `libembedding.def` (ligne 156), donc utilisable aussi bien en DLL Windows qu'en
> compilation header-only.

## `lembed_ensure_text_model()`

Garantit qu'un modèle texte est téléchargé et disponible en cache.

| Paramètre | Description |
|-----------|-------------|
| `model_idx` | Index du modèle texte (`lembed_text_model_t`) |
| `cache_dir` | Répertoire de cache (NULL = défaut) |
| `progress` | Afficher la barre de progression (0/1) |
| `offline` | Mode hors-ligne (1) ou téléchargement (0) |
| `out_path` | Chemin résolu du modèle |

## Certificats HTTPS

La résolution des certificats est faite par `find_ca_source()`
(`include/libembedding/detail/downloader_impl.hpp`) :

- **Linux, Android, iOS** : si `SSL_CERT_FILE` ou `CURL_CA_BUNDLE` désigne un
  fichier existant, il est utilisé (`CURLOPT_CAINFO`). Ces variables ne sont
  pas lues sur Windows ni sur macOS, qui gardent le défaut de libcurl.
- **Linux, Android** : sinon, le premier bundle existant parmi
  `/etc/ssl/certs/ca-certificates.crt`, `/etc/pki/tls/certs/ca-bundle.crt`,
  `/etc/ssl/ca-bundle.pem`, `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`,
  `/etc/ssl/cert.pem` ; à défaut, le répertoire Android
  `/system/etc/security/cacerts` (`CURLOPT_CAPATH`).
- **iOS** : sans variable, le bundle CA de Mozilla compilé dans la bibliothèque
  (option CMake `LIBEMBEDDING_CA_BUNDLE_EMBED`, macro
  `LIBEMBEDDING_EMBEDDED_CA_BUNDLE`) est donné à libcurl par
  `CURLOPT_CAINFO_BLOB`.

Le répertoire de cache par défaut est `~/.cache/libembedding`, ou
`$HOME/Library/Caches/libembedding` sur iOS ; `LIBEMBEDDING_CACHE_DIR` et
`FASTEMBED_CACHE_DIR` priment.

Chaque fichier est demandé au plus 3 fois. Une nouvelle tentative suit une
erreur réseau (connexion impossible ou coupée, transfert incomplet), un code
HTTP 429 ou 5xx, ou un transfert resté sans aucun octet reçu pendant
`LIBEMBEDDING_DOWNLOAD_STALL_TIMEOUT` secondes (30 par défaut). Une connexion
lente mais active n'est jamais interrompue. Un 404, un 401 ou une erreur de
certificat échouent tout de suite. Le message d'erreur précise le nombre de
tentatives.

## Limite de taille sur mobile

Sur Android et iOS, un modèle du registre de plus de
`LEMBED_MOBILE_MODEL_LIMIT_MB` (150 Mo) est refusé avant tout téléchargement :
les fonctions `lembed_ensure_*` (texte, sparse, image, reranker) et
`lembed_ensure_gguf_model` retournent `LEMBED_ERROR_UNSUPPORTED`, et
`lembed_last_error()` donne la taille, la limite et les modèles légers du même
type. Un chemin local n'est jamais contrôlé. `lembed_mobile_model_limit_mb()`
retourne la limite active (0 = aucune) et `lembed_model_download_size_mb()` la
taille d'un modèle (-1 si inconnu) ; ces deux fonctions sont déclarées dans
`model_registry.h`. `LIBEMBEDDING_ALLOW_LARGE_MODELS=1` lève la limite,
`LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT=1` l'applique sur un poste de bureau.
Détails : [Plateformes](../platforms.html).

## Voir aussi

- [Modèles](models.html) — Liste des modèles disponibles
- [Registry de modèles GGUF](gguf_registry.html) — Registre GGUF C
