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

## Voir aussi

- [Modèles](models.html) — Liste des modèles disponibles
- [Registry de modèles GGUF](gguf_registry.html) — Registre GGUF C
