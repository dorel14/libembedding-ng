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
| `lembed_resolve_gguf_path(model_name, out_path)` | `lembed_status_t` | Résoudre le chemin d'un modèle GGUF |
| `lembed_download_model(model_code, dest_dir, progress)` | `lembed_status_t` | Télécharger un modèle |

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
