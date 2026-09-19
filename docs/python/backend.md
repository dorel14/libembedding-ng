---
title: Auto-détection de backend
nav_order: 7
---

# Auto-détection de backend

Le module `backend` fournit l'auto-détection du backend (ONNX ou llama.cpp) basée sur le nom du modèle ou le chemin du fichier.

## Référence

| Fonction | Retour | Description |
|----------|--------|-------------|
| `detect_backend(model_name, backend="auto")` | `str` | Détecter le backend à utiliser |
| `backend_to_enum(backend)` | `int` | Convertir backend string en valeur enum C |

## `detect_backend()`

Détecte quel backend utiliser selon le modèle et la préférence utilisateur.

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `model_name` | — | Nom du modèle, chemin ou ID HuggingFace |
| `backend` | `"auto"` | `"auto"`, `"onnx"` ou `"llama"` |

| Retour | Description |
|--------|-------------|
| `"onnx"` | Backend ONNX |
| `"llama"` | Backend llama.cpp |

## Logique de détection

1. Si `backend` explicite → retourne directement
2. Si fichier `.gguf` → `"llama"`
3. Si fichier `.onnx` → `"onnx"`
4. Chemin local → détecte par contenu du répertoire
5. HuggingFace ID → GGUF si modèle GGUF connu, sinon ONNX

## Utilisation

```python
from libembedding import detect_backend

# Auto-détection
backend = detect_backend("BAAI/bge-small-en-v1.5")
print(backend)  # "onnx"

# Forcer llama.cpp
backend = detect_backend("meta-llama/Llama-3-8B", backend="llama")
print(backend)  # "llama"

# Détection par fichier local
backend = detect_backend("/path/to/model.gguf")
print(backend)  # "llama"
```

## Voir aussi

- [API Python](api_reference.html) — Vue d'ensemble
- [Modes d'embedding](c_api/embedding_mode.html) — Sélection de modèle côté C
