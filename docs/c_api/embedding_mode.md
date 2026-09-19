---
title: Modes d'embedding — C API
nav_order: 10
---

# Modes d'embedding C API

Ce module définit les modes de qualité/vitesse pour l'embedding et la correspondance avec les modèles recommandés.

## Enumération `lembed_embedding_mode_t`

| Valeur | Nom | Description |
|--------|-----|-------------|
| `0` | `LEMBED_MODE_FAST` | Mode rapide (quantifié, mini modèle) |
| `1` | `LEMBED_MODE_BALANCED` | Compromis qualité/vitesse |
| `2` | `LEMBED_MODE_QUALITY` | Qualité maximale (FP32, grand modèle) |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_mode_to_string(mode)` | `const char*` | Nom du mode en chaîne |
| `lembed_recommended_model_for_mode(mode)` | `lembed_text_model_t` | Modèle recommandé pour le mode |

## Correspondance modèle/mode

| Mode | Modèle recommandé |
|------|-------------------|
| FAST | Paraphrase-MiniLM-L12-v2 |
| BALANCED | BAAI/bge-small-en-v1.5 |
| QUALITY | BAAI/bge-base-en-v1.5 |

## Exemple

```c
#include <libembedding/embedding_mode.h>

const char *name = lembed_mode_to_string(LEMBED_MODE_BALANCED);
printf("Mode: %s\n", name);  // "balanced"

lembed_text_model_t model = lembed_recommended_model_for_mode(LEMBED_MODE_QUALITY);
// LEMBED_TEXT_BGE_BASE_EN_V15
```

## Voir aussi

- [API Python](api_reference.html) — Vue d'ensemble Python
- [Démarrage](getting_started.html) — Utilisation des modes en Python
