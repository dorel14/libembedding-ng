---
title: Sélection automatique de modèle — C API
nav_order: 17
---

# Sélection automatique de modèle C API

Ce module permet de sélectionner automatiquement le meilleur modèle et la meilleure configuration selon le matériel disponible.

## Types

| Type | Description |
|------|-------------|
| `lembed_model_selection_t` | Résultat de la sélection de modèle |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Sélectionner un modèle automatiquement |
| `lembed_detect_hardware(hw)` | `lembed_status_t` | Détecter le hardware |

## `lembed_model_selection_t`

| Champ | Type | Description |
|-------|------|-------------|
| `model_code` | `const char*` | Code HuggingFace du modèle |
| `model_name` | `const char*` | Nom du modèle |
| `dim` | `int` | Dimension de l'embedding |
| `workers` | `int` | Nombre de workers |
| `threads` | `int` | Nombre de threads |
| `batch_size` | `int` | Taille de batch |
| `throughput_docs_sec` | `double` | Débit (docs/s) |
| `latency_ms` | `double` | Latence (ms) |
| `memory_mb` | `double` | Mémoire (Mo) |
| `score` | `double` | Score de qualité |

## Cas d'utilisation

| Valeur | Description |
|--------|-------------|
| `"speed"` | Optimisé pour la vitesse |
| `"quality"` | Optimisé pour la qualité |
| `"balanced"` | Compromis (défaut) |

## Exemple

```c
#include <libembedding/model_selector.h>

lembed_model_selection_t result;
lembed_status_t status = lembed_auto_select_model("balanced", &result);
if (status == LEMBED_OK) {
    printf("Model: %s\n", result.model_code);
    printf("Workers: %d, Threads: %d, Batch: %d\n",
        result.workers, result.threads, result.batch_size);
}
```

## Voir aussi

- [Sélection C API](c_api/model_selector.html) — Page dédiée
- [Benchmark C API](c_api/embedding_benchmark.html) — Benchmark avec scoring
