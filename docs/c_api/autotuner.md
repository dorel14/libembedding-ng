---
title: Auto-tuneur — C API
nav_order: 18
---

# Auto-tuneur C API

Ce module fournit l'auto-tuning complet pour trouver la configuration optimale (workers, threads, batch_size).

## Types

| Type | Description |
|------|-------------|
| `lembed_tuning_result_t` | Résultat de l'autotuning |
| `lembed_unified_tuning_result_t` | Résultat de l'autotuning unifié |
| `lembed_model_selection_t` | Résultat de sélection de modèle |

## Constantes

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `LEMBED_AUTOTUNE_QUICK` | `0` | Mode rapide (5-15s) |
| `LEMBED_AUTOTUNE_FULL` | `1` | Mode exhaustif (30-120s) |

## Fonctions d'autotuning

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_autotune(model_code, mode, out)` | `lembed_status_t` | Auto-tuner un modèle texte |
| `lembed_autotune_custom(model_code, texts, n, mode, out)` | `lembed_status_t` | Auto-tuner avec corpus custom |
| `lembed_autotune_unified(task, model_code, mode, out)` | `lembed_status_t` | Auto-tuner unifié (tous types) |
| `lembed_autotune_clear_cache(model_name)` | `void` | Effacer le cache d'autotune |
| `lembed_reranker_autotune(model_code, mode, objective, out)` | `lembed_status_t` | Auto-tuner un reranker |
| `lembed_reranker_autotune_constrained(model_code, mode, objective, min_tokens, max_latency_ms, out)` | `lembed_status_t` | Auto-tune avec contraintes |
| `lembed_reranker_auto_config(model_code, target_latency_ms, objective, out)` | `lembed_status_t` | Auto-configurer selon latence |
| `lembed_reranker_auto_config_profile(model_code, profile, out)` | `lembed_status_t` | Auto-configurer par profil |
| `lembed_sparse_autotune(model_code, mode, out)` | `lembed_status_t` | Auto-tuner un modèle sparse |
| `lembed_image_autotune(model_code, mode, out)` | `lembed_status_t` | Auto-tuner un modèle image |

## Tâches unifiées

| Constante | Description |
|-----------|-------------|
| `LEMBED_TASK_EMBEDDING` | Tâche d'embedding texte |
| `LEMBED_TASK_RERANKING` | Tâche de reranking |
| `LEMBED_TASK_IMAGE` | Tâche d'embedding image |
| `LEMBED_TASK_SPARSE` | Tâche d'embedding sparse |

## Objectifs

| Constante | Description |
|-----------|-------------|
| `LEMBED_OBJECTIVE_LATENCY` | Minimiser la latence |
| `LEMBED_OBJECTIVE_THROUGHPUT` | Maximiser le débit |
| `LEMBED_OBJECTIVE_BALANCED` | Compromis |
| `LEMBED_OBJECTIVE_MEMORY` | Minimiser la mémoire |

## Exemple

```c
#include <libembedding/autotuner.h>

lembed_tuning_result_t result;
lembed_status_t status = lembed_autotune(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    LEMBED_AUTOTUNE_QUICK,
    &result
);
if (status == LEMBED_OK) {
    printf("Workers: %d, Threads: %d, Batch: %d\n",
        result.workers, result.threads, result.batch_size);
}
```

## Voir aussi

- [Cache d'autotune C API](autotune_cache.html) — Cache fingerprinting
- [Worker Auto-Tune C API](worker_autotune.html) — Détection workers
