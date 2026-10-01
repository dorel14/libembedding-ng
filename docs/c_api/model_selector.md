---
title: Sélection automatique de modèle — C API
nav_order: 17
---

# Sélection automatique de modèle C API

Ce module permet de sélectionner automatiquement le meilleur modèle et la meilleure configuration selon le matériel disponible.

## Types

| Type | Description |
|------|-------------|
| `lembed_model_selection_t` | Résultat de la sélection de modèle (déclaré dans `autotuner.h`) |
| `lembed_use_case_t` | Cas d'utilisation |
| `lembed_model_candidate_t` | Candidat modèle |
| `lembed_hardware_info_t` | **Déprécié** — matériel détecté |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Sélectionner un modèle automatiquement |
| `lembed_model_select(logical_cores, ram_mb, use_case, out_selected)` | `int` | Choisir un candidat à partir d'une configuration matérielle donnée |
| `lembed_detect_hardware(out_info)` | `int` | **Déprécié** — détecter le matériel |

> `lembed_detect_hardware()` retourne un **`int`**, pas un `lembed_status_t` : son
> implémentation convertit `lembed_cache_detect_hardware()` et renvoie 0 en cas
> d'échec. Utilisez `lembed_cache_detect_hardware(lembed_cache_hardware_info_t*)`
> à la place — c'est ce que la documentation de l'en-tête recommande.

## `lembed_use_case_t`

| Constante | Description |
|-----------|-------------|
| `LEMBED_USE_CASE_SPEED` | Optimisé pour la vitesse |
| `LEMBED_USE_CASE_QUALITY` | Optimisé pour la qualité |
| `LEMBED_USE_CASE_BALANCED` | Compromis (défaut) |

## `lembed_model_candidate_t`

| Champ | Type | Description |
|-------|------|-------------|
| `model_name` | `char[256]` | Nom du modèle (ex. `"BAAI/bge-small-en-v1.5"`) |
| `dim` | `int` | Dimension d'embedding |
| `max_length` | `int` | Longueur maximale en tokens |
| `pooling` | `int` | `LEMBED_POOLING_CLS` ou `LEMBED_POOLING_MEAN` |
| `estimated_ram_mb` | `int` | RAM estimée (Mo) |
| `estimated_throughput` | `double` | Débit estimé (docs/s) |

## `lembed_hardware_info_t` (déprécié)

| Champ | Type | Description |
|-------|------|-------------|
| `cpu_model` | `char[256]` | Modèle du CPU |
| `physical_cores` | `int` | Cœurs physiques |
| `logical_cores` | `int` | Cœurs logiques |
| `ram_mb` | `int` | RAM totale (Mo) |

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

`lembed_auto_select_model()` prend une **chaîne**, pas l'enum :

| Chaîne | Constante équivalente | Description |
|--------|----------------------|-------------|
| `"speed"` | `LEMBED_USE_CASE_SPEED` | Optimisé pour la vitesse |
| `"quality"` | `LEMBED_USE_CASE_QUALITY` | Optimisé pour la qualité |
| `"balanced"` | `LEMBED_USE_CASE_BALANCED` | Compromis (défaut) |

## Exemple

```c
#include <libembedding/model_selector.h>
#include <libembedding/autotuner.h>

lembed_model_selection_t result;
lembed_status_t status = lembed_auto_select_model("balanced", &result);
if (status == LEMBED_OK) {
    printf("Model: %s\n", result.model_code);
    printf("Workers: %d, Threads: %d, Batch: %d\n",
        result.workers, result.threads, result.batch_size);
}

// Sélection à partir d'un matériel connu, sans sonder la machine
lembed_model_candidate_t chosen;
if (lembed_model_select(8, 16384, LEMBED_USE_CASE_SPEED, &chosen) > 0) {
    printf("Choisi: %s (%d dim)\n", chosen.model_name, chosen.dim);
}
```

## Voir aussi

- [Sélection C API](c_api/model_selector.html) — Page dédiée
- [Benchmark C API](c_api/embedding_benchmark.html) — Benchmark avec scoring
