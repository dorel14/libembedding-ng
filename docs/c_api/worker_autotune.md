---
title: Auto-tune workers — C API
nav_order: 19
---

# Auto-tune workers C API

Ce module détecte la configuration optimale de workers et sessions pour le backend llama.cpp.

## Types

| Type | Description |
|------|-------------|
| `lembed_worker_config_t` | Configuration worker optimale détectée |

| Champ | Type | Description |
|-------|------|-------------|
| `optimal_workers` | `int` | Nombre de workers recommandé |
| `optimal_threads` | `int` | Nombre de threads recommandé |
| `physical_cores` | `int` | Cœurs physiques détectés |
| `logical_cores` | `int` | Cœurs logiques détectés |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_detect_optimal_workers()` | `lembed_worker_config_t` | Détecter la configuration worker/session optimale |
| `lembed_recommended_workers_for_model(model_path)` | `int` | Nombre de workers recommandé pour un modèle |

> `lembed_detect_optimal_workers()` ne prend **aucun argument** et retourne une
> structure, pas un `void` : elle inspecte le matériel de la machine. La variante
> qui prend un chemin de modèle est `lembed_recommended_workers_for_model()`, qui
> retourne un simple `int`.

## Exemple

```c
#include <libembedding/worker_autotune.h>

lembed_worker_config_t cfg = lembed_detect_optimal_workers();
printf("workers: %d, threads: %d (physical=%d, logical=%d)\n",
       cfg.optimal_workers, cfg.optimal_threads,
       cfg.physical_cores, cfg.logical_cores);

// Recommandation spécifique à un modèle
int workers = lembed_recommended_workers_for_model("meta-llama/Llama-3-8B");
```

## Voir aussi

- [Auto-tuneur C API](autotuner.html) — Auto-tuning complet
- [Backend llama.cpp C API](llamacpp_backend.html) — Backend llama.cpp
