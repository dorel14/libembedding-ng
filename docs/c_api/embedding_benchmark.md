---
title: Benchmark — C API
nav_order: 20
---

# Benchmark C API

Ce module fournit le benchmark de modèles avec scoring Pareto, contraintes et recommandations.

## Types

| Type | Description |
|------|-------------|
| `lembed_benchmark_result_t` | Résultat d'un benchmark |
| `lembed_backend_config_t` | Configuration backend pour benchmark |
| `lembed_benchmark_metrics_t` | Métriques de benchmark |
| `lembed_benchmark_weights_t` | Poids de scoring (qualité / débit / coût) |
| `lembed_benchmark_constraints_t` | Contraintes dures (plafonds) |
| `lembed_corpus_type_t` | Catégorie de corpus de test |

> Le type de hardware n'est pas `lembed_benchmark_hardware_info_t` : il s'agit de
> `lembed_cache_hardware_info_t`, déclaré dans `autotune_cache.h`, avec
> `lembed_cache_detect_hardware()` et `lembed_cache_detect_software()`.

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_benchmark_run(model_path, backend, corpus_type, config, result)` | `lembed_status_t` | Exécuter un benchmark |
| `lembed_benchmark_compare(onnx_path, gguf_path, corpus_type, results)` | `int` | Même modèle sur les deux backends (0 à 2 résultats) |
| `lembed_benchmark_autotune(model_path, backend, objective, result)` | `lembed_status_t` | Benchmark avec autotune |
| `lembed_benchmark_select_model(model_dir, objective, constraints, weights, result)` | `lembed_status_t` | Contraintes dures puis front de Pareto puis scoring |
| `lembed_benchmark_detect_sessions(model_path, max_sessions, optimal, throughput)` | `lembed_status_t` | Détecter le nombre de sessions optimal |
| `lembed_benchmark_get_corpus(type, out_texts, out_count)` | `lembed_status_t` | Récupérer le corpus de test (statique, ne pas libérer) |
| `lembed_benchmark_default_cache_dir()` | `const char*` | Répertoire de cache GGUF par défaut |
| `lembed_benchmark_profile_weights(obj)` | `lembed_benchmark_weights_t` | Poids normalisés pour un objectif |
| `lembed_benchmark_custom_weights(q, t, c)` | `lembed_benchmark_weights_t` | Poids personnalisés, renormalisés |

> `lembed_benchmark_detect_hardware()` **n'existe pas** : la détection
> matérielle est `lembed_cache_detect_hardware(lembed_cache_hardware_info_t*)`.

## `lembed_benchmark_metrics_t`

| Champ | Type | Description |
|-------|------|-------------|
| `throughput_docs_sec` | `float` | Débit (docs/s) |
| `latency_p50_ms` | `float` | Latence médiane (ms) |
| `latency_p95_ms` | `float` | Latence P95 (ms) |
| `load_time_ms` | `float` | Temps de chargement (ms) |
| `peak_memory_mb` | `float` | Pic mémoire (Mo) |
| `dim` | `int` | Dimension |
| `num_texts` | `int` | Nombre de textes |
| `num_errors` | `int` | Nombre d'erreurs |

> Les métriques sont des **`float`**, pas des `double`.

## Corpus types

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `LEMBED_CORPUS_SHORT` | `0` | Court (< 20 tokens) |
| `LEMBED_CORPUS_MEDIUM` | `1` | Moyen (20-80 tokens) |
| `LEMBED_CORPUS_LONG` | `2` | Long (80-200 tokens) |
| `LEMBED_CORPUS_VERY_LONG` | `3` | Très long (200+ tokens) |
| `LEMBED_CORPUS_MIXED` | `4` | Mélangé |
| `LEMBED_CORPUS_MULTILINGUAL` | `5` | Multilingue |
| `LEMBED_CORPUS_EDGE_CASES` | `6` | Cas limites |

## Exemple

```c
#include <libembedding/embedding_benchmark.h>

lembed_backend_config_t config = {
    .backend = "onnx",     /* char[32], pas un pointeur */
    .num_threads = 4,
    .batch_size = 32,
    .workers = 1,
};

lembed_benchmark_result_t result;
lembed_status_t status = lembed_benchmark_run(
    "path/to/model.onnx",
    "onnx",
    LEMBED_CORPUS_MIXED,
    &config,
    &result
);
```

## Voir aussi

- [Benchmark Python](python/benchmark.html) — Version Python
- [Sélection C API](model_selector.html) — Sélection de modèle
