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
| `lembed_benchmark_hardware_info_t` | Informations hardware |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_benchmark_run(model, backend, corpus, config, result)` | `lembed_status_t` | Exécuter un benchmark |
| `lembed_benchmark_autotune(model, backend, objective, result)` | `lembed_status_t` | Benchmark avec autotune |
| `lembed_benchmark_detect_hardware(hw)` | `lembed_status_t` | Détecter le hardware |

## `lembed_benchmark_metrics_t`

| Champ | Type | Description |
|-------|------|-------------|
| `throughput_docs_sec` | `double` | Débit (docs/s) |
| `latency_p50_ms` | `double` | Latence médiane (ms) |
| `latency_p95_ms` | `double` | Latence P95 (ms) |
| `load_time_ms` | `double` | Temps de chargement (ms) |
| `peak_memory_mb` | `double` | Pic mémoire (Mo) |
| `dim` | `int` | Dimension |
| `num_texts` | `int` | Nombre de textes |
| `num_errors` | `int` | Nombre d'erreurs |

## Corpus types

| Constante | Description |
|-----------|-------------|
| `0` | Court (< 20 tokens) |
| `1` | Moyen (20-80 tokens) |
| `2` | Long (80-200 tokens) |
| `3` | Très long (200+ tokens) |
| `4` | Mélangé |
| `5` | Multilingue |
| `6` | Cas limites |

## Exemple

```c
#include <libembedding/embedding_benchmark.h>

lembed_benchmark_result_t result;
lembed_backend_config_t config = {
    .backend = "llama.cpp",
    .num_threads = 4,
    .batch_size = 32,
    .workers = 1,
};

lembed_status_t status = lembed_benchmark_run(
    "path/to/model.gguf",
    "llama.cpp",
    4,  // corpus type MIXED
    &config,
    &result
);
```

## Voir aussi

- [Benchmark Python](python/benchmark.html) — Version Python
- [Sélection C API](model_selector.html) — Sélection de modèle
