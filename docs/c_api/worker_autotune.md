---
title: Auto-tune workers — C API
nav_order: 19
---

# Auto-tune workers C API

Ce module détecte la configuration optimale de workers et sessions pour le backend llama.cpp.

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_detect_optimal_workers(model, n_threads, n_sessions)` | `void` | Détecter workers optimaux |
| `lembed_recommended_workers_for_model(model_name)` | `int` | Workers recommandés pour un modèle |

## Exemple

```c
#include <libembedding/worker_autotune.h>

int n_workers = lembed_recommended_workers_for_model("meta-llama/Llama-3-8B");
printf("Recommended workers: %d\n", n_workers);

// Détection fine
int threads, sessions;
lembed_detect_optimal_workers("meta-llama/Llama-3-8B", &threads, &sessions);
```

## Voir aussi

- [Auto-tuneur C API](autotuner.html) — Auto-tuning complet
- [Backend llama.cpp C API](llamacpp_backend.html) — Backend llama.cpp
