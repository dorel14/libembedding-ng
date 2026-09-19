---
title: Cache d'autotune — C API
nav_order: 12
---

# Cache d'autotune C API

Ce module gère le cache des résultats d'autotuning (workers, threads, batch_size) avec fingerprinting matériel/logiciel/modèle.

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_tune_cache_clear()` | `lembed_status_t` | Effacer tout le cache de tuning |
| `lembed_tune_cache_path()` | `const char*` | Chemin du fichier de cache |
| `lembed_tune_cache_fingerprint(model, hw, sw, result)` | `lembed_status_t` | Calculer l'empreinte du cache |

## Exemple

```c
#include <libembedding/autotune_cache.h>

// Récupérer le chemin du cache
const char *path = lembed_tune_cache_path();
printf("Cache path: %s\n", path);

// Effacer le cache
lembed_status_t status = lembed_tune_cache_clear();

// Calculer une empreinte
char model[] = "BAAI/bge-small-en-v1.5";
char hw[] = "Intel i7-1065G7";
char sw[] = "Windows 11";
char fingerprint[256];
size_t fingerprint_len = 256;
status = lembed_tune_cache_fingerprint(
    model, hw, sw, fingerprint, &fingerprint_len);
```

## Voir aussi

- [Auto-tuneur C API](autotuner.html) — Auto-tuning complet
- [Worker Auto-Tune C API](worker_autotune.html) — Détection workers llama.cpp
