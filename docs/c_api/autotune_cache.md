---
title: Cache d'autotune — C API
nav_order: 12
---

# Cache d'autotune C API

Ce module gère le cache des résultats d'autotuning (workers, threads, batch_size) avec fingerprinting matériel/logiciel/modèle.

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_tune_cache_load(hw, sw, model, backend, entry)` | `lembed_status_t` | Charger une entrée (hit) ou renvoyer `LEMBED_ERROR_CACHE_MISS` |
| `lembed_tune_cache_save(entry)` | `lembed_status_t` | Enregistrer l'entrée et toutes les configs mesurées |
| `lembed_tune_cache_clear()` | `lembed_status_t` | Effacer tout le cache de tuning |
| `lembed_tune_cache_path()` | `const char*` | Chemin du fichier de cache |
| `lembed_tune_cache_key(hw, sw, model, backend, key_out)` | `void` | Calculer la clé d'entrée |
| `lembed_tune_cache_add_config(entry, config)` | `void` | Ajouter une config mesurée à l'entrée |
| `lembed_tune_cache_set_best(entry, idx)` | `void` | Définir l'index de la meilleure config |
| `lembed_cache_detect_hardware(hw)` | `lembed_status_t` | Détecter le matériel (CPU, cœurs, OS, RAM, features) |
| `lembed_cache_detect_software(sw)` | `lembed_status_t` | Détecter les versions libembedding / llama.cpp |

## Format de la clé

La clé est le hash **FNV-1a 64 bits** de l'empreinte complète (CPU, OS, version
libembedding, version llama.cpp, model id, backend), rendu en 16 caractères
hexadécimaux. Les champs lisibles sont trop longs pour un buffer fixe : une clé
tronquée ferait coïncider deux empreintes différentes et renverrait un résultat
mesuré sur une autre machine ou un autre modèle.

Le buffer fourni par l'appelant doit faire au moins `LEMBED_TUNE_CACHE_KEY_SIZE`
octets (17). Ce format a remplacé la concaténation lisible en septembre 2026 :
`LEMBED_TUNE_CACHE_SCHEMA_VERSION` est passé à `2` et les anciennes entrées ne
sont plus atteintes, donc les fingerprints concerned sont remesurés au lieu d'être
lus de travers.

## Exemple

```c
#include <libembedding/autotune_cache.h>

// Récupérer le chemin du cache
const char *path = lembed_tune_cache_path();
printf("Cache path: %s\n", path);

// Effacer le cache
lembed_status_t status = lembed_tune_cache_clear();

// Clé d'entrée pour une empreinte donnée
lembed_cache_hardware_info_t hw;
lembed_cache_software_info_t sw;
lembed_cache_model_info_t model;
lembed_cache_detect_hardware(&hw);
lembed_cache_detect_software(&sw);
snprintf(model.model_id, sizeof(model.model_id), "%s", "BAAI/bge-small-en-v1.5");

char key[LEMBED_TUNE_CACHE_KEY_SIZE];
lembed_tune_cache_key(&hw, &sw, &model, "llama.cpp", key);
printf("Cache key: %s\n", key);
```

## Voir aussi

- [Auto-tuneur C API](autotuner.html) — Auto-tuning complet
- [Worker Auto-Tune C API](worker_autotune.html) — Détection workers llama.cpp
