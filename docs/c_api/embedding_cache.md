---
title: Cache LRU — C API
nav_order: 11
---

# Cache LRU C API

Ce module fournit un cache LRU thread-safe pour les embeddings denses en C.

## Types

| Type | Description |
|------|-------------|
| `lembed_cache_t` | Handle opaque du cache |
| `lembed_cache_config_t` | Configuration du cache |
| `lembed_cache_hardware_info_t` | Informations hardware |

## Configuration

| Champ | Défaut | Description |
|-------|--------|-------------|
| `capacity` | `4096` | Capacité maximale |
| `ttl_seconds` | `0` | Durée de vie (0 = pas d'expiration) |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_cache_create(config)` | `lembed_cache_t*` | Créer un cache |
| `lembed_cache_free(cache)` | `void` | Libérer un cache |
| `lembed_cache_get(cache, text, out_vec, out_dim)` | `int` (hit=1, miss=0) | Récupérer un embedding |
| `lembed_cache_put(cache, text, vec, dim)` | `void` | Stocker un embedding |
| `lembed_cache_clear(cache)` | `void` | Vider le cache |
| `lembed_cache_capacity(cache)` | `int` | Capacité du cache |
| `lembed_cache_size(cache)` | `int` | Taille actuelle |
| `lembed_cache_config_default()` | `lembed_cache_config_t*` | Configuration par défaut |
| `lembed_cache_detect_hardware(hw)` | `lembed_status_t` | Détecter le hardware |

## Exemple

```c
#include <libembedding/embedding_cache.h>

// Configuration
lembed_cache_config_t cfg = {
    .capacity = 2048,
    .ttl_seconds = 3600,
};

// Créer le cache
lembed_cache_t *cache = lembed_cache_create(&cfg);

// Stocker un embedding
float vec[] = {0.1f, 0.2f, 0.3f};
lembed_cache_put(cache, "hello world", vec, 3);

// Récupérer un embedding
float *out_vec = NULL;
int out_dim = 0;
int hit = lembed_cache_get(cache, "hello world", &out_vec, &out_dim);
if (hit) {
    printf("Cache hit, dim=%d\n", out_dim);
}

// Statistiques
printf("Capacity: %d, Size: %d\n",
    lembed_cache_capacity(cache),
    lembed_cache_size(cache));

// Nettoyer
lembed_cache_free(cache);
```

## Voir aussi

- [Cache LRU Python](python/cache.html) — Version Python
- [Statistiques runtime](python/stats.html) — Type `Stats`
