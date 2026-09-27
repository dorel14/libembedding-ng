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
| `lembed_cache_get_copy(cache, text, out_vec, capacity, out_dim)` | `int` (copié=1, miss=0, buffer trop petit=-1) | **Recommandé** — copie sous verrou dans un tampon fourni par l'appelant |
| `lembed_cache_get(cache, text, out_vec, out_dim)` | `int` (hit=1, miss=0) | Déprécié — rend un pointeur **emprunté**, invalidé à l'éviction |
| `lembed_cache_put(cache, text, vec, dim)` | `void` | Stocker un embedding |
| `lembed_cache_clear(cache)` | `void` | Vider le cache |
| `lembed_cache_capacity(cache)` | `int` | Capacité du cache |
| `lembed_cache_size(cache)` | `int` | Taille actuelle |
| `lembed_cache_config_default()` | `lembed_cache_config_t*` | Configuration par défaut |
| `lembed_cache_detect_hardware(hw)` | `lembed_status_t` | Détecter le hardware |

## Durée de vie des pointeurs

`lembed_cache_get()` rend **un pointeur interne au cache**, pas une copie. Il reste
valide jusqu'à ce que l'entrée soit écrasée (`lembed_cache_put`), évacée par
pression LRU, vidée (`lembed_cache_clear`) ou libérée (`lembed_cache_free`).
L'appelant doit copier ce dont il a besoin et **ne jamais** appeler `free()` sur
ce pointeur.

Le cache relâchant son verrou avant le retour, une éviction concurrente peut
libérer le tampon pendant que l'appelant le copie. Utilisez
`lembed_cache_get_copy()`, qui copie **sous le verrou** : aucun pointeur
détenu par le cache ne sort de la bibliothèque.

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

// Récupérer un embedding : la copie est faite sous le verrou
int dim = 0;
int rc = lembed_cache_get_copy(cache, "hello world", NULL, 0, &dim);
if (rc == -1) {
    // rc == -1 : la clé existe, `dim` est la taille à allouer
    float *out_vec = (float *)malloc(dim * sizeof(float));
    if (lembed_cache_get_copy(cache, "hello world", out_vec, dim, &dim) == 1) {
        printf("Cache hit, dim=%d, out_vec[0]=%f\n", dim, out_vec[0]);
    }
    free(out_vec);
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
