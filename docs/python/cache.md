---
title: Cache d'embeddings LRU
nav_order: 5
---

# Cache d'embeddings LRU

La classe `EmbeddingCache` fournit un cache LRU thread-safe pour les embeddings denses. Elle peut être utilisée de manière autonome ou intégrée automatiquement dans `TextEmbedding` et `Reranker` via le paramètre `cache_size`.

## Construction

```python
from libembedding import EmbeddingCache

cache = EmbeddingCache(capacity=4096, ttl_seconds=0, dim=0)
```

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `capacity` | `4096` | Nombre maximum d'entrées dans le cache |
| `ttl_seconds` | `0` | Durée de vie en secondes (0 = pas d'expiration) |
| `dim` | `0` | Dimension attendue des embeddings (pour validation) |

## Méthodes

| Méthode | Retour | Description |
|---------|--------|-------------|
| `put(text, vec)` | `None` | Stocker un embedding dans le cache |
| `get(text, dim=None)` | `np.ndarray \| None` | Récupérer un embedding par sa clé texte |
| `clear()` | `None` | Supprimer toutes les entrées |
| `stats()` | `dict` | Statistiques du cache (capacity, current_size) |
| `close()` | `None` | Libérer les ressources C sous-jacentes |

## Propriétés

| Propriété | Type | Description |
|-----------|------|-------------|
| `capacity` | `int` | Capacité maximale du cache |
| `current_size` | `int` | Nombre actuel d'entrées |

## Propriété des données renvoyées

`get()` renvoie un tableau NumPy **appartenant à Python** : la copie est faite
par `lembed_cache_get_copy()` pendant que le cache C tient encore son verrou.
Aucun pointeur détenu par le cache n'est exposé, donc une éviction, un
écrasement ou un `clear()` concurrent ne peut pas libérer le tampon pendant la
copie. Le tableau renvoyé reste valide après l'éviction de son entrée, et deux
lectures successives de la même clé renvoient deux copies indépendantes.

Voir [Cache LRU C API](c_api/embedding_cache.html) pour l'équivalent C.

## Contexte manager

```python
with EmbeddingCache(capacity=1024) as cache:
    cache.put("hello", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    vec = cache.get("hello")
```

## Configuration par défaut

```python
from libembedding import cache_config_default

cfg = cache_config_default()
# {"capacity": 4096, "ttl_seconds": 0}
```

## Intégration avec TextEmbedding

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_size=2048,  # Active le cache LRU interne
)
```

## Voir aussi

- [API Python](api_reference.html) — Vue d'ensemble
- [Statistiques runtime](python/stats.html) — Type `Stats`
