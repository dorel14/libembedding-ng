---
title: Statistiques runtime
nav_order: 8
---

# Statistiques runtime

Le type `Stats` fournit des statistiques d'utilisation pour un contexte d'embedding actif. Chaque classe d'embedding (`TextEmbedding`, `Reranker`, `ImageEmbedding`, `SparseTextEmbedding`) expose une méthode `.stats()`.

## Structure `Stats`

| Champ | Type | Description |
|-------|------|-------------|
| `texts_embedded` | `int` | Nombre total de textes traités depuis la création du contexte |
| `batches_run` | `int` | Nombre total de lots d'inférence ONNX exécutés |
| `avg_latency_ms` | `float` | Temps moyen par appel embed() (en ms) |
| `cache_hits` | `int` | Nombre de hits dans le cache LRU (0 si désactivé) |
| `cache_misses` | `int` | Nombre de misses dans le cache LRU (0 si désactivé) |

## Récupération des statistiques

```python
from libembedding import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
model.embed(["hello world", "test text"])

stats = model.stats()
print(f"Textes: {stats.texts_embedded}")
print(f"Lots: {stats.batches_run}")
print(f"Latence moyenne: {stats.avg_latency_ms:.2f} ms")
print(f"Cache hits: {stats.cache_hits}")
print(f"Cache misses: {stats.cache_misses}")
```

## Statistiques avec cache LRU

```python
from libembedding import TextEmbedding, EmbeddingCache

cache = EmbeddingCache(capacity=1024, dim=384)
model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_size=1024)

model.embed(["hello world"])
model.embed(["hello world"])  # Devrait être un cache hit

stats = model.stats()
print(f"Hits: {stats.cache_hits}")   # 1
print(f"Misses: {stats.cache_misses}")  # 1
```

## Voir aussi

- [Cache LRU](python/cache.html) — Classe `EmbeddingCache`
- [API Python](api_reference.html) — Vue d'ensemble
