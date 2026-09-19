---
title: Fonctions de similarité
nav_order: 4
---

# Fonctions de similarité

Les fonctions de similarité permettent de comparer deux vecteurs d'embedding de manière native, sans coût d'instanciation. Elles opèrent directement sur des tableaux `numpy` 1-D de type `float32`.

## Référence

| Fonction | Description |
|----------|-------------|
| `cosine_similarity(a, b)` | Similarité cosinus entre deux vecteurs |
| `dot_product(a, b)` | Produit scalaire entre deux vecteurs |
| `euclidean_distance(a, b)` | Distance L2 entre deux vecteurs |

## Utilisation

```python
import numpy as np
from libembedding import cosine_similarity, dot_product, euclidean_distance

a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
b = np.array([0.4, 0.5, 0.6], dtype=np.float32)

# Similarité cosinus (0.0 = orthogonal, 1.0 = identique)
sim = cosine_similarity(a, b)

# Produit scalaire
dp = dot_product(a, b)

# Distance euclidienne
dist = euclidean_distance(a, b)
```

## Voir aussi

- [API Python](api_reference.html) — Vue d'ensemble de l'API Python
- [Statistiques runtime](python/stats.html) — Type `Stats` utilisé par `.stats()`
