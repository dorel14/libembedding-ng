---
title: Fonctions de similarité — C API
nav_order: 9
---

# Similarité C API

Ce module fournit des fonctions de similarité natives opérant directement sur des tableaux `float`.

## Référence

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_cosine_similarity(a, b, dim)` | `float` | Similarité cosinus |
| `lembed_dot_product(a, b, dim)` | `float` | Produit scalaire |
| `lembed_euclidean_distance(a, b, dim)` | `float` | Distance euclidienne (L2) |

## Exemple

```c
#include <libembedding/similarity.h>
#include <stdio.h>

float a[] = {0.1f, 0.2f, 0.3f};
float b[] = {0.4f, 0.5f, 0.6f};
int dim = 3;

float sim = lembed_cosine_similarity(a, b, dim);
float dp  = lembed_dot_product(a, b, dim);
float dist = lembed_euclidean_distance(a, b, dim);

printf("Cosine: %f\n", sim);
printf("Dot product: %f\n", dp);
printf("Euclidean: %f\n", dist);
```

## Voir aussi

- [API Python](api_reference.html) — Vue d'ensemble de l'API Python
- [Fonctions de similarité](similarity.html) — Version Python
