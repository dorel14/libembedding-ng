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
| `lembed_maxsim(q, nq, d, nd, dim, out_score)` | `lembed_status_t` | Score MaxSim (ColBERT) d'une requête contre un document |
| `lembed_maxsim_batch(q, nq, docs, offsets, num_docs, dim, out_scores)` | `lembed_status_t` | Un score MaxSim par document, dans l'ordre d'entrée |

## MaxSim (interaction tardive)

`lembed_maxsim` renvoie la somme, sur les `nq` vecteurs de la requête, du produit scalaire maximal avec les `nd` vecteurs du document. Les matrices sont row-major (`dim` floats par ligne) ; des entrées normalisées L2 donnent le MaxSim cosinus. `nq == 0` ou `nd == 0` donne un score de 0. Aucune allocation, aucun état, ré-entrant.

`lembed_maxsim_batch` score une requête contre plusieurs documents concaténés : le document `i` occupe les lignes `offsets[i]` à `offsets[i+1]-1` de `docs` (`offsets` a `num_docs + 1` entrées, `offsets[0] == 0`, croissantes). Ni tri, ni top-K, ni index. Retour `LEMBED_ERROR_INVALID_ARGUMENT` pour un pointeur de sortie NULL, un `q`/`d` NULL avec un compte non nul, un compte négatif, `dim <= 0` ou des offsets invalides.

```c
float score;
lembed_status_t st = lembed_maxsim(q, nq, d, nd, dim, &score);
```

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
