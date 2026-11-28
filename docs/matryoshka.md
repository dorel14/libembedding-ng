---
title: Matryoshka
nav_order: 4
---

# Troncature Matryoshka

Un modèle entraîné avec *Matryoshka Representation Learning* (MRL) range
l'information la plus utile dans les premières composantes de ses vecteurs.
Garder seulement les `k` premières composantes, puis renormaliser, donne un
vecteur plus petit qui reste utilisable pour la recherche : moins de mémoire,
moins de disque, des produits scalaires plus rapides dans votre index.

libembedding applique cette troncature **après** l'inférence : le modèle tourne
à sa dimension native, puis chaque vecteur est coupé à `k` composantes et
renormalisé en L2. Le coût de l'encodeur ne change pas.

> ⚠️ Sur un modèle **non** entraîné en Matryoshka, le préfixe d'un vecteur n'a
> aucune garantie de qualité. La bibliothèque ne l'interdit pas (elle ne connaît
> pas l'entraînement d'un modèle local), mais les résultats peuvent s'effondrer.

## Modèles du registre entraînés en Matryoshka

| Modèle | Dimension native | Dimensions documentées par l'auteur |
|--------|------------------|--------------------------------------|
| `nomic-ai/nomic-embed-text-v1.5` (FP32 et INT8) | 768 | 768, 512, 256, 128, 64 |
| `mixedbread-ai/mxbai-embed-large-v1` (FP32 et INT8) | 1024 | 1024 → 64 (toute valeur ; 512 recommandé) |
| `google/embeddinggemma-300m` | 768 | 768, 512, 256, 128 |

Les autres modèles du registre (dont `nomic-ai/nomic-embed-text-v1`, la v1 sans
MRL) ne sont pas entraînés en Matryoshka.

## Python

```python
from libembedding import TextEmbedding

model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5", truncate_dim=256)
model.dim          # 256  (taille des vecteurs renvoyés)
model.native_dim   # 768  (taille produite par le modèle)
model.truncate_dim # 256

vecs = model.embed(["search_query: qu'est-ce que Matryoshka ?"])
vecs.shape         # (1, 256), chaque ligne de norme 1
```

- `truncate_dim` doit valoir entre `1` et la dimension native. `0`, une valeur
  négative ou une valeur supérieure à la dimension native lèvent `ValueError`
  à la construction ; un type autre que `int` lève `TypeError`.
- `None` (défaut) conserve le vecteur complet.
- Les vecteurs sont **toujours renormalisés** : cosinus et produit scalaire
  restent équivalents, comme pour un vecteur complet.
- `embed()`, `embed_iter()`, `embed_bytes()`, `embed_stream()` et
  `embed_batched()` appliquent tous la troncature. `TextEmbeddingPool` accepte
  le même paramètre et le transmet à chaque worker.

Le résultat est identique, à 1e-6 près, à la référence numpy
`v[:, :k] / np.linalg.norm(v[:, :k], axis=1, keepdims=True)`
(`python/tests/test_matryoshka.py`).

## API C

```c
#include <libembedding/libembedding.h>

lembed_status_t lembed_embeddings_truncate(lembed_embeddings_t* emb,
                                           int new_dim,
                                           int renormalize);
```

| Paramètre | Description |
|-----------|-------------|
| `emb` | Résultat dense (`lembed_text_embedding_embed`, `lembed_image_embedding_embed`, ou un tampon de l'appelant). Modifié en place. |
| `new_dim` | Nouvelle dimension, `1 <= new_dim <= emb->dim`. |
| `renormalize` | `1` = renormalisation L2 de chaque ligne tronquée (une ligne nulle reste nulle) ; `0` = simple préfixe. |

Comportement :

- Les lignes sont compactées en place (`emb->data` devient un tableau
  `[num_embeddings * new_dim]`) et `emb->dim` est mis à jour.
- Le tampon **n'est pas réalloué** : `emb->data` garde la même adresse,
  `lembed_embeddings_free()` le libère toujours. La fonction ne libère ni
  n'alloue rien, elle accepte donc aussi un tampon possédé par l'appelant.
- `new_dim == emb->dim` est valide : rien n'est déplacé, la renormalisation est
  appliquée si demandée.
- Erreurs : `emb` NULL, `emb->dim <= 0`, `emb->num_embeddings < 0`, `data` NULL
  avec des lignes, ou `new_dim` hors de `[1, emb->dim]` →
  `LEMBED_ERROR_INVALID_ARGUMENT`, détail dans `lembed_last_error()`. En cas
  d'erreur, `emb` n'est pas modifié.

```c
lembed_embeddings_t emb = {0};
if (lembed_text_embedding_embed(ctx, texts, n, 0, &emb) == LEMBED_OK) {
    if (lembed_embeddings_truncate(&emb, 256, 1) != LEMBED_OK)
        fprintf(stderr, "%s\n", lembed_last_error());
    /* emb.dim == 256 */
    lembed_embeddings_free(&emb);
}
```

## Qualité par dimension (nDCG@10, BEIR SciFact)

nDCG@10 sur BEIR SciFact (split test, 5 183 documents, 300 requêtes, `max_length` 512, ONNX fp32). Entre parenthèses, la part de la qualité obtenue à la dimension native.

| Modèle | 1024 | 768 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|---|
| `nomic-ai/nomic-embed-text-v1.5` | — | 0,7032 (natif) | 0,7007 (99,6 %) | 0,6825 (97,1 %) | 0,6425 (91,4 %) | 0,5274 (75,0 %) |
| `mixedbread-ai/mxbai-embed-large-v1` | 0,7389 (natif) | 0,7316 (99,0 %) | 0,7277 (98,5 %) | 0,6932 (93,8 %) | 0,6714 (90,9 %) | 0,5985 (81,0 %) |

En pratique, 512 dimensions conservent plus de 98 % de la qualité pour 1,5 à 2 fois moins de mémoire, et 256 dimensions plus de 93 %. En dessous de 128, la perte devient nette.

Mesuré le 2026-11-24 sur un Intel Core i7-1065G7 (4 cœurs, Windows 11), ONNX Runtime 1.31.0, commit `071b1ce` de `dev`. Pour reproduire :

```bash
python benchmarks/matryoshka/bench_matryoshka_ndcg.py \
  --beir ~/.cache/libembedding/datasets/beir-scifact/scifact \
  --model nomic-ai/nomic-embed-text-v1.5 \
  --model mixedbread-ai/mxbai-embed-large-v1
```
