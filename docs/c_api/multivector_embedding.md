---
title: Multi-vecteurs (ColBERT) — C API
nav_order: 22
---

# Multi-vecteurs (ColBERT) C API

`include/libembedding/multivector_embedding.h` : un vecteur **par token** au lieu d'un vecteur par texte (late interaction). Backend ONNX Runtime uniquement. Le scoring se fait avec `lembed_maxsim` / `lembed_maxsim_batch` ([Similarité](similarity.html)). Cette API calcule des embeddings : pas d'index, pas de tri, pas de top-K.

## Référence

| Fonction | Description |
|---|---|
| `lembed_multivector_options_default()` | Options par défaut |
| `lembed_multivector_create(options, &mv)` | Charge le modèle depuis `options->model_dir` (répertoire local) |
| `lembed_multivector_embed(mv, texts, n, mode, &out)` | Embed `n` textes en mode `LEMBED_MV_QUERY` ou `LEMBED_MV_DOCUMENT` |
| `lembed_multivectors_free(&out)` | Libère `data` et `offsets`, remet la struct à zéro (idempotent) |
| `lembed_multivector_dim(mv)` | Dimension des vecteurs (96 pour answerai-colbert-small-v1) |
| `lembed_multivector_desc(mv)` | Configuration effective (`lembed_model_desc_t`, possédée par le handle) |
| `lembed_multivector_backend_name(mv)` | `"onnxruntime"` |
| `lembed_multivector_stats(mv, &st)` | Compteurs d'usage |
| `lembed_multivector_free(mv)` | Libère le handle |

## Résultat

`lembed_multivectors_t` est ragged : les vecteurs du texte `i` sont `data[offsets[i] * dim .. offsets[i+1] * dim)`, `offsets` a `num_docs + 1` entrées, `offsets[0] == 0` et `offsets[num_docs] == total_vectors`.

## Pipeline

Pour chaque texte : tokenisation (avec tokens spéciaux), troncature à `maxlen - 1`, insertion du marqueur (`[unused0]` requête, `[unused1]` document) à l'index 1, puis :

- **requête** : par défaut, **expansion** : complétée par `[MASK]` jusqu'à `query_length`, attention à 0 sur ces pads (sauf `attend_to_expansion = 1`), **tous** les vecteurs sont conservés ;
- **document** : les vecteurs des tokens de ponctuation (skiplist) sont retirés ; `[CLS]` et `[SEP]` sont conservés ;
- seules les entrées présentes dans le graphe sont fournies (`input_ids`, `attention_mask`, `token_type_ids` si le graphe l'attend) ;
- `normalize = 1` : L2 par vecteur (idempotent : le graphe officiel est déjà normalisé).

## Conventions du modèle (`onnx_config.json`, ou `colbert_config.json`)

Champs lus, tous optionnels : `query_prefix_id`, `document_prefix_id`, `mask_token_id`, `query_prefix` / `document_prefix` (marqueur en chaîne, résolu dans le vocabulaire : `"[Q] "` est tronqué à `"[Q]"`), `query_length`, `document_length`, `attend_to_expansion_tokens`, `skiplist_words`, `embedding_dim`.

Défauts si absents : marqueurs `[unused0]` / `[unused1]`, `[MASK]`, `query_length` 32, `document_length` 300, attention des pads à 0, skiplist `string.punctuation`. Les options `query_maxlen` / `doc_maxlen` (> 0) priment sur le modèle.

`do_query_expansion` n'est **pas** consulté : le fichier officiel d'answerai-colbert-small-v1 le déclare à `false`, alors que la référence (pylate) étend ses requêtes. Mesure : cosinus par token minimal 0,999995 avec expansion (pads masqués), 0,94 avec pads attendus, nombre de vecteurs différent sans expansion. Désactiver avec `query_expansion = 0`.

Le fichier ONNX est cherché dans l'ordre : `model.onnx`, `vespa_colbert.onnx`, `onnx/model.onnx`. Marqueur en chaîne introuvable dans le vocabulaire des tokens du modèle : erreur (TODO : tokens ajoutés hors `model.vocab`, par exemple ModernBERT).

## Erreurs

| Cas | Statut |
|---|---|
| Argument NULL / vide / négatif | `LEMBED_ERROR_INVALID_ARGUMENT` |
| Chemin `.gguf` | `LEMBED_ERROR_UNSUPPORTED` |
| Provider indisponible | `LEMBED_ERROR_UNSUPPORTED` |
| Sortie du graphe ≠ `embedding_dim` (projection absente) | `LEMBED_ERROR_UNSUPPORTED` |
| Répertoire ou fichier absent, configuration invalide, marqueur non résolu | `LEMBED_ERROR_IO` |

Le détail est dans `lembed_last_error()`. Les appels sur un même handle sont sérialisés.

## Exemple

```c
lembed_multivector_options_t o = lembed_multivector_options_default();
o.model_dir = "models--answerdotai-answerai-colbert-small-v1";
lembed_multivector_t* mv = NULL;
if (lembed_multivector_create(&o, &mv) != LEMBED_OK) { /* lembed_last_error() */ }

const char* docs[] = { "Paris is the capital of France.", "Bananas grow in the tropics." };
const char* q[] = { "capital of France" };
lembed_multivectors_t dv, qv;
lembed_multivector_embed(mv, docs, 2, LEMBED_MV_DOCUMENT, &dv);
lembed_multivector_embed(mv, q, 1, LEMBED_MV_QUERY, &qv);

float scores[2];
lembed_maxsim_batch(qv.data, (int)qv.total_vectors, dv.data, dv.offsets, dv.num_docs, dv.dim, scores);

lembed_multivectors_free(&dv);
lembed_multivectors_free(&qv);
lembed_multivector_free(mv);
```

## Fidélité

Référence pylate 1.6.0, `answerai-colbert-small-v1`, 3 requêtes et 3 documents (`tests/data/colbert/`) : même nombre de vecteurs, cosinus par token minimal 0,999995 (requêtes et documents). Test : `tests/test_multivector_fidelity.cpp` (ignoré si le modèle est absent).
