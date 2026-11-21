---
title: Conventions de l'API
nav_order: 11
---

# Conventions de l'API

> **English:** [API conventions](en/api_conventions.html)

Règles communes à l'API C publique (`include/libembedding/*.h`) et à sa traduction Python. Cette page décrit ce que le code fait ; en cas de doute, les en-têtes font foi.

## Nommage

- Fonctions, types et enums publics portent le préfixe `lembed_` (types : `lembed_<module>_t`, enums : `LEMBED_<MODULE>_<VALEUR>`).
- Les handles sont opaques : `lembed_text_embedding_t`, `lembed_sparse_embedding_ctx_t`, `lembed_image_embedding_t`, `lembed_reranker_t` (`types.h`). Seul un pointeur est manipulé.

## Options : défauts puis surcharge

Chaque famille a une fonction `lembed_<famille>_options_default()` qui renvoie la struct par valeur : `lembed_text_options_default()`, `lembed_sparse_options_default()`, `lembed_image_options_default()`, `lembed_reranker_options_default()`. On part de ce défaut et on ne modifie que les champs voulus ; une struct remplie à la main laisse les champs oubliés à zéro, ce qui n'est pas toujours le défaut (par exemple `backend` du reranker : `0` = `ONNX`, le défaut est `AUTO`).

```c
lembed_reranker_options_t o = lembed_reranker_options_default();
o.num_threads = 4;
lembed_reranker_t* rr = NULL;
lembed_status_t s = lembed_reranker_create(&o, &rr);
```

## Création, libération, propriété

| Objet | Création | Libération |
|---|---|---|
| Contexte texte / sparse / image / reranker | `lembed_<module>_create*()` : renvoie un `lembed_status_t`, le handle sort par `**out` | `lembed_text_embedding_free`, `lembed_sparse_text_embedding_free`, `lembed_image_embedding_free`, `lembed_reranker_free` |
| Cache LRU | `lembed_cache_create()` : renvoie le pointeur (ou `NULL`) | `lembed_cache_free` |
| Résultat dense (`lembed_embeddings_t`) | rempli par `*_embed()` | `lembed_embeddings_free` |
| Résultat sparse (`lembed_sparse_embeddings_t`) | rempli par `*_embed()` | `lembed_sparse_embeddings_free` (libère chaque `indices` / `values`, puis `items`) |
| Résultat reranking (`lembed_rerank_results_t`) | rempli par `*_rerank()` | `lembed_rerank_results_free` |

- Les tampons d'un résultat appartiennent à l'appelant, qui les libère avec la fonction `*_free` correspondante, pas avec `free()` isolément. Ces fonctions remettent les pointeurs à `NULL` et les compteurs à 0 et tolèrent un pointeur `NULL` (`types.h`).
- Les pointeurs d'introspection (`lembed_*_desc()`, `lembed_*_desc_v2()`, `model_name()`) appartiennent au contexte : ils sont invalides après `*_free()`. Copier les champs utiles, ne pas conserver le pointeur (`text_embedding.h`, `sparse_text_embedding.h`).
- Les fonctions `*_backend_name()` et `*_effective_provider()` renvoient une chaîne statique à ne pas libérer.

## Codes de retour et erreur détaillée

Les fonctions faillibles renvoient un `lembed_status_t` (`error.h`) :

| Code | Valeur |
|---|---|
| `LEMBED_OK` | 0 |
| `LEMBED_ERROR_INVALID_ARGUMENT` | 1 |
| `LEMBED_ERROR_OUT_OF_MEMORY` | 2 |
| `LEMBED_ERROR_ONNX_RUNTIME` | 3 |
| `LEMBED_ERROR_TOKENIZER` | 4 |
| `LEMBED_ERROR_DOWNLOAD` | 5 |
| `LEMBED_ERROR_IO` | 6 |
| `LEMBED_ERROR_MODEL_NOT_FOUND` | 7 |
| `LEMBED_ERROR_UNSUPPORTED` | 8 |
| `LEMBED_ERROR_BATCH_SIZE` | 9, réservé : jamais renvoyé par l'implémentation actuelle |
| `LEMBED_ERROR_LLAMA` | 10 |
| `LEMBED_ERROR_CACHE_MISS` | 11 |
| `LEMBED_ERROR_GGUF` | 12, runtime GGUF maison (sparse), distinct de `LLAMA` |

- `lembed_status_message(status)` renvoie une chaîne statique par code.
- `lembed_last_error()` renvoie le détail de la dernière erreur **du thread appelant** (tampon `thread_local` de 1024 octets, tronqué au-delà). Il est écrit lors d'un échec ; un appel réussi ne l'efface pas (seuls les points d'entrée d'autotune le remettent à vide au départ, `detail/autotune_result.hpp`). Lire `lembed_last_error()` juste après un échec, sur le même thread, et ne pas s'en servir pour détecter un succès : c'est le `lembed_status_t` qui fait foi.

## Concurrence

- Les appels sur un **même contexte** texte, sparse ou reranker sont sérialisés par un mutex interne (`call_mutex`) : partager un contexte entre threads est sûr, mais n'apporte pas de parallélisme (`detail/text_embedding_impl.hpp`, `detail/reranker_impl.hpp`, `sparse_text_embedding.h`). Pour paralléliser, utiliser un contexte par thread (`TextEmbeddingPool`).
- Les compteurs de statistiques ont un verrou séparé : lire les stats n'attend pas une inférence en cours.
- Le cache LRU est thread-safe pour ses fonctions, avec la réserve de `lembed_cache_get()` ci-dessous.
- `lembed_last_error()` est par thread, voir plus haut.

## Stabilité de l'ABI

- La disposition (ordre, taille, type des champs) d'une struct publique ne change pas. Un champ nouveau ne s'ajoute pas à une struct existante : il va dans une struct **versionnée** qui embarque l'ancienne en premier champ `base`.
  - `lembed_text_options_v2_t` (`base` + `quantization`), `lembed_reranker_options_v2_t`, `lembed_model_desc_v2_t`, `lembed_stats_v2_t` (`types.h`), `lembed_sparse_tuning_result_v2_t` (`autotuner.h`).
  - Les fonctions associées portent le suffixe `_v2` : `lembed_text_embedding_create_v2()`, `lembed_reranker_create_v2()`, `lembed_sparse_autotune_v2()`, `lembed_*_stats_v2()`, `lembed_*_desc_v2()`.
- Un enum public n'est ni réordonné ni amputé : une valeur retirée de l'usage reste en place. `LEMBED_ERROR_BATCH_SIZE` n'est plus renvoyée mais conservée pour que `LLAMA`, `CACHE_MISS` et `GGUF` ne changent pas de valeur ; `LEMBED_ERROR_GGUF` a été ajouté à la fin (`error.h`).
- Toute modification de l'API C se répercute dans `python/src/libembedding/_cdefs.h` et `libembedding.def`. `tests/test_struct_sizes.cpp` fige les tailles côté C, `python/tests/test_cdefs.py` côté cffi.

## Dépréciations

La dépréciation est documentaire (commentaire dans l'en-tête), sans attribut `deprecated` du compilateur ; ADR-004 liste les retraits prévus pour une version majeure future.

- `lembed_cache_get()` : renvoie un pointeur **emprunté** au cache, qui peut être libéré par un `put`, une éviction ou un `clear` concurrent. Ne jamais le `free()`. Utiliser `lembed_cache_get_copy(cache, text, out_vec, capacity, &out_dim)` : copie sous verrou ; renvoie 1 (succès), 0 (absent ou argument invalide) ou -1 (`capacity` trop petit : `out_dim` donne la taille requise). Voir `embedding_cache.h`.

## Choix du backend

Le backend se choisit par l'extension et le point d'entrée (`.gguf` : llama.cpp ; `.onnx`, identifiant HuggingFace ou nom inconnu : ONNX Runtime). Voir [Routage des modèles](model_routing.html).

Le champ `backend` (`LEMBED_BACKEND_ONNX`, `LEMBED_BACKEND_LLAMACPP`, `LEMBED_BACKEND_AUTO`) est une option officielle **du reranker uniquement** :

- `lembed_reranker_options_t.backend` (défaut `AUTO`) force un backend. En `AUTO`, le reranker essaie ONNX Runtime, puis bascule sur l'équivalent GGUF seulement si les fichiers ONNX sont introuvables (`LEMBED_ERROR_MODEL_NOT_FOUND`) (`detail/reranker_impl.hpp`, `lembed_reranker_create`).
- Python : `Reranker(..., backend="auto" | "onnx" | "llama")`.
- Dans `lembed_text_options_t`, le champ existe mais `lembed_text_embedding_create()` ne fait que vérifier que la valeur est dans `[ONNX, AUTO]` (sinon `LEMBED_ERROR_INVALID_ARGUMENT`), puis l'ignore (`detail/text_embedding_onnx_impl.hpp`). Il n'y a pas de paramètre `backend=` sur `TextEmbedding` ni `SparseTextEmbedding`.

## Correspondance Python

- Chaque statut d'échec devient une exception levée par `check_status()` (`python/src/libembedding/_status.py`). Toutes héritent de `LembedError` et portent `status_code`, `message` (texte de `lembed_status_message`) et `detail` (texte de `lembed_last_error()`).

| Statut C | Exception Python |
|---|---|
| `INVALID_ARGUMENT` | `InvalidArgumentError` |
| `OUT_OF_MEMORY` | `OutOfMemoryError` |
| `ONNX_RUNTIME` | `OnnxRuntimeError` |
| `TOKENIZER` | `TokenizerError` |
| `DOWNLOAD` | `DownloadError` |
| `IO` | `IOError` (classe de `libembedding.exceptions`, pas le builtin) |
| `MODEL_NOT_FOUND` | `ModelNotFoundError` |
| `UNSUPPORTED` | `UnsupportedError` |
| `BATCH_SIZE` | `BatchSizeError` |
| `LLAMA` | `LlamaError` |
| `CACHE_MISS`, `GGUF` | `LembedError` (aucune sous-classe dédiée dans `_STATUS_MAP`) |

- `TextEmbedding`, `SparseTextEmbedding`, `ImageEmbedding`, `Reranker` exposent `close()` (appelle le `*_free` C) et le protocole de context manager (`with ... as m:`). Après `close()`, le handle interne est à `None`.

## Formats de sortie

D'après les structs de `types.h` :

- Dense : `lembed_embeddings_t` = tableau `float` à plat de `num_embeddings * dim` éléments (`float*`, 32 bits).
- Sparse : par texte, `indices` en `int32_t*`, `values` en `float*`, `length` éléments.
- Reranking : `lembed_rerank_result_t` = `{ int index; float score; }`.

Les ordres de tri et la normalisation dépendent du modèle et ne sont pas garantis ici : se référer à la documentation du modèle concerné.
