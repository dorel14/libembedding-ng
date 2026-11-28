---
title: Quantification
nav_order: 8
---

# Quantification

> **English:** [Quantization](en/quantization.html)

Cette page décrit les modes de quantification que la bibliothèque connaît, la façon d'en choisir un, ce que fait le mode `auto`, et les types de tenseurs acceptés par le runtime sparse GGUF.

**Règle du projet : ne jamais supposer qu'une quantification est plus rapide. Mesurer** (voir [Mesurer](#mesurer)).

## Modes disponibles

Énumération C `lembed_quantization_t` (`include/libembedding/types.h`) :

| Constante C | Nom Python | Valeur | Sens |
|---|---|---|---|
| `LEMBED_QUANTIZATION_NONE` | `"none"` | 0 | Poids FP32 |
| `LEMBED_QUANTIZATION_STATIC` | `"static"` | 1 | INT8 statique |
| `LEMBED_QUANTIZATION_DYNAMIC` | `"dynamic"` | 2 | INT8 dynamique |
| `LEMBED_QUANTIZATION_AUTO` | `"auto"` | 3 | Demande de sélection automatique (jamais un état : voir plus bas) |
| `LEMBED_QUANTIZATION_FP16` | `"fp16"` | 4 | Poids float16 |

`FP16` n'est pas une quantification INT8 : il divise par deux le fichier et la mémoire résidente, mais sur un CPU sans arithmétique FP16 native il reconvertit en FP32 pour calculer. Il échange donc du débit contre de la taille (commentaire de `types.h`).

Les valeurs numériques font partie de l'ABI publique ; `FP16` a été ajouté en dernier pour ne pas les décaler.

## Une variante est un autre fichier

Dans le registre, chaque variante quantifiée est **sa propre entrée** avec son propre `model_file`, par exemple `onnx/model_quantized.onnx` à côté du fichier FP32 (`include/libembedding/model_registry.h`). Choisir un mode de quantification revient donc à choisir d'autres poids, pas une autre configuration de session.

- Le champ `quantization` de chaque entrée du registre dit quel mode elle implémente. En Python, `list_text_models()` renvoie des `ModelInfo` dont l'attribut `quantization` vaut `"none"`, `"static"`, `"dynamic"`, `"auto"` ou `"fp16"` (`python/src/libembedding/models.py`).
- Tous les modèles n'ont pas toutes les variantes. `lembed_find_text_model_variant(model_name, quantization)` renvoie l'index de l'entrée, ou `-1` si le modèle n'a pas ce mode ; l'erreur thread-local nomme alors le mode demandé et ceux que le modèle possède (`model_registry.h`).
- La correspondance se fait sur le nom canonique du registre (`model_name`), pas sur un identifiant HuggingFace arbitraire.

## Choisir un mode

### Python

```python
from libembedding import TextEmbedding

# Mode explicite : charge la variante correspondante du registre
emb = TextEmbedding("BAAI/bge-small-en-v1.5", quantization="dynamic")

# Sélection automatique (mesure, puis décision mise en cache)
emb = TextEmbedding("BAAI/bge-small-en-v1.5", preferred_quantization="auto")
```

(`python/src/libembedding/text_embedding.py`.)

- `quantization=` : `None` (défaut) garde le mode par défaut du modèle. Les valeurs acceptées sont les clés de `_QUANTIZATION_ENUM` : `"none"`, `"static"`, `"dynamic"`, `"auto"`, `"fp16"` (insensible à la casse). Toute autre valeur lève `ValueError`. Note : la docstring de `TextEmbedding` ne cite que `"none"`, `"static"`, `"dynamic"`.
- `preferred_quantization=` : `None` par défaut ; accepte les mêmes noms, `"auto"` déclenche la sélection automatique.
- `TextEmbedding` (et `TextEmbeddingPool`, `Reranker`) acceptent `quantization=`.
- Si le modèle n'a pas de variante pour le mode demandé, la création échoue avec `LEMBED_ERROR_MODEL_NOT_FOUND`, soit `ModelNotFoundError` en Python.

### C

Le champ `quantization` de `lembed_text_options_v2_t` (type `lembed_quantization_t`) est lu par `lembed_text_embedding_create_v2()` (`include/libembedding/detail/text_embedding_onnx_impl.hpp`). Pour un mode explicite différent de celui de l'entrée demandée, la fonction résout la variante avant de créer le modèle ; sans variante, elle renvoie `LEMBED_ERROR_MODEL_NOT_FOUND`.

## Que fait `auto`

`lembed_quantization_auto_select()` (`include/libembedding/autotuner.h`) :

1. **Candidats** : seules les variantes réellement présentes dans le registre pour ce modèle sont considérées (`NONE`, `STATIC`, `DYNAMIC`, `FP16`, au plus 4). FP32 est la référence (`detail/quantization_auto_impl.hpp`, `quant_auto_collect_variants`).
2. **Un seul jeu de poids** : si la famille n'a qu'une entrée dans le registre, rien n'est mesuré et le modèle demandé est chargé tel quel.
3. **Mesure** : première fois pour un triplet (modèle, machine, version de la bibliothèque), les variantes sont benchmarkées ; la décision est mise en cache, les appels suivants relisent le cache. Les variantes dont les poids sont absents sont **ignorées, jamais téléchargées**.
4. **Paramètres** : `num_docs` est borné à [16, 256] (0 = 16 par défaut) ; `dry_run = 1` mesure sans écrire le cache.
5. **Règle de décision** (`quant_auto_pick`) :
   - FP32 reste choisi sauf si une variante est au moins **1,05x** plus rapide (`kQuantMinGain`) ;
   - si INT8 dynamique a passé ce seuil, il reste le choix par défaut sauf si une autre variante est au moins **1,15x** plus rapide que lui (`kQuantDynamicPreference`) ;
   - sans mesure de référence FP32 exploitable, FP32 est conservé.
6. **Mémoire** : `file_mb` est la taille des poids sur disque (fichier annexe `_data` compris), pas la mémoire résidente, car le pic RSS est un maximum par processus et ne peut pas être attribué à une variante mesurée après une autre.
7. **Échec** : si la sélection échoue à la création du modèle, l'entrée demandée est conservée ; `auto` est une optimisation, pas une raison de refuser le chargement.

Le résultat (`lembed_quantization_choice_t`, `types.h`) contient le gagnant, l'index d'entrée à charger, le tableau `measured[4]` (débit docs/s, latence ms, `file_mb`), `from_cache` et une phrase `reason`. En Python, le mode retenu est mémorisé sur l'objet (`self._quantization`) et la raison dans `self._quantization_reason`.

Pour le fichier de cache et le format du cache d'autotune, voir [C API — Cache Autotune](c_api/autotune_cache.html).

## Runtime sparse GGUF : types de tenseurs

Le runtime ggml maison pour le sparse GGUF (voir [Runtime sparse GGUF](gguf_sparse_runtime.html)) valide chaque tenseur requis au chargement (`include/libembedding/detail/gguf/gguf_weights.hpp`, `validate_tensors`) :

| Règle | Détail |
|---|---|
| Types acceptés | `F32`, `F16`, `Q8_0` uniquement. Tout autre type est refusé par son nom : `tensor '<nom>' has type <type>; this runtime supports f32, f16 and q8_0`. |
| Vecteurs (1-D) | Biais et poids de LayerNorm doivent être `F32` : `tensor '<nom>' is a vector of type <type>; vectors must be f32`. Les convertisseurs gardent les tenseurs 1-D en F32. |
| `token_type_embd.weight` | Doit être `F32` (`token_type_embd.weight is not F32`). |
| Forme | Vérifiée contre ce que le graphe attend ; sinon `tensor '<nom>' has shape [...], expected [...]`. Un tenseur absent donne `missing tensor '<nom>'`. |

Ces refus sont remontés par `LEMBED_ERROR_UNSUPPORTED` avec le détail dans `lembed_last_error()` (`detail/sparse_text_embedding_gguf_impl.hpp`). Voir [Dépannage](troubleshooting.html).

Cette liste concerne le **sparse GGUF**. Elle ne dit rien des quantifications acceptées par le backend dense llama.cpp, qui lit les GGUF via llama.cpp lui-même.

## Qualité par variante

Qualité de recherche mesurée sur BEIR SciFact (test : 5183 documents, 300 requêtes), similarité cosinus exhaustive, sans mesure de temps. La colonne « Cosinus vs FP32 » est le cosinus moyen entre l'embedding d'un document et celui de l'entrée FP32 du même modèle.

| Modèle | Entrée du registre | Quantification | nDCG@10 | MRR@10 | Recall@100 | Cosinus vs FP32 |
|---|---|---|---:|---:|---:|---:|
| bge-small-en-v1.5 | Xenova/bge-small-en-v1.5 | none | 0,7127 | 0,6822 | 0,9417 | 1,00000 |
| bge-small-en-v1.5 | onnx-community/bge-small-en-v1.5-ONNX | dynamic | 0,7077 | 0,6766 | 0,9467 | 0,99831 |
| bge-small-en-v1.5 | Qdrant/bge-small-en-v1.5-onnx-Q | fp16 | 0,7127 | 0,6822 | 0,9417 | 1,00000 |
| all-MiniLM-L6-v2 | Qdrant/all-MiniLM-L6-v2-onnx | none | 0,6541 | 0,6071 | 0,9317 | 1,00000 |
| all-MiniLM-L6-v2 | Xenova/all-MiniLM-L6-v2 | dynamic | 0,6519 | 0,6077 | 0,9367 | 0,99336 |

Une variante du registre est un autre dépôt, donc potentiellement un autre export du modèle : le tableau mesure l'entrée que l'utilisateur obtient, pas le seul effet du quantificateur. Mesuré sur un i7-1065G7, ONNX Runtime 1.31.0, une seule exécution (pas d'intervalle de confiance : les écarts de l'ordre de 0,005 sur 300 requêtes ne sont pas significatifs).

Reproduire :

```powershell
python benchmarks/quantization/bench_quality.py --models BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2
```

Le script écrit `quality_results.json` et `quality_results.md` dans `benchmarks/quantization/` (option `--out`), avec les métadonnées de la machine.

## Mesurer

Une quantification plus petite n'est pas forcément plus rapide : cela dépend du CPU, du modèle, du batch et de la longueur des séquences. Avant de choisir :

- Mesurer vos propres variantes avec la même méthode (un sous-process par configuration, warmup séparé du run chronométré, médiane sur au moins 10 tours, écart inférieur à 5 % = bruit) : voir [Méthodologie benchmark](benchmark_methodology.html), qui impose aussi les métadonnées (machine, modèle, corpus, date).
- Laisser `auto` mesurer pour vous (ci-dessus), ou utiliser les outils de [Performance](performance_tuning.html).
- Ne pas réutiliser un chiffre mesuré sur une autre machine : le cache `auto` est d'ailleurs indexé par empreinte matérielle.
