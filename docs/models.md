---
title: Modèles
nav_order: 4
---

# Modèles disponibles

libembedding intègre un registre de modèles pré-configurés. Vous pouvez les lister dynamiquement depuis Python :

> **Android et iOS :** seuls les modèles légers (≤ 150 Mo) se chargent par nom,
> et `list_*_models()` ne renvoie qu'eux. Voir
> [Plateformes — limite des modèles sur mobile](platforms.html).

> **Colonne Licence :** la licence est lue sur la carte HuggingFace de chaque modèle
> (relevé du 2026-11-14). Un modèle téléchargé conserve sa propre licence, indépendante
> de la licence MIT de libembedding. Lorsque le dépôt réellement téléchargé est une
> conversion sans étiquette de licence (Xenova, Qdrant, onnx-community…), elle hérite de
> celle du modèle d'origine, indiqué entre parenthèses.

```python
import libembedding

for m in libembedding.list_text_models():
    print(f"{m.model_name:45} dim={m.dim:<5} {m.pooling} quant={m.quantization}")
```

## Modèles de texte (44 modèles)

### Anglais — BGE

| Nom HuggingFace | Dim | Pooling | Quantification | Licence |
|-----------------|-----|---------|----------------|---------|
| `BAAI/bge-small-en-v1.5` | 384 | CLS | none | MIT (BAAI/bge-small-en-v1.5) |
| `BAAI/bge-small-en-v1.5` | 384 | CLS | static | MIT |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | none | MIT |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | static | MIT |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | none | MIT (BAAI/bge-large-en-v1.5) |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | static | MIT (BAAI/bge-large-en-v1.5) |

### Anglais — MiniLM / MPNet

| Nom HuggingFace | Dim | Pooling | Quantification | Licence |
|-----------------|-----|---------|----------------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | none | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | dynamic | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | none | Apache-2.0 (sentence-transformers/all-MiniLM-L12-v2) |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | dynamic | Apache-2.0 (sentence-transformers/all-MiniLM-L12-v2) |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Mean | none | Apache-2.0 (sentence-transformers/all-mpnet-base-v2) |

### Multilingue — E5 / MiniLM / MPNet

| Nom HuggingFace | Dim | Pooling | Quantification | Licence |
|-----------------|-----|---------|----------------|---------|
| `intfloat/multilingual-e5-small` | 384 | Mean | none | MIT |
| `intfloat/multilingual-e5-base` | 768 | Mean | none | MIT |
| `intfloat/multilingual-e5-large` | 1024 | Mean | none | MIT |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | none | Apache-2.0 (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | static | Apache-2.0 |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Mean | none | Apache-2.0 (sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |

### Chinois — BGE

| Nom HuggingFace | Dim | Pooling | Quantification | Licence |
|-----------------|-----|---------|----------------|---------|
| `BAAI/bge-small-zh-v1.5` | 512 | CLS | none | MIT (BAAI/bge-small-zh-v1.5) |
| `BAAI/bge-large-zh-v1.5` | 1024 | CLS | none | MIT (BAAI/bge-large-zh-v1.5) |

### Multilingue — BGE-M3 / ModernBERT

| Nom HuggingFace | Dim | Pooling | Quantification | Licence |
|-----------------|-----|---------|----------------|---------|
| `BAAI/bge-m3` | 1024 | CLS | none | MIT |
| `lightonai/modernbert-embed-large` | 1024 | Mean | none | Apache-2.0 |

### Nomic

| Nom HuggingFace | Dim | Pooling | Quantification | Context | Licence |
|-----------------|-----|---------|----------------|---------|---------|
| `nomic-ai/nomic-embed-text-v1` | 768 | Mean | none | 8192 | Apache-2.0 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | none | 8192 | Apache-2.0 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | dynamic | 8192 | Apache-2.0 |

### GTE (Alibaba)

| Nom HuggingFace | Dim | Pooling | Quantification | Context | Licence |
|-----------------|-----|---------|----------------|---------|---------|
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | none | 8192 | Apache-2.0 |
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | dynamic | 8192 | Apache-2.0 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | none | 8192 | Apache-2.0 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | dynamic | 8192 | Apache-2.0 |

### MixedBread

| Nom HuggingFace | Dim | Pooling | Quantification | Licence |
|-----------------|-----|---------|----------------|---------|
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | none | Apache-2.0 |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | dynamic | Apache-2.0 |

### CLIP

| Nom HuggingFace | Dim | Pooling | Quantification | Note | Licence |
|-----------------|-----|---------|----------------|------|---------|
| `openai/clip-vit-base-patch32` | 512 | Mean | none | Encodeur texte CLIP | MIT (dépôt OpenAI CLIP ; la carte HF n'a pas d'étiquette) |

### Jina

| Nom HuggingFace | Dim | Pooling | Quantification | Context | Licence |
|-----------------|-----|---------|----------------|---------|---------|
| `jinaai/jina-embeddings-v2-base-en` | 768 | Mean | none | 8192 | Apache-2.0 |
| `jinaai/jina-embeddings-v2-base-code` | 768 | Mean | none | 8192 | Apache-2.0 |

### EmbeddingGemma (Google)

| Nom HuggingFace | Dim | Pooling | Quantification | Context | Licence |
|-----------------|-----|---------|----------------|---------|---------|
| `google/embeddinggemma-300m` | 768 | Mean | none | 2048 | ⚠ Gemma Terms of Use (non OSI) |

> **Attention :** le projet n'accepte que des licences compatibles MIT. La licence Gemma
> n'est pas une licence OSI (le dépôt `google/embeddinggemma-300m` est à accès restreint ;
> le modèle est téléchargé via `onnx-community/embeddinggemma-300m-ONNX`). Il est donc
> refusé par défaut (`LEMBED_ERROR_UNSUPPORTED`, avant tout téléchargement) ;
> `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1` le charge quand même, une fois sa licence lue.
> Il quittera le registre en 2.0.

### Snowflake Arctic

| Nom HuggingFace | Dim | Pooling | Quantification | Licence |
|-----------------|-----|---------|----------------|---------|
| `snowflake/snowflake-arctic-embed-xs` | 384 | CLS | none | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-xs` | 384 | CLS | dynamic | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-s` | 384 | CLS | none | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-s` | 384 | CLS | dynamic | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-m` | 768 | CLS | none | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-m` | 768 | CLS | dynamic | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-m-long` | 768 | CLS | none | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-m-long` | 768 | CLS | dynamic | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-l` | 1024 | CLS | none | Apache-2.0 |
| `snowflake/snowflake-arctic-embed-l` | 1024 | CLS | dynamic | Apache-2.0 |

> **Note :** Les variantes suffixées par `_Q` utilisent la quantification (static ou dynamic) pour réduire la taille du modèle — de 2 à 4x sur les embeddings texte denses, d'après le [benchmark de quantification](performance_tuning.html#quantification--fp32-vs-int8) — au détriment d'une légère perte de précision. L'effet sur le débit dépend du modèle et du format réel des poids, et doit être mesuré, pas supposé. Deux pièges : le suffixe `_Q` et l'étiquette « quantized » désignent souvent du **FP16**, pas de l'INT8 — les poids peuvent être en `FLOAT16` (c'est le cas des quatre entrées `Qdrant/*-onnx-Q`). Sur le CPU de référence (Intel Core i7-1065G7, ONNX Runtime 1.29.0), l'INT8 dynamique s'est révélé **1,7x plus rapide** que le FP32 sur MiniLM et **2,0x** sur BGE, avec 3,9x de disque en moins et 1,8 à 2,1x de RAM en moins ; l'entrée BGE `_Q`, en FP16, s'est révélée **9,5x plus lente**. Benchmarkez votre propre modèle avant de choisir.

---

## Modèles d'images (6 modèles)

| Nom HuggingFace | Dim | Description | Licence |
|-----------------|-----|-------------|---------|
| `openai/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 (défaut) | MIT (dépôt OpenAI CLIP) |
| `microsoft/resnet-50` | 2048 | ResNet-50 | Apache-2.0 |
| `open-metric-learning/unicom-vit-b-16` | 768 | Unicom ViT-B/16 | MIT |
| `open-metric-learning/unicom-vit-b-32` | 512 | Unicom ViT-B/32 | MIT |
| `nomic-ai/nomic-embed-vision-v1.5` | 768 | Nomic embed vision v1.5 | Apache-2.0 |
| `Xenova/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 INT8 quantifié | MIT |

---

## Modèles GGUF (backend llama.cpp)

Les modèles GGUF sont chargés par chemin de fichier (pas par enum). Ils utilisent le backend llama.cpp.

| Nom du modèle | Dim | Quantification | Description | Licence |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Q4_K_M | MiniLM L6, ~4-bit | Apache-2.0 |
| `all-MiniLM-L12-v2` | 384 | Q4_K_M | MiniLM L12, ~4-bit | non vérifié (dépôt GGUF non identifié) |
| `bge-small-en-v1.5` | 384 | Q4_K_M | BGE small, ~4-bit | non vérifié (dépôt GGUF non identifié) |
| `bge-base-en-v1.5` | 768 | Q4_K_M | BGE base, ~4-bit | non vérifié (dépôt GGUF non identifié) |
| `bge-large-en-v1.5` | 1024 | Q4_K_M | BGE large, ~4-bit | non vérifié (dépôt GGUF non identifié) |
| `snowflake-arctic-embed-xs` | 384 | Q4_K_M | Snowflake XS, ~4-bit | Apache-2.0 |

---

## Modèles sparse (2 modèles)

| Nom HuggingFace | Dim | Tokens max | Description | Licence |
|-----------------|-----|-----------|-------------|---------|
| `prithivida/Splade_PP_en_v1` | variable | 512 | SPLADE++ v1 (défaut) | Apache-2.0 |
| `BAAI/bge-m3` | variable | 8192 | BGE-M3, 100+ langues | MIT |

> **Note :** Les embeddings sparse n'ont pas de dimension fixe. La dimension retournée correspond au nombre de tokens actifs dans le vocabulaire.

---

## Modèles de reranking (5 modèles)

| HuggingFace name | Max tokens | Description | Licence |
|------------------|-----------|-------------|---------|
| `BAAI/bge-reranker-base` | 512 | BGE Reranker base (défaut) | MIT |
| `BAAI/bge-reranker-v2-m3` | 512 | BGE Reranker v2 multilingue | Apache-2.0 (BAAI/bge-reranker-v2-m3) |
| `jinaai/jina-reranker-v1-turbo-en` | 8192 | Jina Reranker v1 turbo anglais | Apache-2.0 |
| `jinaai/jina-reranker-v2-base-multilingual` | 8192 | Jina Reranker v2 multilingue | ⚠ CC-BY-NC-4.0, usage non commercial |
| `jinaai/jina-reranker-v1-turbo-en` (quantized) | 8192 | Jina Reranker v1 turbo (INT8) | Apache-2.0 |

> **Attention :** le projet n'accepte que des licences compatibles MIT. La licence
> CC-BY-NC-4.0 de `jinaai/jina-reranker-v2-base-multilingual` ne l'est pas (la carte du
> modèle le réserve à la recherche et à l'évaluation). Il est donc refusé par défaut
> (`LEMBED_ERROR_UNSUPPORTED`, avant tout téléchargement) ;
> `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1` le charge quand même, une fois sa licence lue.
> Il quittera le registre en 2.0.

---

## Obtention des informations de modèle

### Depuis Python

```python
import libembedding

# Lister tous les modèles d'une catégorie
text_models = libembedding.list_text_models()
sparse_models = libembedding.list_sparse_models()
image_models = libembedding.list_image_models()
reranker_models = libembedding.list_reranker_models()

# Inspecter un modèle
for m in text_models:
    if m.model_name == "BAAI/bge-small-en-v1.5":
        print(f"dim={m.dim}, max_tokens={m.max_tokens}, pooling={m.pooling}")
        break
```

### Depuis le C

```c
#include <libembedding/model_registry.h>

const lembed_model_info_t* models;
int count;
lembed_list_text_models(&models, &count);

for (int i = 0; i < count; i++) {
    printf("%-40s dim=%-4d %s pooling  %s\n",
           models[i].model_name,
           models[i].dim,
           models[i].pooling == LEMBED_POOLING_CLS ? "CLS" : "Mean",
           models[i].description);
}
```
