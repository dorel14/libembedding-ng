---
title: Modèles
nav_order: 4
---

# Modèles disponibles

libembedding intègre un registre de modèles pré-configurés. Vous pouvez les lister dynamiquement depuis Python :

```python
import libembedding

for m in libembedding.list_text_models():
    print(f"{m.model_name:45} dim={m.dim:<5} {m.pooling} quant={m.quantization}")
```

## Modèles de texte (44 modèles)

### Anglais — BGE

| Nom HuggingFace | Dim | Pooling | Quantification |
|-----------------|-----|---------|----------------|
| `BAAI/bge-small-en-v1.5` | 384 | CLS | none |
| `BAAI/bge-small-en-v1.5` | 384 | CLS | static |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | none |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | static |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | none |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | static |

### Anglais — MiniLM / MPNet

| Nom HuggingFace | Dim | Pooling | Quantification |
|-----------------|-----|---------|----------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | none |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | dynamic |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | none |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | dynamic |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Mean | none |

### Multilingue — E5 / MiniLM / MPNet

| Nom HuggingFace | Dim | Pooling | Quantification |
|-----------------|-----|---------|----------------|
| `intfloat/multilingual-e5-small` | 384 | Mean | none |
| `intfloat/multilingual-e5-base` | 768 | Mean | none |
| `intfloat/multilingual-e5-large` | 1024 | Mean | none |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | none |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | static |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Mean | none |

### Chinois — BGE

| Nom HuggingFace | Dim | Pooling | Quantification |
|-----------------|-----|---------|----------------|
| `BAAI/bge-small-zh-v1.5` | 512 | CLS | none |
| `BAAI/bge-large-zh-v1.5` | 1024 | CLS | none |

### Multilingue — BGE-M3 / ModernBERT

| Nom HuggingFace | Dim | Pooling | Quantification |
|-----------------|-----|---------|----------------|
| `BAAI/bge-m3` | 1024 | CLS | none |
| `lightonai/modernbert-embed-large` | 1024 | Mean | none |

### Nomic

| Nom HuggingFace | Dim | Pooling | Quantification | Context |
|-----------------|-----|---------|----------------|---------|
| `nomic-ai/nomic-embed-text-v1` | 768 | Mean | none | 8192 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | none | 8192 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | dynamic | 8192 |

### GTE (Alibaba)

| Nom HuggingFace | Dim | Pooling | Quantification | Context |
|-----------------|-----|---------|----------------|---------|
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | none | 8192 |
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | dynamic | 8192 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | none | 8192 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | dynamic | 8192 |

### MixedBread

| Nom HuggingFace | Dim | Pooling | Quantification |
|-----------------|-----|---------|----------------|
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | none |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | dynamic |

### CLIP

| Nom HuggingFace | Dim | Pooling | Quantification | Note |
|-----------------|-----|---------|----------------|------|
| `openai/clip-vit-base-patch32` | 512 | Mean | none | Encodeur texte CLIP |

### Jina

| Nom HuggingFace | Dim | Pooling | Quantification | Context |
|-----------------|-----|---------|----------------|---------|
| `jinaai/jina-embeddings-v2-base-en` | 768 | Mean | none | 8192 |
| `jinaai/jina-embeddings-v2-base-code` | 768 | Mean | none | 8192 |

### EmbeddingGemma (Google)

| Nom HuggingFace | Dim | Pooling | Quantification | Context |
|-----------------|-----|---------|----------------|---------|
| `google/embeddinggemma-300m` | 768 | Mean | none | 2048 |

### Snowflake Arctic

| Nom HuggingFace | Dim | Pooling | Quantification |
|-----------------|-----|---------|----------------|
| `snowflake/snowflake-arctic-embed-xs` | 384 | CLS | none |
| `snowflake/snowflake-arctic-embed-xs` | 384 | CLS | dynamic |
| `snowflake/snowflake-arctic-embed-s` | 384 | CLS | none |
| `snowflake/snowflake-arctic-embed-s` | 384 | CLS | dynamic |
| `snowflake/snowflake-arctic-embed-m` | 768 | CLS | none |
| `snowflake/snowflake-arctic-embed-m` | 768 | CLS | dynamic |
| `snowflake/snowflake-arctic-embed-m-long` | 768 | CLS | none |
| `snowflake/snowflake-arctic-embed-m-long` | 768 | CLS | dynamic |
| `snowflake/snowflake-arctic-embed-l` | 1024 | CLS | none |
| `snowflake/snowflake-arctic-embed-l` | 1024 | CLS | dynamic |

> **Note :** Les variantes suffixées par `_Q` utilisent la quantification (static ou dynamic) pour réduire la taille du modèle et accélérer l'inférence au détriment d'une légère perte de précision.

---

## Modèles d'images (6 modèles)

| Nom HuggingFace | Dim | Description |
|-----------------|-----|-------------|
| `openai/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 (défaut) |
| `microsoft/resnet-50` | 2048 | ResNet-50 |
| `open-metric-learning/unicom-vit-b-16` | 768 | Unicom ViT-B/16 |
| `open-metric-learning/unicom-vit-b-32` | 512 | Unicom ViT-B/32 |
| `nomic-ai/nomic-embed-vision-v1.5` | 768 | Nomic embed vision v1.5 |
| `Xenova/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 INT8 quantifié |

---

## Modèles GGUF (backend llama.cpp)

Les modèles GGUF sont chargés par chemin de fichier (pas par enum). Ils utilisent le backend llama.cpp.

| Nom du modèle | Dim | Quantification | Description |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Q4_K_M | MiniLM L6, ~4-bit |
| `all-MiniLM-L12-v2` | 384 | Q4_K_M | MiniLM L12, ~4-bit |
| `bge-small-en-v1.5` | 384 | Q4_K_M | BGE small, ~4-bit |
| `bge-base-en-v1.5` | 768 | Q4_K_M | BGE base, ~4-bit |
| `bge-large-en-v1.5` | 1024 | Q4_K_M | BGE large, ~4-bit |
| `snowflake-arctic-embed-xs` | 384 | Q4_K_M | Snowflake XS, ~4-bit |

---

## Modèles sparse (2 modèles)

| Nom HuggingFace | Dim | Tokens max | Description |
|-----------------|-----|-----------|-------------|
| `prithvida/SPLADE_PP_en_v1` | variable | 512 | SPLADE++ v1 (défaut) |
| `BAAI/bge-m3` | variable | 8192 | BGE-M3, 100+ langues |

> **Note :** Les embeddings sparse n'ont pas de dimension fixe. La dimension retournée correspond au nombre de tokens actifs dans le vocabulaire.

---

## Modèles de reranking (5 modèles)

| HuggingFace name | Max tokens | Description |
|------------------|-----------|-------------|
| `BAAI/bge-reranker-base` | 512 | BGE Reranker base (défaut) |
| `BAAI/bge-reranker-v2-m3` | 512 | BGE Reranker v2 multilingue |
| `jinaai/jina-reranker-v1-turbo-en` | 8192 | Jina Reranker v1 turbo anglais |
| `jinaai/jina-reranker-v2-base-multilingual` | 8192 | Jina Reranker v2 multilingue |
| `jinaai/jina-reranker-v1-turbo-en` (quantized) | 8192 | Jina Reranker v1 turbo (INT8) |

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
