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

Les modèles GGUF sont chargés par chemin de fichier ou par dépôt HuggingFace + fichier
(pas par enum). Ils utilisent le backend llama.cpp. Le registre ci-dessous
(`lembed_list_gguf_models()`, source : `include/libembedding/detail/gguf_registry_impl.hpp`)
liste les fichiers vérifiés.

| Nom du registre | Modèle d'origine | Dépôt GGUF | Quant. | Dim | Taille | Licence |
|---|---|---|---|---|---|---|
| `Snowflake-XS-Q4` (défaut) | `snowflake/snowflake-arctic-embed-xs` | `ChristianAzinn/snowflake-arctic-embed-xs-gguf` | Q4_K_M | 384 | ~20 Mo | Apache-2.0 |
| `MiniLM-L6-Q4` | `sentence-transformers/all-MiniLM-L6-v2` | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` | Q4_K_M | 384 | ~20 Mo | Apache-2.0 |
| `MiniLM-L6-Q8` | `sentence-transformers/all-MiniLM-L6-v2` | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` | Q8_0 | 384 | ~24 Mo | Apache-2.0 |
| `Snowflake-S-Q4` | `snowflake/snowflake-arctic-embed-s` | `ChristianAzinn/snowflake-arctic-embed-s-gguf` | Q4_K_M | 384 | ~28 Mo | Apache-2.0 |
| `E5-small-Q4` | `intfloat/e5-small-v2` | `ChristianAzinn/e5-small-v2-gguf` | Q4_K_M | 384 | ~28 Mo | MIT |
| `GIST-small-Q4` | `avsolatorio/GIST-small-Embedding-v0` | `ChristianAzinn/gist-small-embedding-v0-gguf` | Q4_K_M | 384 | ~28 Mo | MIT |
| `BGE-Reranker-v2-M3` (reranker, défaut) | `BAAI/bge-reranker-v2-m3` | `smarttasks/bge-reranker-v2-m3-GGUF` | Q4_K_M | — | ~438 Mo | Apache-2.0 |
| `BGE-Reranker-Base` (reranker) | `BAAI/bge-reranker-base` | `sinjab/bge-reranker-base-Q4_K_M-GGUF` | Q4_K_M | — | ~219 Mo | MIT (conversion étiquetée Apache-2.0) |
| `BGE-Reranker-Large` (reranker) | `BAAI/bge-reranker-large` | `sinjab/bge-reranker-large-Q4_K_M-GGUF` | Q4_K_M | — | ~407 Mo | MIT (conversion étiquetée Apache-2.0) |
| `Qwen3-Reranker-0.6B` (reranker) | `Qwen/Qwen3-Reranker-0.6B` | `sinjab/Qwen3-Reranker-0.6B-Q4_K_M-GGUF` | Q4_K_M | — | ~396 Mo | Apache-2.0 |
| `Jina-Reranker-v1-Turbo-En` (reranker) | `jinaai/jina-reranker-v1-turbo-en` | `sinjab/jina-reranker-v1-turbo-en-Q8_0-GGUF` | Q8_0 | — | ~36 Mo | Apache-2.0 |

Tout autre fichier GGUF reste chargeable par chemin ; le registre ne recense que des fichiers vérifiés.

---

## Modèles sparse (2 modèles)

| Nom HuggingFace | Dim | Tokens max | Description | Licence |
|-----------------|-----|-----------|-------------|---------|
| `prithivida/Splade_PP_en_v1` | variable | 512 | SPLADE++ v1 (défaut) | Apache-2.0 |
| `BAAI/bge-m3` | variable | 8192 | BGE-M3, 100+ langues | MIT |

> **Note :** Les embeddings sparse n'ont pas de dimension fixe. La dimension retournée correspond au nombre de tokens actifs dans le vocabulaire.

### Modèles sparse GGUF (1 modèle)

Exécutés par le runtime ggml de libembedding, pas par llama.cpp. Listés par
`lembed_list_sparse_gguf_models()` ; en Python, passer le nom du registre.

| Nom du registre | Fichier | Taille | Licence | Validation |
|-----------------|---------|--------|---------|------------|
| `SPLADE-PP-En-v1-GGUF` | `cstr/splade-pp-en-v1-GGUF`, `splade-pp-en-v1-q8_0.gguf` | ~112 Mo | Apache-2.0 | cosinus ≥ 0.995 contre l'ONNX `prithivida/Splade_PP_en_v1` (minimum mesuré 0.9998) |

```python
from libembedding import SparseTextEmbedding
model = SparseTextEmbedding("SPLADE-PP-En-v1-GGUF")   # télécharge le fichier Q8_0
```

Le nom est comparé exactement (sans tenir compte de la casse) : `"Splade_PP_en_v1"`
et les autres noms du modèle ONNX chargent toujours le modèle ONNX.

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

## Matrice de compatibilité (validation par référence)

Un modèle n'est *validé* que lorsque libembedding reproduit une référence. La référence
est le **même fichier ONNX** exécuté avec `onnxruntime` Python, alimenté par les ids, le
masque d'attention et les types de jetons de `tokenizers` (HuggingFace), puis poolé et
normalisé comme l'indique l'entrée du registry. Poids et runtime sont communs : l'écart
mesuré vient uniquement du tokenizer, de la mise en forme des entrées et du pooling de
libembedding. Le test est `python/tests/test_model_reference.py` (marqueur `network`,
ignoré si le modèle n'est pas en cache ou si `onnxruntime`/`tokenizers` manquent) ; le
rapport se régénère avec `python python/tests/model_reference.py --json report.json`.

Corpus : les 54 textes de `tests/data/tokenizer/corpus.json` (anglais, latin accentué,
vietnamien, grec, cyrillique, arabe, hébreu, devanagari, CJK, coréen, thaï, emoji,
ponctuation, accents combinants, pleine chasse) plus un texte de plus de 512 jetons.
La troncature est fixée à 512 des deux côtés. La colonne *Anglais* exclut un cas limite
présent sur tous les modèles WordPiece : un mot de plus de 100 caractères
(`en-long-word`), que HF remplace par `[UNK]` alors que libembedding le découpe (cosinus
0,36 à 0,88).

Statut selon le cosinus minimal : **validé** >= 0,999, **dégradé** 0,99 à 0,999,
**cassé** < 0,99. Mesuré le 2026-11-17 ; les variantes quantifiées sont dans la même
ligne (plages).

| Modèle | Famille de tokenizer | Cosinus min / moyen (tous) | Anglais | Statut (tous textes) | Statut (anglais) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | WordPiece | 0,265-0,282 / 0,743-0,747 | 1,000 | cassé | validé |
| all-MiniLM-L12-v2 | WordPiece | 0,265-0,275 / 0,733-0,735 | 1,000 | cassé | validé |
| bge-small-en-v1.5 | WordPiece | 0,623-0,629 / 0,862-0,864 | 1,000 | cassé | validé |
| bge-base-en-v1.5 | WordPiece | 0,666-0,669 / 0,863-0,864 | 1,000 | cassé | validé |
| nomic-embed-text-v1.5 | WordPiece | 0,652-0,656 / 0,846-0,850 | 1,000 | cassé | validé |
| mxbai-embed-large-v1 (quantifié) | WordPiece | 0,527 / 0,835 | 1,000 | cassé | validé |
| gte-base-en-v1.5 | WordPiece | 0,494-0,519 / 0,798-0,799 | 1,000 | cassé | validé |
| gte-large-en-v1.5 (quantifié) | WordPiece | 0,628 / 0,858 | 1,000 | cassé | validé |
| snowflake-arctic-embed-xs / s | WordPiece | 0,815-0,860 / 0,955-0,958 | 1,000 | cassé | validé |
| snowflake-arctic-embed-m / m-long / l | WordPiece | 0,776-0,853 / 0,929-0,957 | 1,000 | cassé | validé |
| paraphrase-multilingual-MiniLM-L12-v2 | Unigram (SentencePiece) | 0,0015 / 0,343 | 0,034 | cassé | cassé |
| multilingual-e5-small | Unigram (SentencePiece) | 0,768 / 0,824 | 0,768 | cassé | cassé |
| clip-vit-base-patch32 (texte) | BPE byte-level | 0,295 / 0,763 | 0,316 | cassé | cassé |
| Splade_PP_en_v1 (sparse, cosinus des vecteurs) | WordPiece | 0,161 / 0,647 | 1,000 | cassé | validé |
| jina-reranker-v1-turbo-en (reranker, Pearson des scores) | BPE (non byte-level) | 0,42-0,46 | n/d | cassé | cassé |

Lecture : **l'anglais sur un modèle WordPiece est exact** (cosinus 1,000, y compris le
texte de plus de 512 jetons). Tout ce qui n'est pas de l'anglais simple décroche, car le
normaliseur ne replie que Latin-1 et Latin Extended-A et le tokenizer n'implémente que
WordPiece et un BPE approximatif. Unigram (XLM-R) et BPE byte-level (CLIP, ModernBERT)
sont cassés même en anglais. Le reranker construit en outre sa paire comme le texte
`query [SEP] document`, là où HF encode une vraie paire.

Non mesurés (absents du cache local ou de plus de 560 Mo ; la famille vient de
l'architecture, pas du `tokenizer.json`) : all-mpnet-base-v2, bge-large-en-v1.5,
bge-small/large-zh-v1.5, nomic-embed-text-v1 et jina-embeddings-v2 (WordPiece), bge-m3,
multilingual-e5-base/large, paraphrase-multilingual-mpnet-base-v2 et les rerankers XLM-R
(Unigram), modernbert-embed-large (BPE byte-level), embeddinggemma-300m (SentencePiece),
les modèles d'image et les modèles GGUF. Ils sont **non validés**.

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
