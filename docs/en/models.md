---
nav_exclude: true
---

# Available models

libembedding ships with a registry of pre-configured models. You can list them dynamically from Python:

```python
import libembedding

for m in libembedding.list_text_models():
    print(f"{m.model_name:45} dim={m.dim:<5} {m.pooling} quant={m.quantization}")
```

## Text models (44 models)

### English — BGE

| HuggingFace name | Dim | Pooling | Quantization |
|------------------|-----|---------|--------------|
| `BAAI/bge-small-en-v1.5` | 384 | CLS | none |
| `BAAI/bge-small-en-v1.5` | 384 | CLS | static |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | none |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | static |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | none |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | static |

### English — MiniLM / MPNet

| HuggingFace name | Dim | Pooling | Quantization |
|------------------|-----|---------|--------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | none |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | dynamic |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | none |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | dynamic |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Mean | none |

### Multilingual — E5 / MiniLM / MPNet

| HuggingFace name | Dim | Pooling | Quantization |
|------------------|-----|---------|--------------|
| `intfloat/multilingual-e5-small` | 384 | Mean | none |
| `intfloat/multilingual-e5-base` | 768 | Mean | none |
| `intfloat/multilingual-e5-large` | 1024 | Mean | none |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | none |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | static |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Mean | none |

### Chinese — BGE

| HuggingFace name | Dim | Pooling | Quantization |
|------------------|-----|---------|--------------|
| `BAAI/bge-small-zh-v1.5` | 512 | CLS | none |
| `BAAI/bge-large-zh-v1.5` | 1024 | CLS | none |

### Multilingual — BGE-M3 / ModernBERT

| HuggingFace name | Dim | Pooling | Quantization |
|------------------|-----|---------|--------------|
| `BAAI/bge-m3` | 1024 | CLS | none |
| `lightonai/modernbert-embed-large` | 1024 | Mean | none |

### Nomic

| HuggingFace name | Dim | Pooling | Quantization | Context |
|------------------|-----|---------|--------------|---------|
| `nomic-ai/nomic-embed-text-v1` | 768 | Mean | none | 8192 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | none | 8192 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | dynamic | 8192 |

### GTE (Alibaba)

| HuggingFace name | Dim | Pooling | Quantization | Context |
|------------------|-----|---------|--------------|---------|
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | none | 8192 |
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | dynamic | 8192 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | none | 8192 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | dynamic | 8192 |

### MixedBread

| HuggingFace name | Dim | Pooling | Quantization |
|------------------|-----|---------|--------------|
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | none |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | dynamic |

### CLIP

| HuggingFace name | Dim | Pooling | Quantization | Note |
|------------------|-----|---------|--------------|------|
| `openai/clip-vit-base-patch32` | 512 | Mean | none | CLIP text encoder |

### Jina

| HuggingFace name | Dim | Pooling | Quantization | Context |
|------------------|-----|---------|--------------|---------|
| `jinaai/jina-embeddings-v2-base-en` | 768 | Mean | none | 8192 |
| `jinaai/jina-embeddings-v2-base-code` | 768 | Mean | none | 8192 |

### EmbeddingGemma (Google)

| HuggingFace name | Dim | Pooling | Quantization | Context |
|------------------|-----|---------|--------------|---------|
| `google/embeddinggemma-300m` | 768 | Mean | none | 2048 |

### Snowflake Arctic

| HuggingFace name | Dim | Pooling | Quantization |
|------------------|-----|---------|--------------|
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

> **Note:** Variants suffixed with `_Q` use quantization (static or dynamic) to reduce model size and speed up inference at the cost of slight accuracy loss.

---

## Image models (6 models)

| HuggingFace name | Dim | Description |
|------------------|-----|-------------|
| `openai/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 (default) |
| `microsoft/resnet-50` | 2048 | ResNet-50 |
| `open-metric-learning/unicom-vit-b-16` | 768 | Unicom ViT-B/16 |
| `open-metric-learning/unicom-vit-b-32` | 512 | Unicom ViT-B/32 |
| `nomic-ai/nomic-embed-vision-v1.5` | 768 | Nomic embed vision v1.5 |
| `Xenova/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 INT8 quantized |

---

## GGUF models (llama.cpp backend)

GGUF models are loaded by file path (not by enum). They use the llama.cpp backend.

| Model name | Dim | Quantization | Description |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Q4_K_M | MiniLM L6, ~4-bit |
| `all-MiniLM-L12-v2` | 384 | Q4_K_M | MiniLM L12, ~4-bit |
| `bge-small-en-v1.5` | 384 | Q4_K_M | BGE small, ~4-bit |
| `bge-base-en-v1.5` | 768 | Q4_K_M | BGE base, ~4-bit |
| `bge-large-en-v1.5` | 1024 | Q4_K_M | BGE large, ~4-bit |
| `snowflake-arctic-embed-xs` | 384 | Q4_K_M | Snowflake XS, ~4-bit |

Browse the full registry at runtime:

```python
import libembedding
# Note: GGUF models are listed via lembed_list_gguf_models() in C
```

```c
#include <libembedding/gguf_registry.h>

const lembed_gguf_model_info_t* models;
int count;
lembed_list_gguf_models(&models, &count);
for (int i = 0; i < count; i++) {
    printf("%-24s dim=%-4d quality=%.3f\n",
           models[i]->name, models[i]->dim, models[i]->quality_mteb);
}
```

---

## Sparse models (2 models)

| HuggingFace name | Dim | Max tokens | Description |
|------------------|-----|-----------|-------------|
| `prithvida/SPLADE_PP_en_v1` | variable | 512 | SPLADE++ v1 (default) |
| `BAAI/bge-m3` | variable | 8192 | BGE-M3, 100+ languages |

> **Note:** Sparse embeddings have no fixed dimension. The returned dimension equals the number of active tokens in the vocabulary.

---

## Reranker models (5 models)

| HuggingFace name | Max tokens | Description |
|------------------|-----------|-------------|
| `BAAI/bge-reranker-base` | 512 | BGE Reranker base (default) |
| `BAAI/bge-reranker-v2-m3` | 512 | Multilingual BGE Reranker v2 |
| `jinaai/jina-reranker-v1-turbo-en` | 8192 | Jina Reranker v1 turbo English |
| `jinaai/jina-reranker-v2-base-multilingual` | 8192 | Jina Reranker v2 multilingual |
| `jinaai/jina-reranker-v1-turbo-en` (quantized) | 8192 | Jina Reranker v1 turbo (INT8) |

---

## Querying model information

### From Python

```python
import libembedding

# List all models in a category
text_models = libembedding.list_text_models()
sparse_models = libembedding.list_sparse_models()
image_models = libembedding.list_image_models()
reranker_models = libembedding.list_reranker_models()

# Inspect a specific model
for m in text_models:
    if m.model_name == "BAAI/bge-small-en-v1.5":
        print(f"dim={m.dim}, max_tokens={m.max_tokens}, pooling={m.pooling}")
        break
```

### From C

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
