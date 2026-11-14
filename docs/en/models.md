---
nav_exclude: true
---

# Available models

libembedding ships with a registry of pre-configured models. You can list them dynamically from Python:

> **Android and iOS:** only light models (≤ 150 MB) load by name, and
> `list_*_models()` returns only those. See
> [Platforms — mobile model limit](platforms.html).

> **Licence column:** the licence was read on each model's HuggingFace card
> (survey of 2026-11-14). A downloaded model keeps its own licence, independent of
> libembedding's MIT licence. When the repository actually downloaded is a conversion
> with no licence tag (Xenova, Qdrant, onnx-community…), it inherits the licence of the
> original model, shown in parentheses.

```python
import libembedding

for m in libembedding.list_text_models():
    print(f"{m.model_name:45} dim={m.dim:<5} {m.pooling} quant={m.quantization}")
```

## Text models (44 models)

### English — BGE

| HuggingFace name | Dim | Pooling | Quantization | Licence |
|------------------|-----|---------|--------------|---------|
| `BAAI/bge-small-en-v1.5` | 384 | CLS | none | MIT (BAAI/bge-small-en-v1.5) |
| `BAAI/bge-small-en-v1.5` | 384 | CLS | static | MIT |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | none | MIT |
| `BAAI/bge-base-en-v1.5` | 768 | CLS | static | MIT |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | none | MIT (BAAI/bge-large-en-v1.5) |
| `BAAI/bge-large-en-v1.5` | 1024 | CLS | static | MIT (BAAI/bge-large-en-v1.5) |

### English — MiniLM / MPNet

| HuggingFace name | Dim | Pooling | Quantization | Licence |
|------------------|-----|---------|--------------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | none | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Mean | dynamic | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | none | Apache-2.0 (sentence-transformers/all-MiniLM-L12-v2) |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Mean | dynamic | Apache-2.0 (sentence-transformers/all-MiniLM-L12-v2) |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Mean | none | Apache-2.0 (sentence-transformers/all-mpnet-base-v2) |

### Multilingual — E5 / MiniLM / MPNet

| HuggingFace name | Dim | Pooling | Quantization | Licence |
|------------------|-----|---------|--------------|---------|
| `intfloat/multilingual-e5-small` | 384 | Mean | none | MIT |
| `intfloat/multilingual-e5-base` | 768 | Mean | none | MIT |
| `intfloat/multilingual-e5-large` | 1024 | Mean | none | MIT |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | none | Apache-2.0 (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Mean | static | Apache-2.0 |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Mean | none | Apache-2.0 (sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |

### Chinese — BGE

| HuggingFace name | Dim | Pooling | Quantization | Licence |
|------------------|-----|---------|--------------|---------|
| `BAAI/bge-small-zh-v1.5` | 512 | CLS | none | MIT (BAAI/bge-small-zh-v1.5) |
| `BAAI/bge-large-zh-v1.5` | 1024 | CLS | none | MIT (BAAI/bge-large-zh-v1.5) |

### Multilingual — BGE-M3 / ModernBERT

| HuggingFace name | Dim | Pooling | Quantization | Licence |
|------------------|-----|---------|--------------|---------|
| `BAAI/bge-m3` | 1024 | CLS | none | MIT |
| `lightonai/modernbert-embed-large` | 1024 | Mean | none | Apache-2.0 |

### Nomic

| HuggingFace name | Dim | Pooling | Quantization | Context | Licence |
|------------------|-----|---------|--------------|---------|---------|
| `nomic-ai/nomic-embed-text-v1` | 768 | Mean | none | 8192 | Apache-2.0 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | none | 8192 | Apache-2.0 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Mean | dynamic | 8192 | Apache-2.0 |

### GTE (Alibaba)

| HuggingFace name | Dim | Pooling | Quantization | Context | Licence |
|------------------|-----|---------|--------------|---------|---------|
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | none | 8192 | Apache-2.0 |
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | CLS | dynamic | 8192 | Apache-2.0 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | none | 8192 | Apache-2.0 |
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 | CLS | dynamic | 8192 | Apache-2.0 |

### MixedBread

| HuggingFace name | Dim | Pooling | Quantization | Licence |
|------------------|-----|---------|--------------|---------|
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | none | Apache-2.0 |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | CLS | dynamic | Apache-2.0 |

### CLIP

| HuggingFace name | Dim | Pooling | Quantization | Note | Licence |
|------------------|-----|---------|--------------|------|---------|
| `openai/clip-vit-base-patch32` | 512 | Mean | none | CLIP text encoder | MIT (OpenAI CLIP repository; the HF card has no tag) |

### Jina

| HuggingFace name | Dim | Pooling | Quantization | Context | Licence |
|------------------|-----|---------|--------------|---------|---------|
| `jinaai/jina-embeddings-v2-base-en` | 768 | Mean | none | 8192 | Apache-2.0 |
| `jinaai/jina-embeddings-v2-base-code` | 768 | Mean | none | 8192 | Apache-2.0 |

### EmbeddingGemma (Google)

| HuggingFace name | Dim | Pooling | Quantization | Context | Licence |
|------------------|-----|---------|--------------|---------|---------|
| `google/embeddinggemma-300m` | 768 | Mean | none | 2048 | ⚠ Gemma Terms of Use (not OSI) |

> **Warning:** the project accepts only MIT-compatible licences. The Gemma licence is
> not an OSI licence (the `google/embeddinggemma-300m` repository is gated; the model is
> downloaded through `onnx-community/embeddinggemma-300m-ONNX`). It is therefore refused
> by default (`LEMBED_ERROR_UNSUPPORTED`, before any download);
> `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1` loads it anyway, once you have read its
> licence. It leaves the registry in 2.0.

### Snowflake Arctic

| HuggingFace name | Dim | Pooling | Quantization | Licence |
|------------------|-----|---------|--------------|---------|
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

> **Note:** Variants suffixed with `_Q` use quantization (static or dynamic) to reduce model size — 2 to 4x on dense text embeddings, per the [quantization benchmark](performance_tuning.html#quantization-fp32-vs-int8) — at the cost of slight accuracy loss. The throughput effect depends on the model and on the weights' actual format, and must be measured, not assumed. Two traps: the `_Q` suffix and the "quantized" label often mean **FP16**, not INT8 — initializers can be `FLOAT16`, which is the case for all four `Qdrant/*-onnx-Q` entries. On the reference CPU (Intel Core i7-1065G7, ONNX Runtime 1.29.0), dynamic INT8 came out **1.7x faster** than FP32 on MiniLM and **2.0x** on BGE, with 3.9x less disk and 1.8-2.1x less peak RAM; the BGE `_Q` entry, being FP16, came out **9.5x slower**. Benchmark your own model before choosing.

---

## Image models (6 models)

| HuggingFace name | Dim | Description | Licence |
|------------------|-----|-------------|---------|
| `openai/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 (default) | MIT (OpenAI CLIP repository) |
| `microsoft/resnet-50` | 2048 | ResNet-50 | Apache-2.0 |
| `open-metric-learning/unicom-vit-b-16` | 768 | Unicom ViT-B/16 | MIT |
| `open-metric-learning/unicom-vit-b-32` | 512 | Unicom ViT-B/32 | MIT |
| `nomic-ai/nomic-embed-vision-v1.5` | 768 | Nomic embed vision v1.5 | Apache-2.0 |
| `Xenova/clip-vit-base-patch32` | 512 | CLIP ViT-B/32 INT8 quantized | MIT |

---

## GGUF models (llama.cpp backend)

GGUF models are loaded by file path (not by enum). They use the llama.cpp backend.

| Model name | Dim | Quantization | Description | Licence |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Q4_K_M | MiniLM L6, ~4-bit | Apache-2.0 |
| `all-MiniLM-L12-v2` | 384 | Q4_K_M | MiniLM L12, ~4-bit | not verified (GGUF repository not identified) |
| `bge-small-en-v1.5` | 384 | Q4_K_M | BGE small, ~4-bit | not verified (GGUF repository not identified) |
| `bge-base-en-v1.5` | 768 | Q4_K_M | BGE base, ~4-bit | not verified (GGUF repository not identified) |
| `bge-large-en-v1.5` | 1024 | Q4_K_M | BGE large, ~4-bit | not verified (GGUF repository not identified) |
| `snowflake-arctic-embed-xs` | 384 | Q4_K_M | Snowflake XS, ~4-bit | Apache-2.0 |

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

| HuggingFace name | Dim | Max tokens | Description | Licence |
|------------------|-----|-----------|-------------|---------|
| `prithivida/Splade_PP_en_v1` | variable | 512 | SPLADE++ v1 (default) | Apache-2.0 |
| `BAAI/bge-m3` | variable | 8192 | BGE-M3, 100+ languages | MIT |

> **Note:** Sparse embeddings have no fixed dimension. The returned dimension equals the number of active tokens in the vocabulary.

---

## Reranker models (5 models)

| HuggingFace name | Max tokens | Description | Licence |
|------------------|-----------|-------------|---------|
| `BAAI/bge-reranker-base` | 512 | BGE Reranker base (default) | MIT |
| `BAAI/bge-reranker-v2-m3` | 512 | Multilingual BGE Reranker v2 | Apache-2.0 (BAAI/bge-reranker-v2-m3) |
| `jinaai/jina-reranker-v1-turbo-en` | 8192 | Jina Reranker v1 turbo English | Apache-2.0 |
| `jinaai/jina-reranker-v2-base-multilingual` | 8192 | Jina Reranker v2 multilingual | ⚠ CC-BY-NC-4.0, non-commercial |
| `jinaai/jina-reranker-v1-turbo-en` (quantized) | 8192 | Jina Reranker v1 turbo (INT8) | Apache-2.0 |

> **Warning:** the project accepts only MIT-compatible licences. The CC-BY-NC-4.0 licence
> of `jinaai/jina-reranker-v2-base-multilingual` is not (the model card restricts it to
> research and evaluation). It is therefore refused by default
> (`LEMBED_ERROR_UNSUPPORTED`, before any download);
> `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES=1` loads it anyway, once you have read its
> licence. It leaves the registry in 2.0.

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
