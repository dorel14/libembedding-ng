# libembedding-ng

**Local-first embedding and reranking engine for C/C++ and Python.**

libembedding-ng provides a unified runtime for ONNX Runtime and llama.cpp models,
supporting dense embeddings, sparse embeddings, image embeddings and reranking
through a single API.

Designed for search engines, RAG systems, vector databases and AI applications,
it runs entirely locally with no external services required.

---

## What it is

libembedding-ng is a **local inference engine specialized in embeddings and reranking**.

It is not a general-purpose ML runtime. It is a focused, production-grade stack
for turning text, images and documents into vectors — with zero network dependency
once models are cached.

Unlike generic serving frameworks, libembedding-ng gives you:

- **One API for ONNX and GGUF** — switch backends without changing application code
- **Native C/C++ core** with **Python bindings** — same backend, two ergonomics
- **First-class Windows support** — native DLL, CMake build, no WSL required
- **Complete modality coverage** — dense, sparse, vision, and reranking in one library
- **Production helpers built-in** — autotuning, session pooling, streaming, benchmarking

No Rust toolchain. No external inference server. No mandatory cloud dependency.

---

## Quick Start

### Python

```bash
pip install libembedding-ng
```

```python
from libembedding import TextEmbedding, SparseTextEmbedding, Reranker
import numpy as np

# Dense text embeddings
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["The cat sat on the mat", "A kitten on a rug"])
similarity = np.dot(embeddings[0], embeddings[1])  # 0.82

# Sparse embeddings (SPLADE)
sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning"])
print(results[0].indices, results[0].values)

# Reranking
reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank("What is deep learning?", [
    "Deep learning uses neural networks",
    "The weather is sunny today",
])
print(ranked[0].score, ranked[0].index)  # highest relevance first
```

### C/C++

```c
#include <libembedding/text_embedding.h>

lembed_text_options_t opts = lembed_text_options_default();
lembed_text_embedding_t* embedder = NULL;
lembed_text_embedding_create(&opts, &embedder);

const char* texts[] = { "Hello world", "How are you?" };
lembed_embeddings_t result = {0};
lembed_text_embedding_embed(embedder, texts, 2, 0, &result);
// result.data = float[2][384], L2-normalized

lembed_embeddings_free(&result);
lembed_text_embedding_free(embedder);
```

---

## Why libembedding-ng

### Unified ONNX + GGUF runtime

Most embedding libraries lock you into one backend. libembedding-ng lets you run
ONNX models and GGUF models through the **same API**:

```python
# Same code path, different model format
onnx_model = TextEmbedding("BAAI/bge-small-en-v1.5")          # ONNX Runtime
gguf_model = TextEmbedding.from_gguf(                          # llama.cpp
    "Xenova/all-MiniLM-L6-v2-GGUF",
    filename="all-MiniLM-L6-v2-Q4_K_M.gguf",
    provider="llama.cpp",
)
```

This matters because GGUF quantized models are smaller, faster on CPU, and avoid
ONNX Runtime dependencies — while ONNX remains the best choice for GPU and
maximum throughput.

### Native performance, two languages

The core is written in C/C++17 with a pure C API (`extern "C"`) for maximum FFI
compatibility. Python bindings use `cffi` to call into the same native code path
with minimal overhead (~13%).

- **Linux / macOS** — header-only INTERFACE library (STB-style `LIBEMBEDDING_IMPLEMENTATION`)
- **Windows** — native DLL built by CMake, exposed to Python through cffi
- **No Rust, no Go, no external server** — just C/C++ and Python

### Complete modality coverage

| Modality | Backend | Models |
|----------|---------|--------|
| Dense text | ONNX / GGUF | BGE, MiniLM, Nomic, E5, GTE, ModernBERT, EmbeddingGemma, ... |
| Sparse text | ONNX | SPLADE++, BGE-M3 |
| Image | ONNX | CLIP ViT-B-32, ResNet-50, Nomic Vision, Unicom |
| Reranking | ONNX | BGE Reranker, Jina Reranker (FP32 + INT8 quantized) |

### Production-grade tooling

- **Autotuner** — single API (`autotune_unified`) for text, sparse, image, and reranker workloads
- **Autotune cache with hardware fingerprint** — instant re-configuration across machines with similar hardware
- **Automatic model selection** — `auto_select_model()` picks the best model for your hardware
- **Session pooling** — `TextEmbeddingPool` for inter-session parallelism (up to ~4x throughput)
- **Streaming embeddings** — `embed_stream` for constant-memory processing of large corpora
- **Unified benchmark** — compare ONNX vs llama.cpp backends on the same corpus
- **Runtime introspection** — `desc()`, `model_name()`, `max_length()`, `stats()`

### Local-first, no strings attached

- Load models from disk (`model.onnx` + `tokenizer.json`) — no registry required
- Offline mode — `offline=1` prevents any network access
- Bring your own ONNX model — `create_custom()` accepts raw bytes
- GGUF models run entirely through llama.cpp — no ONNX Runtime needed

---

## Features

- **44 text embedding models** (BGE, MiniLM, Nomic, E5, CLIP, Jina, GTE, Snowflake, ModernBERT, EmbeddingGemma, etc.) with quantized variants
- **2 sparse embedding models** (SPLADE++, BGE-M3)
- **6 image embedding models** (CLIP ViT-B-32, ResNet-50, Unicom, Nomic Vision, CLIP ViT-B-32 INT8 quantized)
- **5 reranker models** (BGE Reranker, Jina Reranker, Jina V1 Turbo INT8 quantized)
- **llama.cpp / GGUF backend** — run any `.gguf` embedding model (Q4_K_M, Q8_0, etc.)
- **Unified auto-tuner** — single API (`autotune_unified`) for tuning text, sparse, image, and reranker workloads
- **Autotune cache with hardware fingerprint** — results cached by CPU + OS + RAM + software version for instant re-configuration
- **Automatic model selection** — `auto_select_model()` picks the best model for your hardware and use case
- **Quantized image and reranker models** (INT8, ~4x smaller, minimal accuracy impact)
- **Python bindings** via `pip install libembedding-ng` — drop-in fastembed replacement
- Automatic model downloading and caching from HuggingFace Hub
- Pure C API (`extern "C"`) for maximum FFI compatibility
- **Dual distribution**: header-only `INTERFACE` library on Linux/macOS (STB-style `#define LIBEMBEDDING_IMPLEMENTATION`), plus a **compiled shared library** for FFI/bindings and a native **Windows DLL** (built by CMake, exposed to Python through cffi)
- CLS and Mean pooling with L2 normalization
- Batch processing with configurable batch sizes
- CPU, CUDA, CoreML, DirectML, TensorRT execution providers
- Custom/user-defined model support (bring your own ONNX)
- Local model loading from a directory (`model.onnx` + `tokenizer.json`, optional `config.json`)
- Runtime introspection: `desc()`, `model_name()`, `max_length()`, `stats()` (texts, batches, latency)
- Similarity helpers: cosine, dot product, euclidean distance
- Streaming embeddings (`embed_stream`) for constant-memory processing of large corpora
- Multi-worker `TextEmbeddingPool` (inter-session parallelism, up to ~4x throughput)
- Autotuner and automatic model selection for optimal CPU configuration
- **Unified benchmark** — compare ONNX vs llama.cpp backends on the same corpus
- Offline mode (cache-only, no downloads)

---

## Requirements

| Dependency | Required | Notes |
|---|---|---|
| **ONNX Runtime** >= 1.16 | Yes | Bundled in PyPI wheels and on Windows. On macOS/Linux, copied next to executables at build time. |
| **llama.cpp** | Yes | Fetched via CMake `FetchContent` (v0.3.0). Always enabled — provides the GGUF backend. |
| **libcurl** >= 7.0 | Optional | For model downloading. Copied next to executables on all platforms. Disabled with `-DLIBEMBEDDING_NO_DOWNLOAD=ON` |
| **cJSON** | Bundled | Included in `third_party/` |
| **stb_image** | Bundled | Included in `third_party/`. Disable with `-DLIBEMBEDDING_NO_IMAGE=ON` |
| **CMake** >= 3.18 | Build only | |
| **C++17 compiler** | Build only | GCC 7+, Clang 5+, MSVC 2017+ |

No Rust toolchain required. The tokenizer is implemented natively in C++ (supports WordPiece and BPE models via `tokenizer.json`).

---

## Installation

### Python

```bash
pip install libembedding-ng
```

### C/C++ Build

```bash
./build.ps1           # Release build (Windows PowerShell)
./build.ps1 Debug     # Debug build
```

Or manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### Run Tests

```powershell
.\run_tests.ps1                # Unit tests only (no network)
.\run_tests.ps1 --integration  # Include integration tests (downloads models)
```

### Run Examples

```powershell
.\run_examples.ps1 basic_embedding
.\run_examples.ps1 batch_embedding
```

---

## Integration Guide

### Step 1: Add to Your Project

libembedding ships as a **header-only `INTERFACE` library** on Linux/macOS (copy `include/libembedding/` or use `FetchContent`) and as a **compiled shared library / DLL** on Windows and for the Python bindings. Copy the `include/libembedding/` directory into your project, or use CMake's `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(libembedding
    GIT_REPOSITORY https://github.com/dorel14/libembedding
    GIT_TAG main
)
FetchContent_MakeAvailable(libembedding)

target_link_libraries(your_app PRIVATE libembedding::libembedding)
```

> This is the maintained fork. The original upstream is
> [pacifio/libembedding](https://github.com/pacifio/libembedding). On Windows, link
> against the built `libembedding.dll` instead of compiling the implementation inline.

Or with an existing checkout:

```cmake
add_subdirectory(path/to/libembedding)
target_link_libraries(your_app PRIVATE libembedding::libembedding)
```

### Step 2: Implementation File

Create **exactly one** `.cpp` file in your project that defines the implementation:

```cpp
// embedding_impl.cpp
#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
```

This compiles all the library implementation code into a single translation unit. The implementation file **must** be C++ (`.cpp`) because the internal detail headers use C++17. All other files in your project include headers normally and can use the C API from either C or C++.

### Step 3: Use the API

```cpp
// any_file.cpp (or .c if only using declarations, not LIBEMBEDDING_IMPLEMENTATION)
#include <libembedding/text_embedding.h>

void my_function(void) {
    lembed_text_options_t opts = lembed_text_options_default();
    lembed_text_embedding_t* embedder = NULL;

    lembed_text_embedding_create(&opts, &embedder);
    // ... use embedder ...
    lembed_text_embedding_free(embedder);
}
```

> **Note:** On Linux/macOS, files that `#define LIBEMBEDDING_IMPLEMENTATION` must be compiled as C++ (`.cpp`); files that only call the C API (without the implementation define) can be plain C. On Windows, libembedding is consumed as a prebuilt `libembedding.dll` (built by CMake) and linked through its import library / cffi bindings, so no in-project implementation file is required.

---

## Performance Recommendations

### llama.cpp Backend

For CPU-bound embedding workloads with small BERT-style models, the optimal configuration is:

- **threads = 1** per session
- **workers/sessions = physical_cores × 2** (capped at 8)
- **batch_size = 8-32**
- **Use `LENGTH_BUCKET` batching** when text lengths vary significantly

```python
from libembedding import TextEmbedding

# Auto-detect optimal worker count
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    auto_workers=True,        # auto-detect sessions
    cache_size=4096,          # optional LRU cache
)

# Or use a preset mode
model = TextEmbedding.from_mode("balanced")
```

### ONNX Backend

- Use `LENGTH_BUCKET` batch strategy to minimize padding overhead
- Set `threads` to the number of physical cores
- `batch_size = 32-128` depending on available RAM

### Verified Baseline (i7-1065G7, MiniLM-L6-v2 Q4_K_M)

| Sessions | Threads | Strategy | Throughput |
|----------|---------|----------|------------|
| 1 | 1 | naive | 41.8 docs/s |
| 1 | 4 | naive | 72.1 docs/s |
| 6 | 1 | naive | **125.9 docs/s** |
| 8 | 1 | naive | 129.4 docs/s |
| 6 | 2 | naive | 87.6 docs/s |

Key takeaway: intra-session multithreading beyond 1 thread degrades throughput for embedding models. Prefer more single-threaded sessions.

---

## C API Reference

### Error Handling

Every function returns `lembed_status_t`. On failure, call `lembed_last_error()` for a detailed message (thread-local).

```c
lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);
if (s != LEMBED_OK) {
    fprintf(stderr, "Error [%s]: %s\n",
            lembed_status_message(s),  // e.g. "ONNX Runtime error"
            lembed_last_error());      // e.g. "Failed to load model: ..."
}
```

Status codes:

| Code | Meaning |
|---|---|
| `LEMBED_OK` | Success |
| `LEMBED_ERROR_INVALID_ARGUMENT` | NULL pointer or out-of-range enum |
| `LEMBED_ERROR_OUT_OF_MEMORY` | malloc failed |
| `LEMBED_ERROR_ONNX_RUNTIME` | ONNX Runtime session or inference error |
| `LEMBED_ERROR_TOKENIZER` | Tokenizer loading or encoding error |
| `LEMBED_ERROR_DOWNLOAD` | Model download failed |
| `LEMBED_ERROR_IO` | File I/O error |
| `LEMBED_ERROR_MODEL_NOT_FOUND` | Unknown model enum |
| `LEMBED_ERROR_UNSUPPORTED` | Feature disabled at compile time |
| `LEMBED_ERROR_BATCH_SIZE` | Batch size incompatible with dynamic quantization |
| `LEMBED_ERROR_LLAMA` | llama.cpp backend error (model load, inference, etc.) |

---

### Text Embedding

Generate dense vector representations of text. The default model is `BAAI/bge-small-en-v1.5` (384 dimensions).

```c
#include <libembedding/text_embedding.h>

// 1. Configure options
lembed_text_options_t opts = lembed_text_options_default();
opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;  // or any LEMBED_TEXT_* enum
opts.num_threads = 4;                        // 0 = auto

// 2. Create embedder (downloads model on first use)
lembed_text_embedding_t* embedder = NULL;
lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

// 3. Embed texts
const char* texts[] = { "Hello world", "How are you?" };
lembed_embeddings_t result = {0};
s = lembed_text_embedding_embed(embedder, texts, 2, 0 /* batch_size */, &result);

// 4. Access results
//    result.data  — flat float array [num_embeddings * dim]
//    result.dim   — embedding dimension (e.g. 384)
//    result.num_embeddings — number of embeddings
for (int i = 0; i < result.num_embeddings; i++) {
    float* vec = result.data + i * result.dim;
    // vec[0..dim-1] is the L2-normalized embedding for texts[i]
}

// 5. Cleanup
lembed_embeddings_free(&result);
lembed_text_embedding_free(embedder);
```

**Available text models** (44 total):

| Enum | Model | Dim | Pooling |
|---|---|---|---|
| `LEMBED_TEXT_BGE_SMALL_EN_V15` | BAAI/bge-small-en-v1.5 (default) | 384 | CLS |
| `LEMBED_TEXT_ALL_MINILM_L6_V2` | sentence-transformers/all-MiniLM-L6-v2 | 384 | Mean |
| `LEMBED_TEXT_BGE_BASE_EN_V15` | BAAI/bge-base-en-v1.5 | 768 | CLS |
| `LEMBED_TEXT_BGE_LARGE_EN_V15` | BAAI/bge-large-en-v1.5 | 1024 | CLS |
| `LEMBED_TEXT_NOMIC_EMBED_TEXT_V15` | nomic-ai/nomic-embed-text-v1.5 | 768 | Mean |
| `LEMBED_TEXT_BGE_M3` | BAAI/bge-m3 (multilingual) | 1024 | CLS |
| `LEMBED_TEXT_MULTILINGUAL_E5_LARGE` | intfloat/multilingual-e5-large | 1024 | Mean |
| `LEMBED_TEXT_GTE_LARGE_EN_V15` | Alibaba-NLP/gte-large-en-v1.5 | 1024 | CLS |
| `LEMBED_TEXT_CLIP_VIT_B32` | openai/clip-vit-base-patch32 (text) | 512 | Mean |
| ... | *Use `lembed_list_text_models()` for the full list* | | |

Quantized variants (suffix `_Q`) are available for most models.

---

### Sparse Text Embedding

Generate sparse vectors (term weights) using SPLADE++ or BGE-M3.

```c
#include <libembedding/sparse_text_embedding.h>

lembed_sparse_options_t opts = lembed_sparse_options_default();
// opts.model = LEMBED_SPARSE_SPLADE_PP_V1;  // default
// opts.model = LEMBED_SPARSE_BGE_M3;

lembed_sparse_embedding_ctx_t* embedder = NULL;
lembed_sparse_text_embedding_create(&opts, &embedder);

const char* texts[] = { "machine learning algorithms" };
lembed_sparse_embeddings_t result = {0};
lembed_sparse_text_embedding_embed(embedder, texts, 1, 0, &result);

// Each result has (indices, values, length)
for (int i = 0; i < result.items[0].length; i++) {
    int32_t token_id = result.items[0].indices[i];
    float weight     = result.items[0].values[i];
    // Use for sparse retrieval (e.g. inverted index lookup)
}

lembed_sparse_embeddings_free(&result);
lembed_sparse_text_embedding_free(embedder);
```

---

### Image Embedding

Generate dense vectors from images using CLIP or ResNet models.

```c
#include <libembedding/image_embedding.h>

lembed_image_options_t opts = lembed_image_options_default();
// opts.model = LEMBED_IMAGE_CLIP_VIT_B32;  // default, 512-dim

lembed_image_embedding_t* embedder = NULL;
lembed_image_embedding_create(&opts, &embedder);

// From file paths
const char* files[] = { "photo.jpg", "diagram.png" };
lembed_embeddings_t result = {0};
lembed_image_embedding_embed_files(embedder, files, 2, 0, &result);

// Or from memory buffers
// lembed_image_embedding_embed_bytes(embedder, buffers, sizes, 2, 0, &result);

lembed_embeddings_free(&result);
lembed_image_embedding_free(embedder);
```

| Enum | Model | Dim |
|---|---|---|
| `LEMBED_IMAGE_CLIP_VIT_B32` | CLIP ViT-B/32 (default) | 512 |
| `LEMBED_IMAGE_CLIP_VIT_B32_QUANTIZED` | CLIP ViT-B/32 INT8 quantized | 512 |
| `LEMBED_IMAGE_RESNET50` | ResNet-50 | 2048 |
| `LEMBED_IMAGE_UNICOM_VIT_B16` | Unicom ViT-B/16 | 768 |
| `LEMBED_IMAGE_UNICOM_VIT_B32` | Unicom ViT-B/32 | 512 |
| `LEMBED_IMAGE_NOMIC_EMBED_VISION_V15` | Nomic embed vision v1.5 | 768 |

---

### Reranker

Score and sort documents by relevance to a query using cross-encoder models.

```c
#include <libembedding/reranker.h>

lembed_reranker_options_t opts = lembed_reranker_options_default();
// opts.model = LEMBED_RERANKER_BGE_BASE;  // default

lembed_reranker_t* reranker = NULL;
lembed_reranker_create(&opts, &reranker);

const char* query = "What is deep learning?";
const char* docs[] = {
    "Deep learning uses neural networks with many layers",
    "The weather is sunny today",
    "Neural networks are inspired by biological brains"
};

lembed_rerank_results_t result = {0};
lembed_reranker_rerank(reranker, query, docs, 3, 0, &result);

// Results are pre-sorted by score (descending)
for (int i = 0; i < result.count; i++) {
    printf("#%d score=%.4f doc[%d]=\"%s\"\n",
           i + 1, result.items[i].score,
           result.items[i].index,
           docs[result.items[i].index]);
}

lembed_rerank_results_free(&result);
lembed_reranker_free(reranker);
```

| Enum | Model |
|---|---|
| `LEMBED_RERANKER_BGE_BASE` | BAAI/bge-reranker-base |
| `LEMBED_RERANKER_BGE_V2_M3` | BAAI/bge-reranker-v2-m3 (multilingual) |
| `LEMBED_RERANKER_JINA_V1_TURBO_EN` | jinaai/jina-reranker-v1-turbo-en |
| `LEMBED_RERANKER_JINA_V2_BASE_MULTILINGUAL` | jinaai/jina-reranker-v2-base-multilingual |
| `LEMBED_RERANKER_JINA_V1_TURBO_EN_QUANTIZED` | jinaai/jina-reranker-v1-turbo-en (INT8) |

---

### Model Registry

Query available models at runtime without creating an embedder.

```c
#include <libembedding/model_registry.h>

// List all text models
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

// Get info for a specific model
lembed_model_info_t info;
lembed_get_text_model_info(LEMBED_TEXT_BGE_SMALL_EN_V15, &info);
printf("Code: %s, File: %s\n", info.model_code, info.model_file);

// Find model by HuggingFace repo code
int idx = lembed_find_text_model_by_code("Xenova/bge-small-en-v1.5");
// idx == LEMBED_TEXT_BGE_SMALL_EN_V15
```

The `lembed_model_info_t` struct:

```c
typedef struct {
    const char* model_name;    // e.g. "BAAI/bge-small-en-v1.5"
    const char* model_code;    // HF repo, e.g. "Xenova/bge-small-en-v1.5"
    const char* model_file;    // e.g. "onnx/model.onnx"
    const char* description;
    int         dim;           // embedding dimension
    int         max_tokens;    // max input token length
    int         pooling;       // LEMBED_POOLING_CLS or LEMBED_POOLING_MEAN
    int         quantization;  // LEMBED_QUANTIZATION_NONE/STATIC/DYNAMIC
} lembed_model_info_t;
```

---

### Custom Models (Bring Your Own ONNX)

Use any ONNX model not in the built-in registry:

```c
#include <libembedding/text_embedding.h>

// Load your ONNX model and tokenizer.json into memory
unsigned char* onnx_bytes = load_file("my_model.onnx", &onnx_size);
unsigned char* tok_bytes  = load_file("tokenizer.json", &tok_size);

lembed_user_defined_model_t model = {
    .onnx_data          = onnx_bytes,
    .onnx_data_size     = onnx_size,
    .tokenizer_json     = tok_bytes,
    .tokenizer_json_size = tok_size,
    .pooling            = LEMBED_POOLING_MEAN,
    .dim                = 768,
    .max_length         = 512,
};

lembed_text_embedding_t* embedder = NULL;
lembed_text_embedding_create_custom(&model, LEMBED_PROVIDER_CPU, 0, &embedder);

// Use exactly like built-in models
lembed_embeddings_t result = {0};
lembed_text_embedding_embed(embedder, texts, n, 0, &result);

lembed_embeddings_free(&result);
lembed_text_embedding_free(embedder);
```

---

### Options Reference

All option structs have a `_default()` constructor that returns safe defaults:

```c
typedef struct {
    lembed_text_model_t         model;           // default: LEMBED_TEXT_BGE_SMALL_EN_V15
    lembed_execution_provider_t provider;       // default: LEMBED_PROVIDER_CPU
    int                         device_id;      // default: 0
    const char*                 cache_dir;      // default: NULL (~/.cache/libembedding)
    int                         max_length;     // default: 0 (model's default)
    int                         num_threads;    // default: 0 (auto)
    int                         show_download_progress; // default: 1
    int                         batch_size;     // default: 256 (LEMBED_DEFAULT_BATCH_SIZE)
    int                         offline;        // default: 0 (0=allow download, 1=cache-only)
    int                         pooling;        // default: LEMBED_POOLING_MEAN (for local models)
    int                         dim;            // default: 0 (for local models without config.json)
    /* llama.cpp backend options (only used when provider = LEMBED_PROVIDER_LLAMACPP) */
    int                         llama_n_ctx;    // context size (0 = model default)
    int                         llama_n_gpu_layers; // GPU layers (-1 = all, 0 = CPU only)
    int                         llama_verbose;  // 1 = enable llama.cpp logging
} lembed_text_options_t;
```

The `batch_size`, `offline`, `pooling`, and `dim` fields are available in all option structs (`lembed_sparse_options_t`, `lembed_image_options_t`, `lembed_reranker_options_t`).

**Execution providers:**

| Enum | Backend |
|---|---|
| `LEMBED_PROVIDER_CPU` | CPU (default, always available) |
| `LEMBED_PROVIDER_CUDA` | NVIDIA CUDA (requires `USE_CUDA` + ORT CUDA provider) |
| `LEMBED_PROVIDER_COREML` | Apple CoreML (macOS/iOS) |
| `LEMBED_PROVIDER_DIRECTML` | DirectML (Windows, requires `USE_DML` + ORT DML provider) |
| `LEMBED_PROVIDER_TENSORRT` | NVIDIA TensorRT (requires `USE_TENSORRT`) |
| `LEMBED_PROVIDER_LLAMACPP` | llama.cpp backend for GGUF models (always enabled) |

Providers are configured via `configure_provider()` and gracefully fall back to CPU if the provider library is unavailable.

---

### Local Model Loading (`create_from_path`)

Load models from a local directory containing `model.onnx` + `tokenizer.json`:

```c
#include <libembedding/text_embedding.h>

lembed_text_options_t opts = lembed_text_options_default();
opts.dim = 384;
opts.pooling = LEMBED_POOLING_MEAN;

lembed_text_embedding_t* embedder = NULL;
/* dir_path must contain model.onnx, tokenizer.json, and optionally config.json */
lembed_status_t s = lembed_text_embedding_create_from_path(
    "/path/to/model_dir", &opts, &embedder);

if (s == LEMBED_OK) {
    /* Use exactly like built-in models */
    lembed_embeddings_t result = {0};
    lembed_text_embedding_embed(embedder, texts, n, 0, &result);
    lembed_embeddings_free(&result);
    lembed_text_embedding_free(embedder);
}
```

If `config.json` is present, `dim`, `max_length`, and `pooling` are auto-detected. Without it, specify via the options struct.

`create_from_path` is available for all model types: `lembed_text_embedding_create_from_path()`, `lembed_sparse_text_embedding_create_from_path()`, `lembed_image_embedding_create_from_path()`, `lembed_reranker_create_from_path()`.

---

### llama.cpp / GGUF Backend

In addition to ONNX Runtime, libembedding can load GGUF embedding models through the llama.cpp backend. This allows running quantized models (Q4_K_M, Q8_0, etc.) that are smaller and faster on certain hardware.

The llama.cpp backend is always enabled by default, so no extra flag is required:

```bash
cmake ..
cmake --build . --parallel
```

Or via CMake, point to a custom llama.cpp installation if needed:

```cmake
set(LLAMACPP_ROOT "/path/to/llama.cpp")
```

llama.cpp is fetched via CMake `FetchContent` (v0.3.0) by default. The `LEMBED_PROVIDER_LLAMACPP` execution provider and GGUF loading functions are always available:

```c
#include <libembedding/gguf_registry.h>
#include <libembedding/text_embedding.h>
#include <libembedding/llamacpp_backend.h>

/* Check if llama.cpp backend is compiled in */
int available = lembed_llama_backend_available();  /* 1 if compiled, 0 otherwise */
const char* ver = lembed_llama_version();          /* version string */

/* Load a GGUF model from a local file */
lembed_text_options_t opts = lembed_text_options_default();
opts.provider = LEMBED_PROVIDER_LLAMACPP;

lembed_text_embedding_t* embedder = NULL;
lembed_text_embedding_create_from_gguf_path(
    "/path/to/model.Q4_K_M.gguf", &opts, &embedder);

/* List recommended GGUF models */
const lembed_gguf_model_info_t* models;
int count;
lembed_list_gguf_models(&models, &count);
for (int i = 0; i < count; i++) {
    printf("%-32s dim=%-4d quality=%.3f\n",
           models[i]->name, models[i]->dim, models[i]->quality_mteb);
}
```

**Recommended GGUF models** are listed in the registry. Query them with `lembed_list_gguf_models()`, `lembed_find_gguf_model()`, or `lembed_default_gguf_model()`.

---

### Unified Auto-Tuner and Benchmark

libembedding includes a comprehensive auto-tuning system that optimizes thread count, batch size, and worker count for your specific hardware. The unified API covers all task types:

```python
from libembedding import autotune, autotune_unified, LEMBED_TASK_EMBEDDING, LEMBED_AUTOTUNE_QUICK

# Quick auto-tune for text embedding (5-15s)
result = autotune("BAAI/bge-small-en-v1.5", LEMBED_AUTOTUNE_QUICK)
print(result.threads, result.batch_size, result.workers)

# Unified API — same pattern for all task types
from libembedding import autotune_unified, LEMBED_TASK_SPARSE, LEMBED_TASK_IMAGE, LEMBED_TASK_RERANKING
result = autotune_unified(LEMBED_TASK_RERANKING, "BAAI/bge-reranker-base", LEMBED_AUTOTUNE_FULL)
```

The **unified benchmark** compares ONNX and llama.cpp backends on identical corpora:

```python
from libembedding import Benchmark, CorpusType, Objective

bench = Benchmark()
print(bench.hardware)   # Hardware fingerprint
print(bench.software)   # Software versions

# Compare backends for the same model
comparison = bench.compare_all(
    onnx_path="/path/to/model.onnx",
    gguf_path="/path/to/model.Q4_K_M.gguf",
    corpus=CorpusType.MIXED,
    objective=Objective.BALANCED,
)
print(comparison.recommendation)
```

**Autotune cache**: Results are cached using a hardware fingerprint (CPU name, core count, RAM, OS, SIMD features) combined with software versions and model quantization. Subsequent runs load instantly from `lembed_tune_cache_path()`.

**Automatic model selection**: Let libembedding pick the best model for your hardware:

```python
from libembedding import auto_select_model

best = auto_select_model("balanced")  # or "speed" / "quality"
print(best.model_name, best.dim, best.threads)
```

---

### Offline Mode

Use `options.offline = 1` to require models be available in cache without attempting downloads:

```c
lembed_text_options_t opts = lembed_text_options_default();
opts.offline = 1;  // error if model not cached, never downloads
lembed_text_embedding_create(&opts, &embedder);
```

For fully embedded builds (no network/curl dependency at compile time), use `-DLIBEMBEDDING_NO_DOWNLOAD=ON` or `#define LIBEMBEDDING_NO_DOWNLOAD`.

---

### Runtime Introspection (`desc()`, `model_name()`, `max_length()`, `stats()`)

Query runtime configuration of a created context:

```c
/* Get model descriptor */
const lembed_model_desc_t* desc = lembed_text_embedding_desc(embedder);
printf("Model: %s (dim=%d, threads=%d, batch=%d, provider=%d)\n",
       desc->name, desc->dimension, desc->num_threads, desc->batch_size, desc->provider);

/* Convenience accessors */
printf("Name: %s\n", lembed_text_embedding_model_name(embedder));
printf("Max length: %d\n", lembed_text_embedding_max_length(embedder));

/* Runtime statistics */
lembed_stats_t stats = {0};
lembed_text_embedding_stats(embedder, &stats);
printf("Texts embedded: %llu, Batches: %llu, Avg latency: %.2f ms\n",
       stats.texts_embedded, stats.batches_run, stats.avg_latency_ms);
```

`lembed_model_desc_t` struct:
```c
typedef struct {
    const char*                name;          /* model name or local path */
    int                        dimension;     /* embedding dimension */
    int                        max_length;    /* effective max token length */
    int                        pooling;       /* CLS or MEAN */
    int                        num_threads;   /* threads configured */
    int                        batch_size;    /* batch_size configured */
    lembed_execution_provider_t provider;      /* execution provider in use */
    int                        device_id;     /* device id in use */
} lembed_model_desc_t;
```

---

### Streaming API (`embed_stream`)

Process large numbers of texts without allocating a single result buffer. Uses a callback pattern — each embedding is delivered individually:

```c
void on_embedding(const float* embedding, int dim, void* userdata) {
    /* Do something with this single embedding */
    /* userdata is passed through from the caller */
}

lembed_text_embedding_embed_stream(
    embedder, texts, num_texts,
    0 /* batch_size = use context default */,
    on_embedding, userdata);
```

---

### Similarity Functions

Native C functions for comparing embedding vectors:

```c
#include <libembedding/similarity.h>

float sim = lembed_cosine_similarity(vec_a, vec_b, dim);
float dot = lembed_dot_product(vec_a, vec_b, dim);
float dist = lembed_euclidean_distance(vec_a, vec_b, dim);
```

---

### Version

```c
#include <libembedding/config.h>
const char* version = lembed_version();  /* e.g. "1.4.0" */
```

---

### Memory Management

The library allocates output buffers internally. Always free them with the matching free function:

| Allocator | Free function |
|---|---|
| `lembed_text_embedding_embed()` | `lembed_embeddings_free(&result)` |
| `lembed_sparse_text_embedding_embed()` | `lembed_sparse_embeddings_free(&result)` |
| `lembed_image_embedding_embed_*()` | `lembed_embeddings_free(&result)` |
| `lembed_reranker_rerank()` | `lembed_rerank_results_free(&result)` |
| `lembed_text_embedding_create()` | `lembed_text_embedding_free(ctx)` |
| `lembed_ensure_*_model()` | `lembed_free_string(path)` |

---

### Environment Variables

| Variable | Purpose |
|---|---|
| `LIBEMBEDDING_CACHE_DIR` | Override default model cache directory |
| `FASTEMBED_CACHE_DIR` | Alternative cache dir (compatibility with fastembed) |
| `HF_ENDPOINT` | Custom HuggingFace Hub endpoint URL |
| `ONNXRUNTIME_ROOT` | Path to ONNX Runtime installation (for CMake) |

---

## Compile-Time Configuration

Define these **before** including any libembedding header:

```c
#define LIBEMBEDDING_NO_DOWNLOAD  // Disable model downloading (offline only)
#define LIBEMBEDDING_NO_IMAGE     // Disable image embedding (removes stb dependency)
```

Or via CMake:

```bash
cmake .. -DLIBEMBEDDING_NO_DOWNLOAD=ON -DLIBEMBEDDING_NO_IMAGE=ON
```

CMake options:

| Option | Default | Description |
|---|---|---|
| `LIBEMBEDDING_NO_DOWNLOAD` | OFF | Disable model downloading (offline-only) |
| `LIBEMBEDDING_NO_IMAGE` | OFF | Disable image embedding (removes stb_image dependency) |
| `LIBEMBEDDING_BUILD_SHARED` | OFF | Build shared library / DLL (required for Python bindings) |
| `LIBEMBEDDING_BUILD_BENCHMARKS` | OFF | Build benchmark executables |
| `LIBEMBEDDING_BUILD_TESTS` | ON | Build unit tests |

---

## Benchmarks

Measured on Apple M-series (macOS arm64) with `all-MiniLM-L6-v2` (384-dim). Median of 10 runs, 1 warmup, pre-cached models.

### All implementations

| Metric                   | libembedding C++ | libembedding Python | fastembed-rs (Rust) | fastembed (Python) |
|--------------------------|------------------|---------------------|---------------------|--------------------|
| Model load (ms)          | **81**           | **79**              | 88                  | 84                 |
| Single text latency (ms) | **3.9**          | **4.4**             | 1.9                 | 38.0               |
| Batch 8 (texts/sec)      | **632**          | **641**             | 231                 | 92                 |
| Batch 32 (texts/sec)     | **687**          | **581**             | 326                 | 89                 |
| Batch 128 (texts/sec)    | **626**          | **449**             | 402                 | 90                 |
| Batch 512 (texts/sec)    | **526**          | **476**             | 398                 | 80                 |
| Peak RSS (MB)            | 717              | **567**             | 672                 | 1,981              |

### Python: libembedding vs fastembed (drop-in replacement)

| Metric                   | libembedding | fastembed | Speedup     |
|--------------------------|-------------|-----------|-------------|
| Single text latency (ms) | **4.4**     | 38.0      | **8.6x**    |
| Batch 8 (texts/sec)      | **641**     | 92        | **7.0x**    |
| Batch 32 (texts/sec)     | **581**     | 89        | **6.5x**    |
| Batch 128 (texts/sec)    | **449**     | 90        | **5.0x**    |
| Peak RSS (MB)            | **567**     | 1,981     | **3.5x less** |

**Key takeaways:**
- `pip install libembedding` is a **5-8x faster** drop-in replacement for fastembed
- **8.6x faster single-text latency** (4.4ms vs 38ms) -- the C backend does the heavy lifting
- **3.5x less memory** (567MB vs 1.98GB peak RSS)
- C++ and Python share the same backend -- Python adds only 13% overhead (4.4ms vs 3.9ms)
- C++ API is **1.7-2.7x faster** than fastembed-rs (Rust) across batch sizes

### Reproducing benchmarks

```bash
# C++, Rust, and fastembed (Python) benchmarks
cd benchmarks
pip3 install fastembed
cd bench_fastembed_rs && cargo build --release && cd ..
cd .. && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLIBEMBEDDING_BUILD_BENCHMARKS=ON -DLIBEMBEDDING_BUILD_SHARED=ON
cmake --build . --parallel
cd ../benchmarks && ./run_benchmarks.sh

# Python bindings vs fastembed
pip install libembedding
PYTHONPATH=../python/src python3 bench_python_compare.py
```

---

## Performance Characteristics

Based on comprehensive benchmarks (Intel i7-1065G7, Windows 11):

### Key Findings

- **Python bindings add ~13% overhead** compared to native C++ (4.4ms vs 3.9ms single-text)
- **Request-level parallelism** (multiple ONNX sessions) significantly outperforms ORT intra-op threading on small Transformers
- **Dynamic INT8 quantized models** may produce embeddings that vary slightly with batch composition (cosine ≈ 0.984 vs FP32 baseline)
- **ONNX Runtime memory** stabilizes after warmup (~70 MB) with no observable growth
- **Throughput is strongly dependent on input length** - always interpret docs/s together with token counts

### Optimal Configuration for CPU

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `workers` | 8 (or CPU cores) | Request-level parallelism |
| `threads` | 1 per worker | Avoids ORT intra-op contention |
| `batch_size` | 32-64 | Balances throughput/latency |

### Text Length Impact (MiniLM-L6-v2, 8 workers)

| Tokens | Docs/s | ms/text |
|--------|--------|---------|
| 16 | 166 | 6.0 |
| 64 | 59 | 16.9 |
| 128 | 31 | 32.6 |
| 256 | 15 | 68.6 |

### Model Comparison (8 workers, ~16 tok/text)

| Model | Docs/s | RAM | Deterministic |
|-------|--------|-----|---------------|
| MiniLM-L6-v2-Q | 474-696 | 230 MB | ~1.6% variance |
| MiniLM-L6-v2 | 305-361 | 740 MB | Yes |
| BGE-small-en | 143-150 | 1.1 GB | Yes |

### Python Multi-Worker Pool

```python
from libembedding import TextEmbeddingPool

# 8 workers = ~4x throughput vs single session
pool = TextEmbeddingPool("sentence-transformers/all-MiniLM-L6-v2", workers=8)
embeddings = pool.embed(texts)
pool.close()
```

| Config | Docs/s | Speedup |
|--------|--------|---------|
| Single (4 threads) | 94 | 1.0x |
| Pool (8 workers) | 180 | ~4x |

---

## Architecture

```
                     +-----------------------+
                     |   libembedding.h      |  <-- single include
                     |   (umbrella header)   |
                     +-----------+-----------+
                                 |
       +--------+--------+---+---+---+--------+--------+
       |        |        |       |   |        |        |
    types.h  error.h  model   text  image  sparse  reranker.h
                     registry emb.  emb.   emb.
                        .h    .h    .h     .h
                               |       |        |
                     +---------+-----------+-----------+
                     |       detail/ (C++ internals)   |
                     |                                 |
                     |  onnx_session_impl.hpp          |
                     |  llama_session_impl.hpp (llama.cpp) |
                     |  tokenizer_impl.hpp (built-in)  |
                     |  pooling.hpp / normalize.hpp    |
                     |  downloader_impl.hpp            |
                     |  sparse_postprocess.hpp         |
                     |  image_preprocess.hpp           |
                     +---------------------------------+
                                 |
                     +-----------+-----------+-----------+
                     |    External deps      |           |
                     |  ONNX Runtime (C API) |  llama.cpp  |
                     |  cJSON (bundled)      |  (optional) |
                     |  stb_image (bundled)  |           |
                     |  libcurl (optional)   |           |
                     +-----------------------------+--------+
```

---

### Python llama.cpp Example

```python
from libembedding import TextEmbedding, LlamaError

# Load a GGUF model (requires build with llama.cpp support)
try:
    model = TextEmbedding.from_gguf(
        "Xenova/all-MiniLM-L6-v2-GGUF",
        filename="all-MiniLM-L6-v2-Q4_K_M.gguf",
        provider="llama.cpp",
        n_gpu_layers=0,
    )
    embeddings = model.embed(["Hello world"])
except LlamaError as e:
    print(f"llama.cpp error: {e}")
    print(f"Details: {e.last_error}")
```

The `LlamaError` exception is raised for llama.cpp-specific failures. Use `TextEmbedding.supports_llamacpp()` to check at runtime.

---

## License

MIT
