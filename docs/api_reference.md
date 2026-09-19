---
title: API Python
nav_order: 3
---

# Référence API Python

## Classes principales

### TextEmbedding

Génère des embeddings denses (vecteurs) à partir de textes.

#### Constructeur

```python
TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    provider="cpu",
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    cache_dir=None,
    max_length=0,
    dim=0,
    pooling="mean",
    auto_workers=False,
    cache_size=0,
    quantization=None,
    num_threads=None,  # déprécié
)
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"BAAI/bge-small-en-v1.5"` | Nom HuggingFace, code de repo, ou chemin local vers un répertoire contenant `model.onnx` + `tokenizer.json` |
| `provider` | `str` | `"cpu"` | Provider d'exécution : `"cpu"`, `"cuda"`, `"coreml"`, `"directml"`, `"tensorrt"`, `"llamacpp"` |
| `threads` | `int` | `0` | Nombre de threads (`0` = auto) |
| `batch_size` | `int` | `256` | Taille de batch interne pour l'inférence |
| `offline` | `bool` | `False` | `True` = utilise uniquement le cache, pas de téléchargement |
| `show_download_progress` | `bool` | `True` | Affiche la barre de progression de téléchargement |
| `cache_dir` | `str \| None` | `None` | Répertoire de cache des modèles (`None` = `~/.cache/libembedding`) |
| `max_length` | `int` | `0` | Longueur max en tokens (`0` = défaut du modèle) |
| `dim` | `int` | `0` | Dimension de l'embedding pour modèles locaux sans `config.json` |
| `pooling` | `str` | `"mean"` | Stratégie de pooling pour modèles locaux : `"cls"` ou `"mean"` |
| `auto_workers` | `bool` | `False` | `True` = auto-détection du nombre optimal de sessions/workers pour llama.cpp |
| `cache_size` | `int` | `0` | Taille du cache LRU d'embeddings (`0` = désactivé) |
| `quantization` | `str \| None` | `None` | Mode de quantification : `"none"`, `"static"`, `"dynamic"` (None = défaut du modèle) |
| `num_threads` | `int \| None` | `None` | **Déprécié** — utiliser `threads` à la place |

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=None)` | `np.ndarray` | Embed les textes. Retourne un tableau de forme `(n, dim)` en `float32`. L2-normalisé. |
| `embed_stream(texts, callback, batch_size=None)` | `None` | Embed en streaming — appelle `callback(array, dim, userdata)` pour chaque embedding |
| `embed_batched(texts, batch_size=None)` | `Generator` | Embed en lots, génère un tableau par lot |
| `dim` | `int` (property) | Dimension de l'embedding |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle ou chemin local |
| `info()` | `ModelDesc` | Descripteur du modèle chargé |
| `stats()` | `Stats` | Statistiques d'utilisation runtime |
| `close()` | `None` | Libère les ressources C sous-jacentes |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste tous les modèles de texte supportés |
| `__enter__()` | `self` | Support du context manager |
| `__exit__()` | `None` | Appelle `close()` automatiquement |
| `from_gguf(repo, filename=None, provider="llamacpp", ...)` | `TextEmbedding` | Charge un modèle GGUF depuis HuggingFace (nécessite le backend llama.cpp) |
| `from_mode(mode="balanced", **kwargs)` | `TextEmbedding` | (classmethod) Crée un TextEmbedding depuis un mode prédéfini : `"fast"`, `"balanced"`, `"quality"` |
| `supports_llamacpp()` | `bool` | (static) Vérifie si le backend llama.cpp est compilé |

#### Exemple

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])
print(embeddings.shape)   # (2, 384)
print(embeddings.dtype)   # float32

# Recherche sémantique simple
query = model.embed(["What is AI?"])
scores = embeddings @ query.T
best_idx = scores.argmax()
```

---

### SparseTextEmbedding

Génère des embeddings **sparse** (vecteurs creux avec indices de tokens et poids).

#### Constructeur

```python
SparseTextEmbedding(
    model_name="prithvida/SPLADE_PP_en_v1",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    max_length=0,
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    top_terms=0,
    min_weight=0.0,
    quantization=None,
    backend="auto",
    cache_size=0,
    num_threads=None,  # déprécié
)
```

**Modèles disponibles :**

| Nom HuggingFace | Description |
|-----------------|-------------|
| `prithvida/SPLADE_PP_en_v1` | SPLADE++ (défaut) |
| `BAAI/bge-m3` | BGE-M3 multilingue |

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `embed(texts, batch_size=0)` | `list[SparseEmbedding]` | Retourne une liste d'objets `SparseEmbedding` |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle |
| `info()` | `ModelDesc` | Descripteur du modèle |
| `stats()` | `Stats` | Statistiques d'utilisation runtime |
| `close()` | `None` | Libère les ressources |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste les modèles sparse |

#### Exemple

```python
from libembedding import SparseTextEmbedding

sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
    # indices : tableau int32 des token IDs
    # values : tableau float32 des poids
```

---

### ImageEmbedding

Génère des embeddings denses à partir d'images.

#### Constructeur

```python
ImageEmbedding(
    model_name="Qdrant/clip-ViT-B-32-vision",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    threads=0,
    batch_size=30,
    offline=False,
    show_download_progress=True,
    dim=0,
    num_threads=None,  # déprécié
)
```

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `embed_files(paths, batch_size=None)` | `np.ndarray` | Embed à partir de chemins de fichiers. Forme : `(n, dim)` |
| `embed_bytes(images, batch_size=None)` | `np.ndarray` | Embed à partir de données bytes brutes (JPEG, PNG, etc.) |
| `dim` | `int` (property) | Dimension de l'embedding |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle |
| `info()` | `ModelDesc` | Descripteur du modèle |
| `close()` | `None` | Libère les ressources |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste les modèles image |

#### Exemple

```python
from libembedding import ImageEmbedding

model = ImageEmbedding()
embeddings = model.embed_files(["photo.jpg", "diagram.png"])
print(embeddings.shape)  # (2, 512)
```

---

### Reranker

Score et trie des documents par pertinence par rapport à une requête (cross-encoder).

#### Constructeur

```python
Reranker(
    model_name="BAAI/bge-reranker-base",
    provider="cpu",
    device_id=0,
    cache_dir=None,
    max_length=0,
    threads=0,
    batch_size=256,
    offline=False,
    show_download_progress=True,
    backend="auto",
    quantization=None,
    cache_size=0,
    num_threads=None,  # déprécié
)
```

**Modèles disponibles :**

| Nom HuggingFace | Description |
|-----------------|-------------|
| `BAAI/bge-reranker-base` | BGE Reranker base (défaut) |
| `BAAI/bge-reranker-v2-m3` | BGE Reranker v2 multilingue |
| `jinaai/jina-reranker-v1-turbo-en` | Jina Reranker v1 turbo |
| `jinaai/jina-reranker-v2-base-multilingual` | Jina Reranker v2 multilingue |
| `jinaai/jina-reranker-v1-turbo-en` (quantized) | Jina Reranker v1 turbo English (INT8) |

#### Méthodes et propriétés

| Membre | Type | Description |
|--------|------|-------------|
| `rerank(query, documents, batch_size=0)` | `list[RerankResult]` | Retourne les résultats triés par score décroissant |
| `batch_size` | `int` (property) | Taille de batch configurée |
| `name` | `str` (property) | Nom du modèle |
| `info()` | `ModelDesc` | Descripteur du modèle |
| `stats()` | `Stats` | Statistiques d'utilisation runtime |
| `close()` | `None` | Libère les ressources |
| `list_supported_models()` | `list[ModelInfo]` | (static) Liste les modèles reranker |
| `auto(profile="balanced", **kwargs)` | `Reranker` | (static) Crée un Reranker auto-configuré par profil |

#### Exemple

```python
from libembedding import Reranker

reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank(
    "What is deep learning?",
    [
        "Deep learning uses neural networks",
        "The weather is sunny today",
        "Neural networks are inspired by biological brains",
    ],
)

for r in ranked:
    print(f"doc[{r.index}] score={r.score:.4f}")
```

---

## Types de données

### TextEmbeddingPool

Pool de sessions ONNX pour le parallélisme inter-sessions. Voir [performance_tuning.html](performance_tuning.html).

```python
@dataclass(frozen=True)
class TextEmbeddingPool:
    model_name: str
    workers: int = 0                  # nombre de sessions (0 = auto-detect)
    threads_per_worker: int = 1       # threads par session
    batch_size: int = 256
    provider: str = "cpu"
    offline: bool = False
    show_download_progress: bool = True
    cache_dir: str | None = None
    max_length: int = 0
    dim: int = 0
    pooling: str = "mean"
    autotune: bool = False            # auto-tune tous les paramètres
    autotune_texts: list[str] = None  # corpus pour l'autotune
    autotune_max_samples: int = 100   # taille d'échantillon max
    cache_size: int = 0               # LRU cache par worker (0 = désactivé)
    quantization: str | None = None   # mode de quantification
```

**Méthodes :**

| Membre | Type | Description |
|--------|------|-------------|
| `embed(texts)` | `np.ndarray` | Embed les textes en parallèle |
| `num_workers` | `int` (property) | Nombre de workers actifs |
| `dim` | `int` (property) | Dimension de l'embedding |
| `close()` | `None` | Libère les ressources |

### TuningResult

Résultat de l'autotune pour un modèle.

```python
@dataclass(frozen=True)
class TuningResult:
    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
```

### ModelSelectionResult

Résultat de la sélection automatique de modèle.

```python
@dataclass(frozen=True)
class ModelSelectionResult:
    model_code: str
    model_name: str
    dim: int
    workers: int
    threads: int
    batch_size: int
    throughput_docs_sec: float
    latency_ms: float
    memory_mb: float
    score: float
```

### Fonctions globales

| Fonction | Description |
|----------|-------------|
| `autotune(model_name, full=False)` | Auto-tune un modèle. Retourne `TuningResult`. |
| `autotune_unified(task, model_name, mode="quick")` | Auto-tune unifié pour tous les types de tâches. Voir [Unified Tuning](#unified-tuning). |
| `sparse_autotune(model_name, mode="quick")` | Auto-tune l'embedding sparse. Retourne `SparseTuningResult`. |
| `image_autotune(model_name, mode="quick")` | Auto-tune l'embedding image. Retourne `ImageTuningResult`. |
| `reranker_autotune(model_name, mode="quick", objective="latency")` | Auto-tune le reranker. Retourne `RerankerTuningResult`. |
| `reranker_auto_config(model_name, target_latency_ms=500, objective="latency")` | Configuration automatique du reranker pour un budget de latence. |
| `reranker_auto_config_profile(model_name, profile="balanced")` | Configuration du reranker via un profil. |
| `reranker_autotune_constrained(model_name, mode="quick", objective="latency", min_tokens=32, max_latency_ms=1000)` | Auto-tune avec contraintes. |
| `clear_reranker_autotune_cache(model_name=None)` | Efface le cache d'autotune du reranker. |
| `auto_select_model(use_case="balanced")` | Sélectionne le meilleur modèle. Retourne `ModelSelectionResult`. |
| `clear_autotune_cache(model_name=None)` | Efface le cache d'autotune. |
| `cache_path()` | Chemin du fichier de cache d'autotune. |

---

### Unified Tuning

L'auto-tune unifié fournit un point d'entrée unique pour tous les types de tâches (texte, sparse, image, reranking).

```python
from libembedding import autotune_unified, LEMBED_TASK_RERANKING, LEMBED_AUTOTUNE_QUICK

result = autotune_unified(
    task=LEMBED_TASK_RERANKING,
    model_name="BAAI/bge-reranker-base",
    mode=LEMBED_AUTOTUNE_QUICK,
)
```

#### UnifiedTuningResult

```python
@dataclass(frozen=True)
class UnifiedTuningResult:
    task: str
    threads: int
    batch_size: int
    workers: int
    max_tokens: int
    top_k: int
    min_weight: float
    storage_format: int
    throughput_docs_sec: float
    latency_ms: float
    p95_latency_ms: float
    memory_mb: float
```

---

### LlamaError

Exception levée pour les erreurs du backend llama.cpp/GGUF :

```python
from libembedding import LlamaError

try:
    model = TextEmbedding.from_gguf("/path/to/model.gguf")
except LlamaError as e:
    print(f"Erreur llama.cpp: {e.detail}")
```

---

### SparseEmbedding

```python
@dataclass(frozen=True)
class SparseEmbedding:
    indices: np.ndarray  # int32 — token IDs
    values:  np.ndarray  # float32 — poids des tokens
```

### RerankResult

```python
@dataclass(frozen=True)
class RerankResult:
    index: int   # index du document dans la liste d'entrée
    score: float # score de pertinence (plus élevé = plus pertinent)
```

### ModelInfo

```python
@dataclass(frozen=True)
class ModelInfo:
    model_name:    str   # ex: "BAAI/bge-small-en-v1.5"
    model_code:    str   # code HF repo, ex: "Xenova/bge-small-en-v1.5"
    model_file:    str   # ex: "onnx/model.onnx"
    description:   str
    dim:           int   # dimension de l'embedding
    max_tokens:    int   # longueur max en tokens
    pooling:       str   # "cls" ou "mean"
    quantization:  str   # "none", "static", "dynamic"
```

### ModelDesc

```python
@dataclass(frozen=True)
class ModelDesc:
    name: str
    dimension: int
    max_length: int
    pooling: str   # "cls" ou "mean"
    num_threads: int
    batch_size: int
    provider: str
    device_id: int
    quantization: str = "none"
    cache_size: int = 0
```

---

### Stats

Statistiques d'utilisation runtime pour un contexte d'embedding.

```python
@dataclass(frozen=True)
class Stats:
    texts_embedded: int    # total texts processed
    batches_run: int       # total ONNX batches executed
    avg_latency_ms: float  # average latency per embed() call (ms)
    cache_hits: int = 0    # LRU cache hits (0 if disabled)
    cache_misses: int = 0  # LRU cache misses (0 if disabled)
```

Disponible via la méthode `.stats()` sur `TextEmbedding`, `Reranker`, `ImageEmbedding` et `SparseTextEmbedding`.

### EmbeddingCache

Cache LRU thread-safe pour les embeddings denses. Peut être utilisé standalone ou via le paramètre `cache_size` des constructeurs.

```python
from libembedding import EmbeddingCache

cache = EmbeddingCache(capacity=4096, ttl_seconds=0, dim=0)
cache.put("text", np.array([...], dtype=np.float32))
vec = cache.get("text")  # np.ndarray | None
cache.clear()
```

| Méthode | Retour | Description |
|---------|--------|-------------|
| `put(text, vec)` | `None` | Stocker un embedding |
| `get(text, dim=None)` | `np.ndarray \| None` | Récupérer un embedding |
| `clear()` | `None` | Supprimer toutes les entrées |
| `capacity` | `int` | Capacité maximale |
| `current_size` | `int` | Nombre actuel d'entrées |
| `close()` | `None` | Libérer les ressources C |

---

## Fonctions de similarité

Comparaison native de vecteurs d'embedding.

| Fonction | Description |
|----------|-------------|
| `cosine_similarity(a, b)` | Similarité cosinus |
| `dot_product(a, b)` | Produit scalaire |
| `euclidean_distance(a, b)` | Distance euclidienne (L2) |

```python
from libembedding import cosine_similarity, dot_product, euclidean_distance
import numpy as np

a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
b = np.array([0.4, 0.5, 0.6], dtype=np.float32)

sim = cosine_similarity(a, b)
dp = dot_product(a, b)
dist = euclidean_distance(a, b)
```

---

## Benchmark unifié

Comparaison des backends ONNX et llama.cpp avec auto-tuning.

| Classe/Constante | Description |
|------------------|-------------|
| `Benchmark` | Benchmark avec auto-tuning et comparaison multi-backends |
| `BenchmarkResult` | Résultat pour un modèle x backend |
| `ComparisonResult` | Résultats de comparaison avec recommandation |
| `Metrics` | Métriques d'un benchmark (throughput, latencies, mémoire) |
| `HardwareInfo` | Informations hardware détectées |
| `CorpusType` | Catégories de corpus (SHORT, MEDIUM, LONG, MIXED, MULTILINGUAL, EDGE_CASES) |
| `Objective` | Objectifs d'optimisation (LATENCY, THROUGHPUT, BALANCED, MEMORY) |

```python
from libembedding import Benchmark, Objective, detect_backend, cache_path, clear_cache

# Détection backend
backend = detect_backend("BAAI/bge-small-en-v1.5")

# Benchmark
bench = Benchmark()
result = bench.autotune("path/to/model.gguf", "llama.cpp", objective=Objective.BALANCED)
comparison = bench.compare_all(onnx_path="model.onnx", gguf_path="model.gguf")

# Utilitaires
path = cache_path()
clear_cache()
```

---

## Gestion des erreurs

Toutes les exceptions héritent de `LembedError` :

```python
from libembedding import LembedError
from libembedding.exceptions import (
    InvalidArgumentError,
    OutOfMemoryError,
    OnnxRuntimeError,
    TokenizerError,
    DownloadError,
    IOError,
    ModelNotFoundError,
    UnsupportedError,
    BatchSizeError,
    LlamaError,
)
```

**Hiérarchie :**

```
LembedError (Exception)
├── InvalidArgumentError   — argument NULL ou hors plage
├── OutOfMemoryError       — échec d'allocation
├── OnnxRuntimeError       — erreur ONNX Runtime
├── TokenizerError         — erreur de chargement/encodage du tokenizer
├── DownloadError          — échec de téléchargement
├── IOError                — erreur fichier
├── ModelNotFoundError     — modèle inconnu
├── UnsupportedError       — fonctionnalité désactivée à la compilation
├── BatchSizeError         — taille de batch incompatible
└── LlamaError             — erreur du backend llama.cpp/GGUF
```

#### Exemple de gestion

```python
from libembedding import TextEmbedding
from libembedding.exceptions import ModelNotFoundError, OnnxRuntimeError

try:
    model = TextEmbedding("modele/inexistant")
except ModelNotFoundError as e:
    print(f"Modèle non trouvé : {e.detail}")
except OnnxRuntimeError as e:
    print(f"Erreur ONNX : {e.detail}")
except LembedError as e:
    print(f"Erreur libembedding [{e.status_code}] : {e.message}")
```

Chaque exception possède :
- `status_code` (`int`) — code d'erreur C
- `message` (`str`) — message générique (ex: `"ONNX Runtime error"`)
- `detail` (`str`) — détail technique (ex: message d'erreur ORT)

---

## Fonctions utilitaires

```python
import libembedding

# Lister les modèles par catégorie
libembedding.list_text_models()      # list[ModelInfo]
libembedding.list_sparse_models()    # list[ModelInfo]
libembedding.list_image_models()     # list[ModelInfo]
libembedding.list_reranker_models()  # list[ModelInfo]
```

### Auto-tuning workers (C API)

```c
#include <libembedding/worker_autotune.h>

lembed_worker_config_t cfg = lembed_detect_optimal_workers();
printf("Physical cores: %d\n", cfg.physical_cores);
printf("Optimal workers: %d\n", cfg.optimal_workers);
printf("Optimal threads: %d\n", cfg.optimal_threads);

int workers = lembed_recommended_workers_for_model("/path/to/model.gguf");
```

### Embedding modes (C API)

```c
#include <libembedding/embedding_mode.h>

lembed_embedding_mode_t mode = LEMBED_MODE_BALANCED;
lembed_text_model_t model = lembed_recommended_model_for_mode(mode);
const char* mode_str = lembed_mode_to_string(mode);  // "balanced"
```

### Embedding cache (C API)

```c
#include <libembedding/embedding_cache.h>

lembed_cache_config_t cfg = lembed_cache_config_default();
cfg.capacity = 4096;
cfg.ttl_seconds = 3600;  // 1 heure

lembed_cache_t* cache = lembed_cache_create(&cfg);
float* vec = NULL;
int dim = 0;
if (lembed_cache_get(cache, "Hello world", &vec, &dim)) {
    // cache hit
} else {
    // cache miss : calculer l'embedding puis stocker
    float embedding[384];
    // ... calcul ...
    lembed_cache_put(cache, "Hello world", embedding, 384);
}

lembed_cache_free(cache);
```
