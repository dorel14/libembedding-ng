---
title: Démarrage
nav_order: 2
---

# Démarrage rapide

## Installation

```bash
pip install libembedding-ng
```

> Le paquet PyPI est **`libembedding-ng`** (importé en Python sous le nom `libembedding`).

Des wheels existent pour Windows x64, Linux x86_64, Linux aarch64 (Raspberry Pi 64 bits compris), macOS arm64 et Android (CPython 3.13, via l'index Chaquopy) ; iOS est livré comme bibliothèque native (xcframework). Voir [Plateformes](platforms.html) pour la matrice complète, l'installation sur Android, le build iOS et la limite de taille des modèles sur mobile.

## Prérequis

### Python (pip)

Le paquet PyPI inclut **ONNX Runtime** automatiquement. Aucune installation système supplémentaire n'est nécessaire.

### C/C++ (compilation)

libembedding nécessite **ONNX Runtime** et **libcurl** pour la compilation :

```bash
# macOS
brew install onnxruntime curl

# Ubuntu / Debian
sudo apt install libonnxruntime-dev libcurl4-openssl-dev

# Windows
# Les fichiers sont inclus dans le dépôt (third_party/onnxruntime/)
# libcurl est fourni via vcpkg ou le système
```

> **Note :** Si ONNX Runtime n'est pas trouvé automatiquement, définissez la variable d'environnement `ONNXRUNTIME_ROOT` pointant vers votre installation.

Au moment de l'exécution, les bibliothèques partagées d'ONNX Runtime et de libcurl sont copiées automatiquement à côté des exécutables sur **toutes les plateformes** (`.dll` sur Windows, `.dylib` sur macOS, `.so` sur Linux).

### Backend llama.cpp

Le backend llama.cpp est activé par défaut et permet de charger les modèles GGUF :

```bash
cmake .. -DLIBEMBEDDING_BUILD_SHARED=ON
cmake --build . --parallel
```

Vérifiez la disponibilité à l'exécution :

```python
from libembedding import TextEmbedding

# Auto-détection par extension de fichier
model = TextEmbedding("Xenova/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2-Q4_K_M.gguf")

# Ou depuis un chemin local
model = TextEmbedding("/path/to/model.gguf")
```

## Vérification de l'installation

```python
import libembedding
print(libembedding.__version__)

# Lister les modèles disponibles
for m in libembedding.list_text_models()[:3]:
    print(m.model_name, m.dim)
```

## Première utilisation

### Embeddings de texte (denses)

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed([
    "The cat sat on the mat",
    "A kitten was sitting on a rug",
    "Quantum physics describes subatomic particles",
])

print(embeddings.shape)   # (3, 384)
print(embeddings.dtype)   # float32

# Similarité cosinus
similarity = np.dot(embeddings[0], embeddings[1])  # ≈ 0.82
```

### Modes prédéfinis FAST / BALANCED / QUALITY

```python
from libembedding import TextEmbedding

# FAST : MiniLM-L12-v2 quantifié (vitesse maximale)
model = TextEmbedding.from_mode("fast")

# BALANCED : BGE-small (compromis vitesse/qualité)
model = TextEmbedding.from_mode("balanced")

# QUALITY : BGE-base (meilleure qualité)
model = TextEmbedding.from_mode("quality")
```

### llama.cpp avec auto-tuning

```python
from libembedding import TextEmbedding

# Auto-détection du nombre optimal de sessions/workers
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,      # détecte automatiquement le nombre de sessions
    cache_size=4096,        # cache LRU optionnel
)

embeddings = model.embed(["Hello world", "How are you?"])
```

### Sparse embeddings

```python
from libembedding import SparseTextEmbedding

sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning algorithms"])

for r in results:
    print(r.indices.shape, r.values.shape)
    # (N,) (N,) où N est le nombre de tokens actifs
```

### Embeddings d'images

```python
from libembedding import ImageEmbedding

img_model = ImageEmbedding()
embeddings = img_model.embed_files(["photo.jpg", "diagram.png"])
print(embeddings.shape)  # (2, 512)
```

### Reranking

```python
from libembedding import Reranker

reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank(
    "What is deep learning?",
    [
        "Deep learning uses neural networks with many layers",
        "The weather is sunny today",
        "Neural networks are inspired by biological brains",
    ],
)

for r in ranked:
    print(f"doc[{r.index}] score={r.score:.4f}")
```

## Utilisation avec un context manager

Toutes les classes supportent le protocole context manager pour une gestion automatique des ressources :

```python
with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
    embeddings = model.embed(["Hello world"])
# Les ressources C sont libérées automatiquement
```

## Exemple complet

Voir [examples/python/example.py](https://github.com/dorel14/libembedding/blob/main/examples/python/example.py) pour un exemple détaillé avec recherche sémantique.
