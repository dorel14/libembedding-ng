---
title: Accueil
nav_order: 1
---

# Documentation libembedding

Bienvenue dans la documentation de **libembedding**, une bibliothèque d'embeddings rapide avec APIs **C/C++** et **Python** basée sur ONNX Runtime et llama.cpp.

> **Projet dérivé de [pacifio/libembedding](https://github.com/pacifio/libembedding).**
> Cette fourche ajoute le support Windows (DLL native), l'empaquetage PyPI sous le nom `libembedding-ng`, le chargement de modèles locaux, l'introspection à l'exécution, les fonctions de similarité, le streaming, le pool multi-workers, l'auto-tuning et la sélection automatique de modèle, le support du backend llama.cpp/GGUF, le bucketing, le cache LRU, le scheduler dynamique et les modes FAST/BALANCED/QUALITY.

## Structure de la documentation

| Section | Fichier | Description |
|---------|---------|-------------|
| **Démarrage** | [getting_started.html](getting_started.html) | Installation, prérequis et premiers pas |
| **API Python** | [api_reference.html](api_reference.html) | Référence complète des classes Python |
| **Modèles** | [models.html](models.html) | Catalogue des modèles disponibles (texte, image, sparse, reranker) |
| **Performance** | [performance_tuning.html](performance_tuning.html) | Workers, threads, bucketing, cache, modes, baseline |
| **Usage avancé** | [advanced_usage.html](advanced_usage.html) | Scheduler, cache LRU, modes FAST/BALANCED/QUALITY, modèles GGUF/llama.cpp |
| **Exceptions** | [api_reference.html#gestion-des-erreurs](api_reference.html#gestion-des-erreurs) | Hiérarchie des exceptions Python |
| **English** | [en/index.html](en/index.html) | English documentation |

## Vue d'ensemble

```python
from libembedding import TextEmbedding, SparseTextEmbedding, Reranker
import numpy as np

# Embeddings denses
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello world", "How are you?"])
print(embeddings.shape)  # (2, 384)

# Mode prédéfini (FAST / BALANCED / QUALITY)
model = TextEmbedding.from_mode("balanced")

# Sparse embeddings
sparse = SparseTextEmbedding()
results = sparse.embed(["machine learning"])
print(results[0].indices.shape, results[0].values.shape)

# Reranking
reranker = Reranker("BAAI/bge-reranker-base")
ranked = reranker.rerank("What is deep learning?", [
    "Deep learning uses neural networks",
    "The weather is sunny today",
])
print(ranked[0].score, ranked[0].index)
```

**Performance** : 5-8x plus rapide que fastembed, 3.5x moins de mémoire.
