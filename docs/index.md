---
title: Accueil
nav_order: 1
---

# Accueil

> **English:** [Home](en/index.html)

libembedding est une bibliothèque C/C++ et Python pour générer des embeddings denses, sparse et d'images à partir de modèles ONNX et GGUF (llama.cpp).

## Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| **Embeddings texte** | 44 modèles texte denses (quantifiés et FP32) |
| **Embeddings sparse** | SPLADE++ et BGE-M3 sparse |
| **Embeddings image** | CLIP, ResNet, Unicom, Nomic Vision |
| **Reranking** | BGE, Jina rerankers |
| **Auto-tuning** | Trouver la config optimale (workers, threads, batch) |
| **Cache LRU** | Cache thread-safe pour embeddings fréquents |
| **Multi-backend** | ONNX Runtime + llama.cpp |
| **Similarité** | Cosinus, produit scalaire, euclidienne |

## Navigation

- [Démarrage](getting_started.html) — Installation et premier usage
- [API Python](api_reference.html) — Référence API Python complète
- [Modèles](models.html) — Liste des modèles supportés
- [Performance](performance_tuning.html) — Optimisations et configurations
- [Usage avancé](advanced_usage.html) — Cache, modes, workers, autotune
- [Similarité](similarity.html) — Comparaison de vecteurs
- [Cache LRU](python/cache.html) — Cache d'embeddings
- [Benchmark](python/benchmark.html) — Comparaison backends
- [Backend](python/backend.html) — Auto-détection
- [Statistiques](python/stats.html) — Métriques runtime
- [C API](c_api/similarity.html) — Référence API C
- [Documentation anglaise](en/index.html)
