---
title: Performance
nav_order: 5
---

# Performance Tuning

Ce guide couvre les fonctionnalités avancées de performance de libembedding : pool de sessions, auto-tuning, sélection automatique de modèle, bucketing, cache LRU, et modes FAST/BALANCED/QUALITY.

## Sommaire

1. [Pool de sessions (EmbeddingPool)](#pool-de-sessions-embeddingpool)
2. [Auto-Tuning](#auto-tuning)
3. [Sélection automatique de modèle](#sélection-automatique-de-modèle)
4. [Performance llama.cpp / GGUF](#performance-llamacpp--gguf)
5. [Benchmarks](#benchmarks)
6. [Bonnes pratiques](#bonnes-pratiques)

---

## Pool de sessions (EmbeddingPool)

Pour les petites architectures Transformer (MiniLM, BGE-small, E5-small), le parallélisme **inter-sessions** (plusieurs sessions ONNX indépendantes) est plus efficace que le parallélisme **intra-session** (threads ONNX).

### Quand l'utiliser

| Scénario | Recommandation |
|----------|----------------|
| < 100 embeddings | `TextEmbedding` simple suffit |
| > 100 embeddings | `TextEmbeddingPool` recommandé |
| Production / haut débit | `TextEmbeddingPool` + `autotune=True` |

### Utilisation

```python
from libembedding import TextEmbeddingPool

# Pool avec 8 workers (sessions ONNX indépendantes)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    workers=8,               # nombre de sessions parallèles
    threads_per_worker=1,    # 1 thread par session (évite la contention)
    batch_size=64,
    offline=True,
)

embeddings = pool.embed(texts)
pool.close()
```

### Gain de performance

| Configuration | Docs/s (textes courts) | Speedup |
|---------------|------------------------|---------|
| 1 session × 4 threads | ~100 | 1.0x |
| 4 workers × 1 thread | ~265 | 2.6x |
| **8 workers × 1 thread** | **~360** | **3.6x** |

> **Règle d'or** : `workers × threads ≤ nombre de cœurs CPU`

---

## Auto-Tuning

L'auto-tuning trouve automatiquement la configuration optimale (workers, threads, batch_size) pour votre machine et votre corpus.

### Utilisation simple

```python
from libembedding import TextEmbeddingPool

# Autotune avec corpus synthétique (défaut)
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,           # active l'auto-tuning
    offline=True,
)
# → benchmark ~5-15s la première fois, puis cache instantané
```

### Autotune avec votre corpus (recommandé)

Pour des résultats plus précis, fournissez un échantillon de vos vrais textes :

```python
# Utilisez un échantillon représentatif de vos données
sample_texts = votre_csv["text_column"].head(1000).tolist()

pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,    # votre corpus
    autotune_max_samples=100,        # échantillonne 100 textes représentatifs
    offline=True,
)
```

### Cache d'autotuning

Les résultats sont mis en cache par machine + modèle :

```
    %LOCALAPPDATA%\libembedding\autotune\8x4_Intel_i7-1065G7_model_ort1.29_v1.4.0.json
```

| Événement | Comportement |
|-----------|--------------|
| Premier appel | Benchmark + sauvegarde cache |
| Même machine + modèle | Cache hit (< 1ms) |
| Changement de CPU/ORT/modèle | Cache miss → re-benchmark |
| Même modèle, corpus différent | Cache miss (l'empreinte du corpus fait partie de la clef) |
| Même corpus, objectif ou mode différent | Cache miss |

Une entrée de cache est réutilisée uniquement si **toutes** ses dimensions
d'identité correspondent : modèle, variante (`synthetic` / `custom`), empreinte
du corpus, objectif et mode. Un fichier de cache corrompu ou tronqué est traité
comme un miss, jamais comme un résultat. Les écritures sont atomiques
(fichier temporaire puis renommage), donc une interruption ne laisse pas de
résultat partiel.

Mesurer sur son propre corpus :

```python
from libembedding import autotune

result = autotune(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    texts=my_documents,   # échantillonné à max_sample_size (100 par défaut)
)
```

```python
from libembedding import clear_autotune_cache

# Effacer le cache d'un modèle (toutes ses variantes : synthetic et custom)
clear_autotune_cache("Qdrant/all-MiniLM-L6-v2-onnx")

# Effacer tout le cache
clear_autotune_cache()
```

### API complète

```python
# TextEmbedding avec autotune
model = TextEmbedding(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)

# TextEmbeddingPool avec autotune
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=sample_texts,
    autotune_max_samples=100,
    offline=True,
)
```

### Auto-tune unifié (API unifiée)

L'auto-tune unifié fournit un point d'entrée unique pour tous les types de tâches :

```python
from libembedding import (
    autotune_unified,
    LEMBED_TASK_RERANKING,
    LEMBED_AUTOTUNE_QUICK,
)

result = autotune_unified(
    task=LEMBED_TASK_RERANKING,
    model_name="BAAI/bge-reranker-base",
    mode=LEMBED_AUTOTUNE_QUICK,
)
# result: UnifiedTuningResult
```

### Benchmark llama.cpp / GGUF

Avec le backend llama.cpp, comparez ONNX et GGUF :

```python
from libembedding import Benchmark, CorpusType, Objective

bench = Benchmark()
comparison = bench.compare_all(
    onnx_path="/path/to/model.onnx",
    gguf_path="/path/to/model.Q4_K_M.gguf",
    corpus=CorpusType.MIXED,
    objective=Objective.BALANCED,
)
print(comparison.recommendation.backend)
```

---

## Sélection automatique de modèle

Pour choisir automatiquement le meilleur modèle selon votre hardware et votre cas d'usage :

```python
from libembedding import auto_select_model, TextEmbeddingPool

# Sélection automatique
result = auto_select_model("balanced")  # "speed", "quality", ou "balanced"

print(f"Modèle: {result.model_name}")
print(f"Dimension: {result.dim}")
print(f"Config: {result.workers} workers × {result.threads} threads")
print(f"Throughput: {result.throughput_docs:.0f} docs/s")

# Utilisation directe
pool = TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
)
```

### Cas d'usage

| Use case | Recommandation |
|----------|----------------|
| Temps réel, latence critique | `"speed"` |
| Recherche sémantique, qualité maximale | `"quality"` |
| Production généraliste | `"balanced"` (défaut) |

---

## Performance llama.cpp / GGUF

Pour le backend llama.cpp sur CPU, la configuration optimale diffère de ONNX :

### Recommandations

| Paramètre | Valeur recommandée | Raison |
|-----------|---------------------|--------|
| `threads` | **1** par session | Évite la contention sur les petits modèles BERT |
| `workers` / `sessions` | **physical_cores × 2** (max 8) | Scaling quasi-linéaire jusqu'à saturation |
| `batch_size` | 8-32 | Selon la longueur moyenne des textes |
| `batch_strategy` | `LENGTH_BUCKET` | Réduit le padding pour des corpus hétérogènes |

### Auto-tuning workers

```python
from libembedding import TextEmbedding

# Détection automatique du nombre optimal de sessions
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,
    cache_size=4096,
)
```

### Modes prédéfinis

```python
# FAST : priorité vitesse
model = TextEmbedding.from_mode("fast")

# BALANCED : compromis vitesse/qualité (défaut)
model = TextEmbedding.from_mode("balanced")

# QUALITY : priorité qualité
model = TextEmbedding.from_mode("quality")
```

### Cache LRU

```python
# Active un cache LRU de 4096 entrées
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    cache_size=4096,
)
```

### Baseline i7-1065G7 (MiniLM-L6-v2 Q4_K_M)

| Sessions | Threads | Docs/s |
|----------|---------|--------|
| 1 | 1 | 41.8 |
| 1 | 4 | 72.1 |
| 4 | 1 | 105.9 |
| **6** | **1** | **125.9** |
| 8 | 1 | 129.4 |
| 6 | 2 | 87.6 |

**Conclusion** : au-delà de 2 sessions, `threads=1` est systématiquement plus rapide. Le multithreading intra-session dégrade les performances sur les embeddings BERT.

---

## Benchmarks

### Configuration testée

- CPU : Intel i7-1065G7 (4c/8t)
- OS : Windows 11
- Modèle : all-MiniLM-L6-v2 (384-dim)
- Backend : ONNX Runtime et llama.cpp (GGUF Q4_K_M)

### Impact de la longueur des textes (ONNX)

| Tokens/texte | Docs/s (8 workers) | ms/texte |
|--------------|---------------------|----------|
| 16 | 696 | 6.0 |
| 64 | 150 | 16.9 |
| 128 | 62 | 32.6 |
| 256 | 15 | 68.6 |

> **Note** : Le throughput est fortement dépendant de la longueur des textes. Les benchmarks avec textes courts ne prédisent pas les performances avec textes longs.

### Comparaison ONNX vs llama.cpp

| Backend | Config | Docs/s | RAM |
|---------|--------|--------|-----|
| ONNX | 8 workers × 1 thread | ~128 | ~500 MB |
| llama.cpp GGUF Q4_K_M | 6 sessions × 1 thread | ~126 | ~20 MB |

### Comparaison des configurations llama.cpp

| Sessions | Threads | Docs/s | Efficacité |
|----------|---------|--------|------------|
| 1 | 1 | 41.8 | Basse |
| 1 | 4 | 72.1 | Moyenne |
| 4 | 1 | 105.9 | Bonne |
| **6** | **1** | **125.9** | **Optimale** |
| 8 | 1 | 129.4 | Saturation |
| 6 | 2 | 87.6 | Dégradée |

**Conclusion** : Le parallélisme inter-sessions (6×1) est optimal sur cette machine. Au-delà de 2 sessions, le multithreading intra-session dégrade les performances.

### Comparaison des modèles

| Modèle | Docs/s (8w×1t) | RAM (8 workers) | Déterministe |
|--------|----------------|-----------------|--------------|
| MiniLM-L6-v2-Q (INT8) | 474-696 | 230 MB | ~1.6% variance |
| MiniLM-L6-v2 (FP32) | 305-361 | 740 MB | Oui |
| BGE-small-en (FP32) | 143-150 | 1.1 GB | Oui |

### Quantification : FP32 vs INT8

Mesuré par `benchmarks/quantization/bench_quantization.py` le 2026-09-30
(hôte `PC_Asus`, Windows 11 AMD64, provider CPU, Python 3.12.10,
libembedding 1.8.0, commit `6ef3b99`). Corpus de 1 000 textes (50 de warmup +
950 chronométrés), modèles pré-cachés. Chaque variante tourne dans son propre
sous-processus : le pic RSS est donc attribuable à la configuration qui l'a
produit. Tous les modes sont mesurés à toutes les tailles de batch, et chaque
variante quantifiée est comparée à la référence FP32 **à batch identique** —
comparer une variante batch 64 à une référence batch 8 mesurerait la taille de
batch, pas la quantification :

| Modèle | Variante | Poids | Débit | Pic RAM | vs FP32 (même batch) |
|--------|----------|-------|-------|---------|---------|
| `all-MiniLM-L6-v2` | FP32 (`none`) | 86,2 Mo | 89,0 docs/s @ batch 64 | 206 Mo | — |
| `all-MiniLM-L6-v2` | INT8 (`dynamic`) | **21,9 Mo** (3,9x plus petit) | **162,2 docs/s** @ batch 64 | **118 Mo** | **1,8x plus rapide**, 1,7x moins de RAM |
| `bge-small-en-v1.5` | FP32 (`none`) | 126,9 Mo | 48,6 docs/s @ batch 32 | 249 Mo | — |
| `bge-small-en-v1.5` | FP16 (`static` dans le registre) | **63,4 Mo** (2,0x plus petit) | 5,1 docs/s @ batch 32 | 176 Mo | **9,5x plus lent**, 1,4x moins de RAM |

> **Lisez attentivement la colonne « variante » : les deux lignes n'utilisent pas
> la même technologie de quantification.** `Xenova/all-MiniLM-L6-v2` est bien
> INT8 (ses initialiseurs sont `INT8`/`UINT8`). Les quatre entrées
> `Qdrant/*-onnx-Q` — dont `bge-small-en-v1.5` — sont en **FP16**, pas en INT8 :
> la totalité de leurs 149 initialiseurs est `FLOAT16`, et elles livrent un graphe
> optimisé par ORT (`Attention` + `SkipLayerNormalization` + `FastGelu` fusionnés).
> Le registre les étiquette `LEMBED_QUANTIZATION_STATIC` et leur description dit
> « Quantized », ce qui explique que cette ligne ne dise pas INT8.
>
> Le registre ne contient **aucune variante INT8 de `bge-small-en-v1.5`** : ce
> benchmark ne mesure donc pas du tout l'INT8 statique sur ce modèle. Il mesure
> FP16 contre FP32 — et le FP16 est le cas pathologique sur un CPU sans calcul
> FP16 natif : le fichier et le jeu résident sont divisés par deux, puis le
> travail se fait en FP32, d'où une lenteur au lieu d'un gain.
>
> Les deux modèles ne se comportent pas de la même façon, et le résumé honnête
> n'est pas une formule unique :

- **L'INT8 dynamique de MiniLM est plus rapide *et* plus petit.** 162,2 contre
  89,0 docs/s (1,8x), 118 contre 206 Mo de pic (1,7x), 21,9 contre 86,2 Mo sur
  disque (3,9x). C'est le seul chemin quantifié réellement optimisé de cet
  ensemble, et il gagne sur tous les axes sur cette machine.
- **L'entrée BGE `_Q` est en FP16 et 9,5x plus lente.** 5,1 contre 48,6 docs/s,
  alors qu'elle économise de la RAM (176 contre 249 Mo, 1,4x) et du disque
  (63,4 contre 126,9 Mo, 2,0x). Ne la choisissez pas pour le débit, seulement si
  le téléchargement ou le disque est la contrainte. Ne lisez pas ce chiffre comme
  « l'INT8 est lent » : le même registre sert du vrai INT8 pour MiniLM, et celui-là
  est 1,8x plus rapide.
- **Le pic RAM est une vraie mesure.** Chaque configuration a tourné seule, donc
  la marque haute du jeu résident lui appartient. Il dépasse le fichier de poids
  parce qu'ONNX Runtime mappe le fichier en mémoire et alloue sa propre arène
  par-dessus — mais *l'écart* entre configurations est réel, et il varie dans le
  même sens que le fichier de poids pour les deux modèles.
- **Le gain fiable reste le disque, le cache et le téléchargement** — la vraie
  contrainte en image de conteneur, en installation hors-ligne et en démarrage
  à froid.

#### Résultats détaillés par batch

`BAAI/bge-small-en-v1.5` :

| Batch | Variante | docs/s | ms/doc | Poids | Pic RAM | vs FP32 |
|-------|----------|--------|--------|-------|---------|---------|
| 8 | FP32 (`none`) | 48,2 | 20,75 | 126,9 Mo | 249 Mo | — |
| 8 | FP16 (`static` dans le registre) | 5,0 | 199,69 | 63,4 Mo | 176 Mo | 9,6x plus lent |
| 32 | FP32 (`none`) | 48,6 | 20,57 | 126,9 Mo | 249 Mo | — |
| 32 | FP16 (`static` dans le registre) | 5,1 | 197,42 | 63,4 Mo | 177 Mo | 9,5x plus lent |
| 64 | FP32 (`none`) | 48,1 | 20,79 | 126,9 Mo | 249 Mo | — |
| 64 | FP16 (`static` dans le registre) | 5,0 | 199,88 | 63,4 Mo | 177 Mo | 9,6x plus lent |
| 128 | FP32 (`none`) | 50,5 | 19,79 | 126,9 Mo | 249 Mo | — |
| 128 | FP16 (`static` dans le registre) | 5,0 | 200,49 | 63,4 Mo | 177 Mo | 10,1x plus lent |

`sentence-transformers/all-MiniLM-L6-v2` :

| Batch | Variante | docs/s | ms/doc | Poids | Pic RAM | vs FP32 |
|-------|----------|--------|--------|-------|---------|---------|
| 8 | FP32 (`none`) | 90,5 | 11,05 | 86,2 Mo | 206 Mo | — |
| 8 | INT8 (`dynamic`) | 161,0 | 6,21 | 21,9 Mo | 118 Mo | 1,8x plus rapide |
| 32 | FP32 (`none`) | 95,6 | 10,46 | 86,2 Mo | 206 Mo | — |
| 32 | INT8 (`dynamic`) | 157,2 | 6,36 | 21,9 Mo | 118 Mo | 1,6x plus rapide |
| 64 | FP32 (`none`) | 89,0 | 11,24 | 86,2 Mo | 206 Mo | — |
| 64 | INT8 (`dynamic`) | 162,2 | 6,16 | 21,9 Mo | 118 Mo | 1,8x plus rapide |
| 128 | FP32 (`none`) | 94,9 | 10,54 | 86,2 Mo | 206 Mo | — |
| 128 | INT8 (`dynamic`) | 112,1 | 8,92 | 21,9 Mo | 118 Mo | 1,2x plus rapide |

#### Comment lire ces chiffres

- **Le pic RAM est mesuré, pas écarté.** Chaque configuration a tourné dans son
  propre processus, donc la marque haute du jeu résident lui appartient. Il
  dépasse le fichier de poids parce qu'ONNX Runtime mappe le fichier en mémoire
  et alloue sa propre arène par-dessus — 118 Mo pour un fichier de 21,9 Mo,
  249 Mo pour un fichier de 126,9 Mo — mais *l'écart* entre configurations est
  réel.
- **La variabilité entre deux exécutions est de l'ordre de 10 %.** Une
  réexécution de la même configuration a mesuré la référence MiniLM FP32 batch 64
  à 89,0 puis 99,5 docs/s sur la même machine. Traitez les faibles écarts comme
  du bruit et les écarts d'ordre de grandeur (1,8x, 9,5x) comme réels ; les
  tableaux affichent deux décimales parce que l'exécution brute le fait, pas
  parce que le troisième chiffre est significatif.
- **Le mode `dynamic` s'effondre au batch 128** (112,1 docs/s, contre 157-162
  aux batchs 8 à 64). Le chemin quantifié n'est pas indifférent à la taille de
  batch : le meilleur batch d'une variante quantifiée n'est donc pas forcément le
  meilleur batch de sa référence FP32.
- **Tous les modes n'existent pas pour tous les modèles.** MiniLM n'a pas de
  variante `static`, `bge-small-en-v1.5` pas de variante `dynamic`. Ces
  combinaisons sont déclarées *non mesurées* avec les modes disponibles, jamais
  comme un échec.
- **`quantization=` sélectionne les poids, pas seulement les options de session.**
  Chaque variante quantifiée est sa propre entrée de registre avec son propre
  `model_file`, et le mode demandé est résolu *avant* la création de la session.
  Si le modèle n'a pas d'entrée dans ce mode, la création échoue avec
  `LEMBED_ERROR_MODEL_NOT_FOUND` (Python : `ModelNotFoundError`) en indiquant les
  modes réellement disponibles :

  ```python
  from libembedding import TextEmbedding

  # Résolu vers l'entrée de registre qui fournit le mode demandé
  TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", quantization="dynamic")
  TextEmbedding("BAAI/bge-small-en-v1.5", quantization="static")

  # Nommer directement le dépôt quantifié est équivalent et sans ambiguïté —
  # c'est ce que fait le benchmark
  TextEmbedding("Xenova/all-MiniLM-L6-v2", quantization="dynamic")
  TextEmbedding("Qdrant/bge-small-en-v1.5-onnx-Q", quantization="static")
  ```

  En C, l'enum `_Q` ou le champ `quantization` des options v2 sélectionne
  l'entrée :

  ```c
  opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15_Q;   /* ou LEMBED_TEXT_ALL_MINILM_L6_V2_Q */
  ```

  Cette résolution n'a pas toujours existé : le mode était simplement marqué sur
  le contexte sans changer le fichier chargé. C'est pourquoi le benchmark archivé
  du 2026-09-27 est sans valeur — il mesurait les mêmes poids FP32 sous trois
  étiquettes, et l'exécution du 2026-09-29 publiée avant le correctif est
  également sans valeur. L'exécution du 2026-09-30 ci-dessus est la première dont
  les chiffres reflètent la sélection d'entrée de registre.

  Deux réserves : `quantization="auto"` ne sélectionne **aucun** poids (il charge
  l'entrée FP32 par défaut alors que `.quantization` rapporte `"auto"`) —
  utilisez `preferred_quantization="auto"` pour une auto-sélection mesurée. Et
  `Reranker(quantization=...)` ne résout pas non plus une variante : passez le nom
  de l'entrée quantifiée, `jinaai/jina-reranker-v1-turbo-en-quantized`.

- **La qualité n'est pas mesurée ici.** Les embeddings INT8 dynamiques varient
  légèrement selon la composition du batch (cos ≈ 0,984 face à la référence
  FP32). À valider sur votre propre corpus avant de basculer un index critique ;
  l'INT8 statique (`_Q`) est le choix par défaut le plus sûr.

Rapport complet, bloc d'environnement inclus, archivé verbatim dans
`docs/archive/benchmarks/quantization-2026-09-30/` : chaque chiffre publié
ci-dessus s'y retrouve ligne à ligne.
`benchmarks/quantization/results.html` n'est que le chemin de sortie du
benchmark, écrasé à chaque exécution. Les exécutions antérieures sont archivées
dans `quantization-2026-09-27/` et `quantization-2026-09-29/` ; toutes deux sont
sans valeur car antérieures au correctif de sélection de variante. Une
exécution incomplète est archivée dans `quantization-2026-09-30-partial/`.

```bash
# Chaque variante tourne dans son propre sous-processus : le pic mémoire est
# attribuable et une configuration qui crash n'entraîne pas le benchmark.
python benchmarks/quantization/bench_quantization.py --num-texts 1000
```

---

## Bonnes pratiques

### 1. Pour les gros corpus (> 100K textes)

```python
# Échantillonnage stratifié automatique
pool = TextEmbeddingPool(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    autotune=True,
    autotune_texts=large_corpus,      # vos 2M de textes
    autotune_max_samples=100,         # échantillonne 100 textes représentatifs
    offline=True,
)
# → ~56s pour autotuner (une fois)
# → 66 docs/s en production
# → ~8h pour 2M de textes
```

### 2. Pour la recherche sémantique

```python
# Privilégiez la qualité
pool = TextEmbeddingPool(
    "Xenova/bge-small-en-v1.5",      # meilleure qualité que MiniLM
    autotune=True,
    autotune_texts=documents,         # vos documents
    offline=True,
)
```

### 3. Pour le temps réel

```python
# Privilégiez la vitesse
pool = TextEmbeddingPool(
    "Xenova/all-MiniLM-L6-v2",        # version quantifiée INT8
    autotune=True,
    autotune_texts=queries,           # vos requêtes courtes
    offline=True,
)
```

### 4. Stratégie de batch

| Taille de batch | Usage |
|-----------------|-------|
| 8-16 | Textes longs (> 100 tokens) |
| 32-64 | Usage général |
| 128-256 | Textes courts (< 20 tokens) |

### 5. Bonnes pratiques générales

- **Réutilisez le pool** : créez-le une fois, réutilisez-le pour tous les embeddings
- **Autotune une seule fois** : le cache évite de re-benchmarker
- **Utilisez vos propres textes** pour l'autotune (plus précis que le corpus synthétique)
- **Fermez le pool** : `pool.close()` ou context manager `with`
- **Mode offline** en production : `offline=True` évite les téléchargements

### Exemple complet : Pipeline RAG

```python
from libembedding import TextEmbeddingPool, auto_select_model
import numpy as np

# 1. Sélection automatique du modèle (une fois)
result = auto_select_model("balanced")
print(f"Modèle sélectionné: {result.model_name}")

# 2. Création du pool avec config optimale
with TextEmbeddingPool(
    result.model_code,
    workers=result.workers,
    threads_per_worker=result.threads,
    batch_size=result.batch_size,
    offline=True,
) as pool:

    # 3. Embedding des documents
    doc_embeddings = pool.embed(documents)

    # 4. Embedding des requêtes
    query_embeddings = pool.embed(queries)

    # 5. Recherche du plus proche voisin
    scores = doc_embeddings @ query_embeddings.T
    top_k = np.argsort(scores, axis=0)[-5:]
```
