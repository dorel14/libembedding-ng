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

Mesuré par `benchmarks/quantization/bench_quantization.py` le 2026-10-01
(hôte `PC_Asus`, Windows 11 AMD64, provider CPU, Python 3.12.10,
libembedding 1.8.0, commit `c2dbe95`). Corpus de 1 000 textes (50 de warmup +
950 chronométrés), modèles pré-cachés. Chaque variante tourne dans son propre
sous-processus : le pic RSS est donc attribuable à la configuration qui l'a
produit. Tous les modes sont mesurés à toutes les tailles de batch, et chaque
variante quantifiée est comparée à la référence FP32 **à batch identique** —
comparer une variante batch 64 à une référence batch 8 mesurerait la taille de
batch, pas la quantification :

| Modèle | Variante | Poids | Débit | Pic RAM | vs FP32 (même batch) |
|--------|----------|-------|-------|---------|---------|
| `all-MiniLM-L6-v2` | FP32 (`none`) | 86,2 Mo | 90,5 docs/s @ batch 64 | 206 Mo | — |
| `all-MiniLM-L6-v2` | INT8 (`dynamic`) | **21,9 Mo** (3,9x plus petit) | **151,7 docs/s** @ batch 64 | **117 Mo** | **1,7x plus rapide**, 1,8x moins de RAM |
| `bge-small-en-v1.5` | FP32 (`none`) | 126,9 Mo | 47,4 docs/s @ batch 64 | 249 Mo | — |
| `bge-small-en-v1.5` | INT8 (`dynamic`) | **32,2 Mo** (3,9x plus petit) | **95,5 docs/s** @ batch 64 | **120 Mo** | **2,0x plus rapide**, 2,1x moins de RAM |
| `bge-small-en-v1.5` | **FP16** (`LEMBED_QUANTIZATION_FP16`) | 63,4 Mo (2,0x plus petit) | 5,0 docs/s @ batch 64 | 177 Mo | **9,5x plus lent**, 1,4x moins de RAM |

**L'INT8 dynamique est le gagnant sur les deux modèles**, et il gagne sur tous
les axes à la fois : 1,7x plus rapide sur MiniLM, 2,0x plus rapide sur BGE, 3,9x
plus petit sur disque, et 1,8 à 2,1x moins de RAM résidente. Il n'y a aucun
arbitrage ici, sauf la précision — que ce benchmark ne mesure pas.

> **Un autre piège, et c'est un problème de nommage.** Les quatre entrées
> `Qdrant/*-onnx-Q` du registre — dont `bge-small-en-v1.5` — livrent des poids
> **FP16**, pas INT8 : la totalité de leurs 149 initialiseurs est `FLOAT16` et
> elles portent un graphe optimisé par ORT (`Attention` +
> `SkipLayerNormalization` + `FastGelu` fusionnés). Elles déclaraient
> `LEMBED_QUANTIZATION_STATIC` avec une description « Quantized », donc
> `quantization="static"` renvoyait des poids float16 — **9,5x plus lent** que
> FP32, ce qu'un lecteur d'une version antérieure de ce tableau aurait conclu de
> « l'INT8 ». Le FP16 est le cas pathologique sur un CPU sans arithmétique FP16
> native : le fichier est divisé par deux, puis le calcul revient en FP32. Le
> registre déclare maintenant ces entrées `LEMBED_QUANTIZATION_FP16`, les
> descriptions disent FP16, et `quantization="static"` échoue proprement via
> `ModelNotFoundError` au lieu de renvoyer silencieusement du float16.
>
> L'export INT8 authentique de `bge-small-en-v1.5` est
> `onnx-community/bge-small-en-v1.5-ONNX` (`onnx/model_quantized.onnx`, 33 Mo,
> 144 initialiseurs `INT8` + 6 `UINT8`, 72 `MatMulInteger` alimentés par 48
> `DynamicQuantizeLinear`). Il est enregistré sous
> `LEMBED_TEXT_BGE_SMALL_EN_V15_INT8` et c'est lui que mesure la ligne INT8
> ci-dessus.
>
> Lancez `python benchmarks/verify_quantization_modes.py` pour vérifier, pour
> chaque entrée du registre, que le mode déclaré correspond aux dtypes
> réellement présents dans son fichier. Il remonte 0 incohérence ; avant le
> correctif `fp16` il en remontait 4.

- **La précision n'est pas mesurée ici.** Les embeddings INT8 dynamiques varient
  légèrement avec la composition du batch (cosinus ~0,984 face à la référence
  FP32). Validez sur votre propre corpus avant de basculer un index critique.
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
| 8 | INT8 (`dynamic`) | 98,0 | 10,21 | 32,2 Mo | 120 Mo | 2,0x plus rapide |
| 8 | FP16 | 5,0 | 198,99 | 63,4 Mo | 178 Mo | 9,6x plus lent |
| 32 | FP32 (`none`) | 50,5 | 19,82 | 126,9 Mo | 249 Mo | — |
| 32 | INT8 (`dynamic`) | 103,5 | 9,66 | 32,2 Mo | 120 Mo | 2,0x plus rapide |
| 32 | FP16 | 5,0 | 201,47 | 63,4 Mo | 177 Mo | 10,1x plus lent |
| 64 | FP32 (`none`) | 47,4 | 21,10 | 126,9 Mo | 249 Mo | — |
| 64 | INT8 (`dynamic`) | 95,5 | 10,47 | 32,2 Mo | 120 Mo | 2,0x plus rapide |
| 64 | FP16 | 5,0 | 199,18 | 63,4 Mo | 177 Mo | 9,5x plus lent |
| 128 | FP32 (`none`) | 47,8 | 20,91 | 126,9 Mo | 249 Mo | — |
| 128 | INT8 (`dynamic`) | 90,5 | 11,06 | 32,2 Mo | 120 Mo | 1,9x plus rapide |
| 128 | FP16 | 5,0 | 200,67 | 63,4 Mo | 178 Mo | 9,6x plus lent |

`sentence-transformers/all-MiniLM-L6-v2` :

| Batch | Variante | docs/s | ms/doc | Poids | Pic RAM | vs FP32 |
|-------|----------|--------|--------|-------|---------|---------|
| 8 | FP32 (`none`) | 97,1 | 10,30 | 86,2 Mo | 206 Mo | — |
| 8 | INT8 (`dynamic`) | 172,4 | 5,80 | 21,9 Mo | 118 Mo | 1,8x plus rapide |
| 32 | FP32 (`none`) | 96,3 | 10,38 | 86,2 Mo | 206 Mo | — |
| 32 | INT8 (`dynamic`) | 165,0 | 6,06 | 21,9 Mo | 118 Mo | 1,7x plus rapide |
| 64 | FP32 (`none`) | 90,5 | 11,05 | 86,2 Mo | 206 Mo | — |
| 64 | INT8 (`dynamic`) | 151,7 | 6,59 | 21,9 Mo | 117 Mo | 1,7x plus rapide |
| 128 | FP32 (`none`) | 99,6 | 10,04 | 86,2 Mo | 206 Mo | — |
| 128 | INT8 (`dynamic`) | 164,0 | 6,10 | 21,9 Mo | 118 Mo | 1,6x plus rapide |

#### Comment lire ces chiffres

- **Le pic RAM est mesuré, pas écarté.** Chaque configuration a tourné dans son
  propre processus, donc la marque haute du jeu résident lui appartient. Il
  dépasse le fichier de poids parce qu'ONNX Runtime mappe le fichier en mémoire
  et alloue sa propre arène par-dessus — 118 Mo pour un fichier de 21,9 Mo,
  249 Mo pour un fichier de 126,9 Mo — mais *l'écart* entre configurations est
  réel.
- **La variabilité entre deux exécutions est de l'ordre de 10 %.** La référence
  MiniLM FP32 batch 64 a été mesurée à 82,7, 89,0, 90,5 puis 99,6 docs/s sur la
  même machine au cours de quatre exécutions. Traitez les faibles écarts comme du
  bruit et les écarts d'ordre de grandeur (1,7x, 2,0x, 9,5x) comme réels ; les
  tableaux affichent deux décimales parce que l'exécution brute le fait, pas
  parce que le troisième chiffre est significatif.
- **Le chemin quantifié est stable sur les batchs** (151,7 à 172,4 docs/s sur
  MiniLM, 90,5 à 103,5 sur BGE) : la taille de batch n'est donc pas ce qui
  pilote la comparaison. Une exécution antérieure montrait un effondrement du
  mode `dynamic` de MiniLM à 112,1 docs/s au batch 128 ; cela ne s'est pas
  reproduit, et c'est le type de valeur aberrante que la variabilité de ~10 %
  explique.
- **Tous les modes n'existent pas pour tous les modèles.** Le registre propose
  exactement `dynamic` (vrai INT8) pour MiniLM, et `none` + `dynamic` + `fp16`
  pour `bge-small-en-v1.5`. Aucun modèle n'a d'entrée `static` (INT8 avec
  calibration). Les combinaisons absentes sont déclarées *non mesurées* avec les
  modes disponibles, jamais comme un échec.
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
  TextEmbedding("BAAI/bge-small-en-v1.5", quantization="dynamic")   # vrai INT8
  TextEmbedding("BAAI/bge-small-en-v1.5", quantization="fp16")      # float16

  # Nommer directement le dépôt est équivalent et sans ambiguïté —
  # c'est ce que fait le benchmark
  TextEmbedding("Xenova/all-MiniLM-L6-v2", quantization="dynamic")
  TextEmbedding("onnx-community/bge-small-en-v1.5-ONNX", quantization="dynamic")
  ```

  Demander un mode qu'un modèle n'a pas est une erreur, pas un repli silencieux :

  ```python
  TextEmbedding("BAAI/bge-small-en-v1.5", quantization="static")
  # ModelNotFoundError: Model 'BAAI/bge-small-en-v1.5' has no 'static'
  # variant (available: dynamic, fp16, none)
  ```

  En C, l'enum de modèle sélectionne l'entrée, ou le champ `quantization` des
  options v2 :

  ```c
  opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15_INT8;  /* ou ..._Q pour le FP16 */
  ```

  Cette résolution n'a pas toujours existé : le mode était simplement marqué sur
  le contexte sans changer le fichier chargé. C'est pourquoi le benchmark archivé
  du 2026-09-27 est sans valeur — il mesurait les mêmes poids FP32 sous trois
  étiquettes, et l'exécution du 2026-09-29 publiée avant le correctif est
  également sans valeur. L'exécution du 2026-10-01 ci-dessus est la première qui
  résout les entrées de registre *et* mesure un vrai fichier INT8.

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
`docs/archive/benchmarks/quantization-2026-10-01/` : chaque chiffre publié
ci-dessus s'y retrouve ligne à ligne.
`benchmarks/quantization/results.html` n'est que le chemin de sortie du
benchmark, écrasé à chaque exécution. Les exécutions antérieures sont archivées
dans `quantization-2026-09-27/`, `quantization-2026-09-29/` et
`quantization-2026-09-30/` ; toutes trois sont sans valeur — la première
mesurait des poids FP32 sous trois étiquettes, la deuxième précède le correctif
de sélection de variante, la troisième mesurait des fichiers FP16 en les nommant
INT8. Une exécution incomplète est archivée dans
`quantization-2026-09-30-partial/`.

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
