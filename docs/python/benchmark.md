---
title: Benchmark unifié
nav_order: 6
---

# Benchmark unifié

Le module `Benchmark` fournit une comparaison unifiée des backends ONNX et llama.cpp sur le même corpus, avec auto-tuning. Il permet de détecter le matériel, exécuter des benchmarks et comparer les résultats.

## Détection matérielle

```python
from libembedding import Benchmark

bench = Benchmark()
hw = bench.hardware
print(f"CPU: {hw.cpu_name}, Cores: {hw.logical_cores}, RAM: {hw.ram_mb} MB")
```

| Attribut | Type | Description |
|----------|------|-------------|
| `cpu_name` | `str` | Nom du processeur |
| `physical_cores` | `int` | Nombre de cœurs physiques |
| `logical_cores` | `int` | Nombre de cœurs logiques |
| `os_name` | `str` | Nom du système d'exploitation |
| `ram_mb` | `int` | Mémoire RAM en Mo |
| `features` | `str` | Fonctionnalités CPU détectées |

## Catégories de corpus

| Constante | Description |
|-----------|-------------|
| `CorpusType.SHORT` | Textes courts (< 20 tokens) |
| `CorpusType.MEDIUM` | Textes moyens (20-80 tokens) |
| `CorpusType.LONG` | Textes longs (80-200 tokens) |
| `CorpusType.VERY_LONG` | Textes très longs (200+ tokens) |
| `CorpusType.MIXED` | Longueurs mixtes |
| `CorpusType.MULTILINGUAL` | Multilingue |
| `CorpusType.EDGE_CASES` | Cas limites |

## Objectifs d'optimisation

| Constante | Description |
|-----------|-------------|
| `Objective.LATENCY` | Minimiser la latence |
| `Objective.THROUGHPUT` | Maximiser le débit |
| `Objective.BALANCED` | Compromis latence/débit |
| `Objective.MEMORY` | Minimiser la mémoire |

## Auto-tuning

```python
from libembedding import Benchmark, Objective

bench = Benchmark()
result = bench.autotune(
    "path/to/model.gguf",
    "llama.cpp",
    objective=Objective.BALANCED,
)
print(result)
```

## Benchmark d'un modèle

```python
result = bench.run(
    "path/to/model.gguf",
    "llama.cpp",
    corpus=CorpusType.MIXED,
    sessions=4,
    threads=1,
)
```

## Comparaison multi-backends

```python
comparison = bench.compare_all(
    onnx_path="path/to/model.onnx",
    gguf_path="path/to/model.gguf",
    objective=Objective.BALANCED,
)
print(comparison.summary())
```

## Sweep multi-modèles

```python
models = {
    "model_a": "path/to/model_a.onnx",
    "model_b": "path/to/model_b.gguf",
}
results = bench.sweep(models, corpora=[CorpusType.SHORT, CorpusType.MIXED])
```

## Utilitaires

```python
from libembedding import cache_path, clear_cache

path = cache_path()
clear_cache()
```

## Voir aussi

- [API Python](api_reference.html) — Vue d'ensemble
- [Statistiques runtime](python/stats.html) — Type `Stats`
