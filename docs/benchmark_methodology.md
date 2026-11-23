---
title: Méthodologie benchmark
---

# Méthodologie des benchmarks

Version: 1.0.0

Un chiffre de performance n'a de valeur que s'il est reproductible. Cette page fixe
le protocole que suivent les benchmarks de `benchmarks/` et les métadonnées que
chaque rapport doit contenir. English: [Benchmark methodology](en/benchmark_methodology.html).

## 1. Protocole

1. **Un sous-process par backend.** Le pic de mémoire (RSS) est un *high-water mark*
   par processus : il ne peut que monter. Un processus qui a déjà exécuté un premier
   backend ne peut donc plus mesurer le pic du suivant. Un sous-process par
   configuration est la seule façon d'attribuer la mémoire, et il isole aussi un backend
   qui plante ou manque de mémoire.
2. **Warmup séparé du run chronométré.** Le warmup s'exécute sur la même instance que
   celle mesurée (le chargement paresseux et le pool de threads sont par instance),
   mais sur son propre sous-ensemble de documents, jamais sur celui qui est chronométré.
3. **Temps de chargement du modèle mesuré à part**, avant le warmup.
4. **Tours alternés pour un A/B.** Pour comparer deux configurations (ou deux builds),
   alterner A, B, A, B, ... plutôt que « tous les A puis tous les B » : la dérive
   thermique et la charge de la machine touchent alors les deux de la même façon.
5. **Médiane sur au moins 10 tours** pour une conclusion de performance, et p95 en plus
   pour la latence. Une moyenne se laisse tirer par un seul tour aberrant.
6. **Seuil de bruit : un écart inférieur à 5 % est du bruit.** On ne conclut ni à un
   gain ni à une régression en dessous de 5 %, sauf si la mesure est répétée et que
   l'écart se maintient.
7. **Machine au repos, pas de builds en parallèle.** Noter toute charge de fond.
8. **Ne pas présumer qu'une quantification est plus rapide : mesurer.**

## 2. Métadonnées obligatoires

Chaque rapport (JSON et Markdown) porte les champs suivants. Un champ absent,
`unknown` ou nul est signalé par `missing_fields()`.

| Groupe | Champs |
|---|---|
| Source | commit git (et indicateur *dirty*), version de libembedding |
| Machine | OS, modèle de CPU, cœurs physiques / logiques, RAM, jeu d'instructions détecté |
| Logiciel | version de Python, version d'ONNX Runtime ; llama.cpp / ggml quand les bindings l'exposent |
| Run | threads, batch, warmup, itérations, format (onnx/gguf), quantization, dataset |
| Résultats | p50 et p95, temps de chargement, RSS pic |

Un rapport dont le commit est *dirty* mesure du code non commité : ne pas le publier
comme référence.

## 3. Utiliser `benchmarks/_benchmeta.py`

```python
import _benchmeta as bm

machine = bm.collect_machine_metadata()
run = bm.RunParams(threads=0, batch=32, warmup=50, iterations=10,
                   format="onnx", quantization="fp32", dataset="synthetic-1000")

model, load_ms = bm.timed_load(lambda: TextEmbedding(name))
# ... warmup, puis boucle chronométrée -> samples_ms
stats = bm.summarize_latencies(samples_ms)    # n, min_ms, p50_ms, p95_ms, max_ms
peak = bm.peak_rss_mb()                       # dans le sous-process du backend

report = bm.build_report(machine, run, results={"p50_ms": stats["p50_ms"],
                                                "load_ms": load_ms, "peak_mb": peak})
assert not bm.missing_fields(report)
bm.write_json(report, "result.json")          # LF, sans BOM
print(bm.to_markdown(report))                 # bloc Markdown « Run metadata »
```

- `threads=0` signifie « valeur par défaut de la bibliothèque ».
- `batch` accepte un entier ou la liste des tailles d'un balayage.
- `percentile()` interpole linéairement entre les rangs ; `p50` et `p95` en dérivent.
- `is_noise(baseline, candidate)` applique le seuil de 5 % de la section 1.
- Les fonctions ne lèvent pas quand une source manque (pas de DLL, pas de git) : le
  champ vaut `unknown` et apparaît dans `missing_fields()`.

Scripts déjà branchés : `benchmarks/sparse/bench_sparse_backend_comparison.py`
(option `--output-json`, colonnes p50/p95 du tableau) et
`benchmarks/quantization/bench_quantization.py` (clé `metadata` du JSON).

## 4. Limites connues

- Les bindings n'exposent pas de numéro de version ggml ni llama.cpp :
  `lembed_llama_version()` indique seulement si llama.cpp est activé. Le snapshot
  vendorisé est décrit dans `third_party/README.md`.
- Le p95 sur peu d'itérations (5) est une interpolation entre les deux plus grands
  échantillons : il n'a de sens statistique qu'à partir d'une vingtaine de tours.

## 5. Smoke-test de performance en CI

Les runners hébergés par GitHub sont des machines partagées : leur vitesse absolue
varie de plusieurs dizaines de pour cent d'un run à l'autre, et aucun runner contrôlé
n'est disponible pour l'instant. La CI ne compare donc pas de docs/s. Une fois les
wheels construites, le job `perf-smoke.yml` lance `benchmarks/perf_smoke.py` sur la
wheel Linux x86_64 et ne relève que des ratios qui neutralisent l'essentiel de la
machine :

| Métrique | Définition |
|---|---|
| `dense_b32_per_gflop` | docs/s en batch 32 divisés par les GFLOP/s d'un produit matriciel float32 512×512 mesuré dans le même processus |
| `dense_b1_per_gflop` | idem en batch 1 |
| `batch_gain` | docs/s en batch 32 sur docs/s en batch 1 |

Chaque valeur est comparée à `benchmarks/perf_smoke_baseline.json`. Une baisse de
plus de 25 % (le bruit observé entre deux runs sur ces runners) produit un
avertissement GitHub. Le job ne fait jamais échouer le run : un avertissement
appelle une mesure sur machine calme, selon les sections 1 à 3. Le rapport JSON est
conservé 90 jours dans l'artefact `perf-smoke-linux-x86_64`.

Pour mettre à jour la référence après un changement voulu, recopier les trois
valeurs affichées par le job dans `perf_smoke_baseline.json`, sous la clé
`linux-x86_64`.

## 6. Campagnes comparatives (LE-11) et multi-vecteurs

Deux campagnes complètes se rejouent sur machine calme (branchée, rien d'autre en
cours), un sous-processus par mesure, avec pic RSS, échauffement séparé, trois passes
chronométrées (médiane) et l'ordre des bibliothèques tourné entre les rounds. Chaque
script écrit son JSON après chaque cellule et reprend là où il s'est arrêté.

| Campagne | Script | Résultats |
|---|---|---|
| libembedding contre SentenceTransformers et FastEmbed (dense, sparse, qualité sur 100 paires BEIR, batch 1 à 128, 3 réglages de threads) | `benchmarks/competitive/bench_competitive.py full` | `benchmarks/competitive/results.md` et `results.json` |
| `MultiVectorEmbedding` (ColBERT) contre pylate, et noyau MaxSim contre numpy | `benchmarks/multivector/bench_multivector.py full` | `benchmarks/multivector/results.md` et `results.json` |

Pour mesurer le code source et non la wheel PyPI, définir `LIBEMBEDDING_LIBRARY` vers
la bibliothèque compilée et ajouter `python/src` à `PYTHONPATH` (le multi-vecteur
n'existe pas encore dans la wheel).
