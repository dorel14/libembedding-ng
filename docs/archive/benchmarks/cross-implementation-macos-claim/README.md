# Archivé — comparaison inter-implémentations « mesurée sur Apple M-series »

> ## ⚠️ RÉTRAITÉE le 2026-10-02 — provenance non établie
>
> Ces chiffres ne sont **pas vérifiables** et l'attribution de plateforme était
> **fausse par construction**. Le tableau a été retiré du `README.md` et de
> `python/README.md`.

## Ce qu'était ce tableau

Une comparaison de quatre implémentations du même modèle
(`all-MiniLM-L6-v2`, 384 dimensions), présentée comme « mesurée sur Apple
M-series (macOS arm64), médiane de 10 exécutions, 1 warmup, modèles
pré-cachés » :

| Metric                   | libembedding C++ | libembedding Python | fastembed-rs (Rust) | fastembed (Python) |
|--------------------------|------------------|---------------------|---------------------|--------------------|
| Model load (ms)          | 81                | 79                  | 88                  | 84                 |
| Single text latency (ms) | 3.9               | 4.4                 | 1.9                 | 38.0               |
| Batch 8 (texts/sec)      | 632               | 641                 | 231                 | 92                 |
| Batch 32 (texts/sec)     | 687               | 581                 | 326                 | 89                 |
| Batch 128 (texts/sec)    | 626               | 449                 | 402                 | 90                 |
| Batch 512 (texts/sec)    | 526               | 476                 | 398                 | 80                 |
| Peak RSS (MB)            | 717               | 567                 | 672                 | 1,981              |

Et la synthèse qui enDerived : « 5-8x faster drop-in replacement for
fastembed », « 8.6x faster single-text latency », « 3.5x less memory ».

## Pourquoi c'est rétracté

**La plateforme était codée en dur dans les harnais de mesure.** Les quatre
scripts qui produisent ces JSON écrivaient tous la même chaîne, sans regarder
la machine sur laquelle ils tournaient :

| Fichier | Ligne | Contenu |
|---|---|---|
| `benchmarks/bench_libembedding.cpp` | 180 | `"platform": "macOS arm64",` |
| `benchmarks/bench_fastembed_rs/src/main.rs` | 124 | `"platform": "macOS arm64",` |
| `docs/archive/benchmarks/bench_fastembed_py.py` | 118 | `"platform": "macOS arm64",` |
| `docs/archive/benchmarks/bench_python_compare.py` | 115, 187 | `"platform": "macOS arm64",` |

Conséquence : **toute exécution, y compris sous Windows, s'annonçait comme une
mesure Apple Silicon.** Le champ `platform` du rapport ne pouvait donc pas
servir de preuve de provenance — il était constant.

Le mainteneur n'a pas de Mac : il ne peut ni reproduire ces chiffres, ni
confirmer qu'une machine Apple a jamais produit le JSON dont ils sont issus.

**Ce que le tableau affirmait, et qui reste non démontré :**

- que `libembedding` est 5 à 8x plus rapide que `fastembed` en Python ;
- que sa latence mono-texte est 8,6x inférieure ;
- qu'il consomme 3,5x moins de mémoire ;
- que l'API C++ est 1,7 à 2,7x plus rapide que `fastembed-rs`.

Ces quatre affirmations ont été retirées de la documentation. Elles ne
doivent pas être réintroduites tant qu'elles ne sont pas **remesurées sur une
machine identifiée**, avec le champ `platform` issu de la machine.

## Correctif appliqué

Les quatre harnais rapportent maintenant la plateforme réelle :

- C++ : `#if defined(__APPLE__)` / `_WIN32` / `__linux__` ;
- Rust : `cfg!(target_os = ...)` ;
- Python : `platform.system() + " " + platform.machine()`.

Un rapport produit par ces scripts ne peut donc plus mentir sur sa provenance.

## Ce qui reste vérifiable

Le [benchmark de quantification](../quantization-2026-10-01/) est la seule
comparaison publiée dont la provenance est vérifiable : rapport archivé avec le
bloc d'environnement complet (hôte, **CPU `Intel(R) Core(TM) i7-1065G7 CPU @
1.30GHz`**, **ONNX Runtime 1.29.0**, `CPUExecutionProvider`, version Python,
révision git), et les types de poids lus directement dans les fichiers ONNX.

## Reproduire la comparaison inter-implémentations

Elle reste faisable, mais il faut une machine identifiée et les quatre
implémentations installées :

```
python benchmarks/run_benchmarks.py            # orchestrate les 4 harnais
```

Le champ `platform` de chaque JSON indiquera alors la machine réelle, et le
tableau pourra être publié avec une provenance vérifiable. Ne pas publier de
chiffres sans ce bloc d'environnement.

## Auteur

Documente d'archivage, conservé pour traçabilité (AGENTS.md §9 : ne jamais
supprimer de la documentation sans l'archiver).

SPDX-License-Identifier: MIT
