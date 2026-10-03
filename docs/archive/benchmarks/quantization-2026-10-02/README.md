# Archives — rapport du benchmark de quantification (2026-10-02)

> ## ✅ Exécution de RÉFÉRENCE
>
> Rapport complet avec identité du runtime vérifiée et types de poids lus dans
> les fichiers ONNX. Remplace l'exécution du 2026-10-01.

Ce dossier conserve le rapport **verbatim** (`results-2026-10-02.json` et
`results-2026-10-02.html`) de l'exécution qui alimente les chiffres publiés dans
le `README.md`, la documentation FR et la documentation EN.

## Bloc d'environnement

```
timestamp 2026-10-02T18:34:46+00:00, hôte PC_Asus, Windows 11 (AMD64)
CPU    Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz
ORT    1.29.0        Provider CPU      git 1330f48
Python 3.12.10, libembedding 1.8.0 (source), 1.4.0 (wheel)
corpus de 1 000 textes (50 warmup + 950 chronométrés), modèles pré-cachés,
batchs 8 / 32 / 64 / 128, un sous-processus par configuration, 20 configurations
```

Le rapport contient `runtime_consistency_problems: []` — le contrôle automatique
a vérifié que les 20 configurations ont tourné sur le **même** ONNX Runtime et le
**même** provider. Sans cette garantie, deux exécutions ne seraient pas
comparables, y compris à elles-mêmes.

## Types de poids détectés dans les fichiers

Relevés directement dans les fichiers ONNX, donc impossibles à falsifier par
l'étiquette du registre :

| Modèle | Variante | Fichier | Types d'initialiseurs |
|---|---|---|---|
| `all-MiniLM-L6-v2` | `none` | `model.onnx` | `FLOAT: 101` |
| `all-MiniLM-L6-v2` | `dynamic` | `onnx/model_quantized.onnx` | `FLOAT: 101, INT64: 1, INT8: 72, UINT8: 6` |
| `bge-small-en-v1.5` | `none` | `model.onnx` | `FLOAT: 197` |
| `bge-small-en-v1.5` | `dynamic` | `onnx/model_quantized.onnx` | `FLOAT: 197, INT8: 144, UINT8: 6` |
| `bge-small-en-v1.5` | `fp16` | `model_optimized.onnx` | **`FLOAT16: 149`** |

La dernière ligne est le piège : un fichier `_Q` décrit comme « Quantized » dont
**aucun** tenseur n'est entier.

## Conclusion publiée

Meilleur débit par variante, et ratios à batch identique (batch 32) :

| Batch | Modèle | Variante | Poids | Débit | Pic RAM | vs FP32 |
|---|---|---|---|---|---|---|
| 32 | `all-MiniLM-L6-v2` | FP32 | 86,2 Mo | 96,3 docs/s | 206 Mo | — |
| 32 | `all-MiniLM-L6-v2` | INT8 `dynamic` | 21,9 Mo | **165,0 docs/s** | **118 Mo** | **1,7x plus rapide** |
| 32 | `bge-small-en-v1.5` | FP32 | 126,9 Mo | 50,5 docs/s | 249 Mo | — |
| 32 | `bge-small-en-v1.5` | INT8 `dynamic` | 32,2 Mo | **103,5 docs/s** | **120 Mo** | **2,0x plus rapide** |
| 32 | `bge-small-en-v1.5` | FP16 | 63,4 Mo | 5,0 docs/s | 177 Mo | **10,1x plus lent** |

**Sur Intel Core i7-1065G7, l'INT8 réduit la taille d'environ 4x, la mémoire de
pic d'environ 2x, et augmente le débit d'environ 1,7 à 2,0x** par rapport au
FP32.

## Observations méthodologiques

- **Variabilité de ~10 % entre exécutions** : la référence MiniLM FP32 batch 64
  a été mesurée à 82,7, 89,0, 90,5 puis 99,6 docs/s sur la même machine. Les
  écarts d'ordre de grandeur (1,7x, 2,0x, 9,5x) sont réels ; le reste est du
  bruit.
- **Le pic RSS est mesuré, pas estimé** : un sous-processus par configuration,
  donc la marque haute du jeu résident lui appartient.
- **`static` n'existe pour aucun modèle** : seuls `none`, `dynamic` et `fp16`
  sont proposés par le registre. Les combinaisons absentes sont déclarées *non
  mesurées* avec les modes disponibles.

## Exécutions antérieures

| Dossier | Statut |
|---|---|
| `quantization-2026-09-27/` | sans valeur (mesurait le FP32 sous trois étiquettes) |
| `quantization-2026-09-29/` | sans valeur (précède le correctif de sélection de variante) |
| `quantization-2026-09-30/` | sans valeur (mesurait du FP16 en le nommant INT8) |
| `quantization-2026-09-30-partial/` | exécution incomplète |
| `quantization-2026-10-01/` | superseded (mêmes chiffres d'INT8, sans identité du runtime) |
| `cross-implementation-macos-claim/` | comparaison inter-implémentations rétractée |

## Reproduire

```
python benchmarks/quantization/bench_quantization.py --num-texts 1000
python benchmarks/verify_quantization_modes.py
```

Sur une machine au repos, **sans aucun autre processus de mesure en cours** : deux
exécutions concurrentes se contaminent mutuellement et produisent des valeurs
erratiques. Archiver le rapport avant toute autre exécution si l'on veut le
conserver.

SPDX-License-Identifier: MIT