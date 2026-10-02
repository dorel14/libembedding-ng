# Archives — rapport du benchmark de quantification (2026-10-01)

> ## ⚠️ REMPLACÉE le 2026-10-02 par `quantization-2026-10-02/`
>
> Ces chiffres sont superseded et ne sont plus publiés. Le rapport du 2026-10-02
> mesure les mêmes variantes avec, en plus, l'identité du runtime vérifiée
> (ONNX Runtime 1.29.0, `CPUExecutionProvider`, CPU `Intel(R) Core(TM)
> i7-1065G7 CPU @ 1.30GHz`) et les types de poids lus dans les fichiers ONNX.
> Conservé pour la traçabilité (AGENTS.md §9).

> ## Exécution mesurée, désormais superseded
>
> Première exécution à mesurer un **vrai INT8** sur les deux modèles.

Ce dossier conserve le rapport **verbatim** (`results-2026-10-01.json` et
`results-2026-10-01.html`) de l'exécution qui alimente les chiffres publiés dans
le `README.md`, la documentation FR et la documentation EN.

## Référence publiée

| Emplacement | Section |
|---|---|
| `README.md` | `### Quantization: FP32 vs INT8` |
| `docs/performance_tuning.md` | `### Quantification : FP32 vs INT8` |
| `docs/en/performance_tuning.md` | `### Quantization: FP32 vs INT8` |

## Bloc d'environnement

```
timestamp 2026-10-01T21:01:19+00:00, hôte PC_Asus, Windows 11 (AMD64),
Python 3.12.10, libembedding 1.8.0 (source), 1.4.0 (wheel), git c2dbe95
corpus de 1 000 textes (50 warmup + 950 chronométrés), modèles pré-cachés,
batchs 8 / 32 / 64 / 128, un sous-processus par configuration, 16 configurations
```

## Conclusion publiée

Comparaison à batch identique (batch 64) :

| Modèle | Variante | Poids | Débit | Pic RAM | vs FP32 |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | FP32 | 86,2 Mo | 82,7 docs/s | 205 Mo | — |
| `all-MiniLM-L6-v2` | INT8 `dynamic` | 21,9 Mo | **158,7 docs/s** | **118 Mo** | **1,9x plus rapide** |
| `bge-small-en-v1.5` | FP32 | 126,9 Mo | 42,4 docs/s | 249 Mo | — |
| `bge-small-en-v1.5` | INT8 `dynamic` | 32,2 Mo | **94,4 docs/s** | **120 Mo** | **2,2x plus rapide** |
| `bge-small-en-v1.5` | FP16 | 63,4 Mo | 5,0 docs/s | 177 Mo | **8,5x plus lent** |

**L'INT8 dynamique gagne sur tous les axes** : 1,9x à 2,2x plus rapide, 3,9x
plus petit sur disque, 1,7 à 2,1x moins de RAM. Le seul compromis est la
précision, que ce benchmark ne mesure pas.

## Ce que cette exécution a corrigé

- **Un modèle INT8 réel pour BGE.** `onnx-community/bge-small-en-v1.5-ONNX`,
  `onnx/model_quantized.onnx` : 33,4 Mo de données pour 132,8 Mo en FP32 (un
  quart), 144 initialiseurs `INT8` + 6 `UINT8`, graphe de quantification
  dynamique (72 `MatMulInteger` alimentés par 48 `DynamicQuantizeLinear`).
  Enregistré sous `LEMBED_TEXT_BGE_SMALL_EN_V15_INT8`, avec son fichier de
  données externes déclaré dans `lembed__additional_files`.
- **Les quatre entrées `Qdrant/*-onnx-Q` sont FP16, pas INT8.** Leurs 149
  initialiseurs sont `FLOAT16` et leur graphe est optimisé ORT (`Attention` 12,
  `SkipLayerNormalization` 24, `FastGelu` 12). Elles déclaraient
  `LEMBED_QUANTIZATION_STATIC` avec une description « Quantized », donc
  `quantization="static"` renvoyait des poids float16 — d'où le **8,5x plus
  lent** mesuré ici. Elles déclarent maintenant `LEMBED_QUANTIZATION_FP16`, et
  `quantization="static"` échoue proprement au lieu de mentir.
- **Validateur automatique** : `benchmarks/verify_quantization_modes.py`
  compare le mode déclaré de chaque entrée aux dtypes réellement présents dans
  son fichier. Il signalait **4 incohérences** avant le correctif, **0** après.
- **`static` n'existe plus pour aucun modèle du registre.** Le mode `static`
  (quantification INT8 avec calibration) n'est proposé par aucune entrée ; seuls
  `none`, `dynamic` et `fp16` le sont. Les combinaisons absentes sont déclarées
  *non mesurées* avec les modes disponibles.

## Observations méthodologiques

- **Variabilité de ~10 % entre exécutions** : la référence MiniLM FP32 batch 64
  a été mesurée à 82,7, 89,0 puis 99,5 docs/s sur la même machine. Les écarts
  d'ordre de grandeur (1,9x, 2,2x, 8,5x) sont réels, le reste est du bruit.
- **Le pic RSS est mesuré, pas estimé** : un sous-processus par configuration,
  donc la marque haute du jeu résident lui appartient. L'arène d'ONNX Runtime
  s'ajoute au fichier (118 Mo pour 21,9 Mo de poids), mais l'écart entre
  configurations est réel.
- **Le chemin quantifié est stable sur les batchs** (142,8 à 159,5 docs/s sur
  MiniLM, 83,1 à 94,7 sur BGE). L'effondrement à 112,1 docs/s au batch 128
  observé lors de l'exécution du 2026-09-30 ne s'est pas reproduit.

## Exécutions antérieures

| Dossier | Statut |
|---|---|
| `quantization-2026-09-27/` | sans valeur (mesurait le FP32 sous trois étiquettes) |
| `quantization-2026-09-29/` | sans valeur (précède le correctif de sélection de variante) |
| `quantization-2026-09-30/` | sans valeur (mesurait du FP16 en le nommant INT8) |
| `quantization-2026-09-30-partial/` | exécution incomplète |

## Reproduire

```
python benchmarks/quantization/bench_quantization.py --num-texts 1000
python benchmarks/verify_quantization_modes.py
```

Sur une machine au repos, et en archivant le rapport avant toute autre
exécution si l'on souhaite le conserver.
