# Archives — rapport du benchmark de quantification (2026-09-30)

> ## ✅ Exécution de RÉFÉRENCE
>
> Première exécution dont les chiffres reflètent la sélection d'entrée de
> registre. C'est la source des chiffres publiés.

Ce dossier conserve le rapport **verbatim** (`results-2026-09-30.json` et
`results-2026-09-30.html`) de l'exécution qui alimente les chiffres publiés dans
le `README.md`, la documentation FR et la documentation EN.

## Référence publiée

| Emplacement | Section |
|---|---|
| `README.md` | `### Quantization: FP32 vs INT8` |
| `docs/performance_tuning.md` | `### Quantification : FP32 vs INT8` |
| `docs/en/performance_tuning.md` | `### Quantization: FP32 vs INT8` |

## Bloc d'environnement

```
timestamp 2026-09-30T20:26:46+00:00, hôte PC_Asus, Windows 11 (AMD64),
Python 3.12.10, libembedding 1.8.0 (source), 1.4.0 (wheel), git 6ef3b99
corpus de 1 000 textes (50 warmup + 950 chronométrés), modèles pré-cachés,
batchs 8 / 32 / 64 / 128, un sous-processus par configuration, 16 configurations
```

> Deux exécutions ont eu lieu le 2026-09-30. La première (18:13) mesurait encore
> le mode `dynamic` sur le corpus entier sans batch, à une époque où le script
> croyait la quantification dynamique non batchée. Elle est **remplacée** par
> celle-ci : le script a été corrigé pour mesurer tous les modes à tous les
> batches. Seuls les fichiers de ce dossier sont à jour.

## Pourquoi le rapport est archivé ici

`benchmarks/quantization/results.html` et `results.json` sont le **chemin de
sortie du benchmark** : la moindre exécution les écrase. Les y laisser serait
faire pointer la documentation publiée vers un fichier dont le contenu peut
changer sans que le README soit touché. Le rapport archivé ici est donc la
référence stable, et les chiffres publiés y sont vérifiables un à un.

Cette fois, **le JSON a pu être archivé** (contrairement à l'exécution du
2026-09-29) : la copie a été faite avant toute exécution ultérieure.

## Conclusion publiée

Les deux modèles ne se comportent pas de la même façon. Comparaison à batch
identique :

| Batch | Modèle | Variante | Poids | Débit | Pic RAM | vs FP32 |
|---|---|---|---|---|---|---|
| 64 | `all-MiniLM-L6-v2` | FP32 | 86,2 Mo | 89,0 docs/s | 206 Mo | — |
| 64 | `all-MiniLM-L6-v2` | INT8 `dynamic` | 21,9 Mo | **162,2 docs/s** | **118 Mo** | **1,8x plus rapide** |
| 32 | `bge-small-en-v1.5` | FP32 | 126,9 Mo | 48,6 docs/s | 249 Mo | — |
| 32 | `bge-small-en-v1.5` | **FP16** (`static` au registre) | 63,4 Mo | 5,1 docs/s | 176 Mo | **9,5x plus lent** |

- MiniLM dynamique : 1,8x plus rapide, 3,9x plus petit, 1,7x moins de RAM.
- BGE `_Q` : **FP16**, 9,5x plus lent, 2,0x plus petit, 1,4x moins de RAM.
- **Attention, la variante `static` du registre BGE n'est pas de l'INT8.**
  `model_optimized.onnx` de `Qdrant/bge-small-en-v1.5-onnx-Q` a 149 initialiseurs
  sur 149 en `FLOAT16`, et un graphe optimisé ORT (`Attention` 12,
  `SkipLayerNormalization` 24, `FastGelu` 12). Idem pour `bge-base`,
  `bge-large` et `paraphrase-multilingual-MiniLM-L12-v2`. Ces entrées sont
  étiquetées `LEMBED_QUANTIZATION_STATIC` et décrites « Quantized », alors que
  leurs poids sont en FP16. Le FP16 est le cas pathologique sur un CPU sans
  FP16 natif : fichier et RSS divisés par deux, puis calcul en FP32.
- Le registre ne contient **aucune variante INT8** de `bge-small-en-v1.5` : ce
  benchmark ne mesure donc pas l'INT8 statique sur ce modèle. Les 10 modèles du
  registre qui portent un vrai graphe QDQ (`Xenova/all-MiniLM-*`,
  `Snowflake/arctic-*`, `nomic-embed-text-v1.5`, `gte-base-en-v1.5`, le reranker
  Jina) sont ceux où la fusion QDQ → `QLinearMatMul` s'applique.
- Le pic RSS **est** discriminant : chaque configuration a tourné seule, donc la
  marque haute lui appartient. Les exécutions antérieures (2026-09-27,
  2026-09-29) le reportaient à tort comme non discriminant.
- Variabilité entre exécutions de l'ordre de 10 % : la référence MiniLM FP32
  batch 64 a été mesurée à 89,0 puis 99,5 docs/s sur la même machine.
- Le mode `dynamic` s'effondre au batch 128 (112,1 docs/s contre 157-162 aux
  batchs 8 à 64) : le chemin quantifié n'est pas indifférent à la taille de batch.

## Exécutions antérieures

| Dossier | Statut |
|---|---|
| `quantization-2026-09-27/` | sans valeur (mesurait le FP32 sous trois étiquettes) |
| `quantization-2026-09-29/` | sans valeur (précède le correctif de sélection de variante) |
| `quantization-2026-09-30-partial/` | exécution incomplète |

## Reproduire

```
python benchmarks/quantization/bench_quantization.py --num-texts 1000
```

Sur une machine au repos, et en archivant le rapport avant toute autre
exécution si l'on souhaite le conserver.