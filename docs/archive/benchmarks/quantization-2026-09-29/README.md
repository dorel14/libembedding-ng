# Archives — rapport du benchmark de quantification (2026-09-29)

> ## ⚠️ Exécution SANS VALEUR — ne plus publier ces chiffres
>
> Cette exécution précède le correctif de **sélection de variante de registre**.
> Le mode `quantization=` était alors appliqué au contexte sans changer le
> fichier de poids chargé : les variantes quantifiées mesurées ici n'étaient pas
> des poids distincts. Les chiffres de cette page sont donc **faux** et ont été
> retirés de la documentation le 2026-09-30.
>
> La référence publiée est
> `docs/archive/benchmarks/quantization-2026-09-30/`.
>
> Ce dossier est conservé uniquement pour la traçabilité historique (AGENTS.md §9 :
> ne jamais supprimer de la documentation sans l'archiver).

Ce dossier conserve le rapport **verbatim** de l'exécution du 2026-09-29.

## Emplacements dont ces chiffres ont été retirés

| Emplacement | Section |
|---|---|
| `README.md` | `### Quantization: FP32 vs INT8` |
| `docs/performance_tuning.md` | `### Quantification : FP32 vs INT8` |
| `docs/en/performance_tuning.md` | `### Quantization: FP32 vs INT8` |

Ces trois sections publient aujourd'hui les chiffres de
`quantization-2026-09-30/`. Les valeurs de cette page n'y figurent plus.

Bloc d'environnement de l'exécution :

```
timestamp 2026-09-29T21:19:17+00:00, hôte PC_Asus, Windows 11 (AMD64),
Python 3.12.10, libembedding 1.8.0 (source), 1.4.0 (wheel), git 6ef3b99
corpus de 1 000 textes (50 warmup + 950 chronométrés), modèles pré-cachés,
batchs 8 / 32 / 64 / 128, un sous-processus par configuration
```

## Pourquoi le rapport est archivé ici

`benchmarks/quantization/results.html` et `results.json` sont le **chemin de
sortie du benchmark** : la moindre exécution les écrase. Les y laisser serait
faire pointer la documentation vers un fichier dont le contenu peut changer sans
que le reste de la documentation soit touché. Chaque rapport est donc archivé
sous `docs/archive/benchmarks/quantization-<date>/`, et la documentation publiée
cite le dossier du rapport de référence — ici `quantization-2026-09-30/`.

## `results-2026-09-29.html`

Copie verbatim du rapport produit par cette exécution. Le `results.json`
correspondant **n'a pas pu être archivé** : il a été écrasé par une exécution
ultérieure avant d'être mis de côté. Le HTML contient les 13 mesures de
l'exécution et leur bloc d'environnement, ce qui suffit à la traçabilité
historique ; ces mesures sont fausses, voir la rétractation ci-dessous.

## Conclusion publiée (RETRACTÉE)

Poids divisés par 2 à 4, débit CPU inchangé. Le gain porte sur le stockage et le
téléchargement, pas sur la vitesse. Le pic RSS de processus (201 à 206 Mo pour
toutes les variantes) est explicitement écarté comme non discriminant : l'arène
CPU d'ONNX Runtime domine le jeu résident.

> **Rétractée le 2026-09-30.** Les trois affirmations ci-dessus sont fausses :
> le débit varie fortement selon le modèle — l'INT8 dynamique de MiniLM est plus
> rapide que le FP32, l'INT8 statique de BGE est bien plus lent — et le pic RSS
> **est** discriminant (de 118 à 249 Mo selon la variante). Les chiffres exacts
> et à jour sont dans `quantization-2026-09-30/`, seul rapport de référence.

## Reproduire

```
python benchmarks/quantization/bench_quantization.py --num-texts 1000
```

Sur une machine au repos, et en récupérant le rapport avant toute autre
exécution si l'on souhaite le conserver.