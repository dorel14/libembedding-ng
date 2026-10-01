# Archives — résultats benchmark de quantification (2026-09-27)

Ces fichiers sont **archivés, pas publiés**. Voir `.kilo/plans/review_plan7.md`
pour l'analyse initiale.

## `results-stale-path.{json,html}`

Emplacement d'origine : `benchmarks/benchmarks/quantization/`.

Chemin erroné : `_REPO_ROOT` vaut `benchmarks/` (et non la racine du dépôt),
et l'ancien `--output` par défaut préfixait déjà `benchmarks/quantization/`.
Les résultats étaient donc écrits dans un répertoire `benchmarks/benchmarks/`
sans valeur. Cause corrigée dans `bench_quantization.py` (défaut `--output`
désormais `quantization/results.json`).

## `results-2026-09-27.{json,html}`

Emplacement d'origine : `benchmarks/quantization/`.

Ces résultats **ne sont pas fiables** et ne doivent pas servir de référence.

1. **Les poids comparés étaient identiques — le bug le plus grave.**
   `text_embedding.py` ne renseignait jamais `opts.base.model`, laissé à sa
   valeur d'initialisation (0). Toute entrée de registre HuggingFace était donc
   chargée depuis l'entrée 0 : `TextEmbedding("BAAI/bge-small-en-v1.5")`
   exécutait les poids **all-MiniLM-L6-v2**, et `info()` annonçait le mauvais
   modèle. Mesuré : `max |diff|` entre les embeddings de bge-small et de MiniLM
   = `0.0`, soit bit à bit identiques. Le cache paraissait correct, car le code
   Python téléchargeait bien les fichiers voulus — seule la session était
   construite depuis l'entrée 0.

   Corrigé dans `text_embedding.py`, couvert par
   `python/tests/test_quantization.py::TestModelResolution`.

2. **`quantization=` ne quantifiait rien.** L'argument ne change pas le fichier
   chargé ; il ne sélectionne que des options de session sans effet sur un
   graphe FP32. Les trois modes mesuraient donc les mêmes poids sous trois
   étiquettes, et `_auto_select_quantization` (`preferred_quantization="auto"`)
   ne faisait que choisir l'étiquette marginalement la plus rapide — du bruit.

   Corrigé : `quantization=` résout désormais l'entrée de registre qui fournit
   le mode, via `lembed_find_text_model_variant()`, et un mode absent du
   registre est refusé avec la liste des modes disponibles.

3. **Colonne mémoire inexploitable** — `peak_memory_mb` valait `227.015625` au
   bit près sur les 4 lignes. La platitude venait du point 1 : toutes les
   lignes chargeaient le même modèle, donc le même pic. Une fois le point 1
   corrigé, le pic discrimine normalement (117 à 249 Mo selon la variante).

4. **Baseline contradictoire** — la base FP32 `none@8` valait 63.76 docs/s ici
   contre 100.69 dans `results-stale-path`, soit −37 % sur le même modèle, le
   même corpus (950 documents) et la même taille de batch.

5. **Comparaison entre batchs différents** — le baseline était la première
   ligne `none` quel que soit son `batch_size`, d'où un gain annoncé de « 1.5x
   faster » là où la même comparaison donnait « 1.3x ».

6. **8 échecs identiques** — `dynamic` était mesuré à 4 tailles de batch alors
   que la bibliothèque définissait ce mode comme non batché. La garde était
   artificielle : après retrait, le mode dynamique accepte le batching et
   produit un résultat **bit à bit identique** (`max delta = 0.00e+00`).
   La garde a été supprimée.

## Reproduire

`benchmarks/quantization/bench_quantization.py` a été réécrit :

- une configuration par sous-processus, donc un pic mémoire attribuable,
- chaque variante du registre est mesurée par **son propre `model_code`**,
- les modes absents du registre sont déclarés « NOT MEASURED » avec les modes
  réellement disponibles, au lieu de produire des échecs répétés,
- deux colonnes mémoire : poids sur disque (coût de téléchargement) et pic
  résident (RAM réelle, arène ONNX Runtime comprise),
- le baseline est indexé par taille de batch,
- un bloc `environment` (timestamp, hôte, plateforme, Python, version
  distribution et version source, SHA git) accompagne les résultats.

```
python benchmarks/quantization/bench_quantization.py --num-texts 1000
```

Les résultats à publier ne doivent provenir que d'une exécution complète sur
une machine au repos.

## Résultats publiés depuis

L'exécution du **2026-09-29** remplit ces conditions et alimente les chiffres
publiés dans le `README.md`, `docs/performance_tuning.md` (FR) et
`docs/en/performance_tuning.md` (EN) — section « Quantification : FP32 vs INT8 ».
Son rapport verbatim est archivé dans
`docs/archive/benchmarks/quantization-2026-09-29/`, parce que
`benchmarks/quantization/results.html` est le chemin de sortie du benchmark et
se fait écraser à chaque exécution.

Conclusion publiée : **poids divisés par 2 à 4, débit CPU inchangé**. Le gain est
le stockage et le téléchargement, pas la vitesse.

Une exécution du 2026-09-30 17:54 a également été écartée : corpus 3,3x plus
petit, balayage incomplet, et résultats contradictoires avec la veille à
configuration identique. Voir `docs/archive/benchmarks/quantization-2026-09-30-partial/`.