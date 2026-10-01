# Archives — exécution partielle du benchmark de quantification (2026-09-30 17:54)

Ces fichiers sont **archivés, pas publiés**. Ils ne doivent pas servir de
référence et ne doivent pas être cités.

## Bloc d'environnement

```
timestamp 2026-09-30T17:54:26+00:00, hôte PC_Asus, Windows 11 (AMD64),
Python 3.12.10, libembedding 1.4.0 (wheel), git 6ef3b99
```

## Trois raisons de ne pas publier

1. **Corpus 3,3x plus petit.** `warmup_texts: 15`, `timed_texts: 285`, contre 50
   et 950 pour l'exécution du 2026-09-29. Les deux exécutions ne mesurent donc
   pas la même chose, et l'exécution du 2026-09-30 n'est comparable à aucune
   autre.
2. **Balayage incomplet.** Seuls les batchs 8 et 32 figurent, contre 8 / 32 /
   64 / 128 par défaut. Les deux tableaux sont tronqués.
3. **Résultats contradictoires avec l'exécution de la veille, à configuration
   identique.**

| Configuration | 2026-09-29 | 2026-09-30 | Écart |
|---|---|---|---|
| BGE `none` @8 | 86.4 docs/s | 55.1 docs/s | −36 % |
| BGE `static` @8 | 86.1 docs/s | 4.9 docs/s | **11,3x plus lent** |
| MiniLM `dynamic` | 88.6 docs/s | 168.9 docs/s | 1,9x plus rapide |

   `static` arrive 11x plus lent que `none` sur la même machine, alors que la
   veille les deux variantes étaient à parité sur la même configuration. Un
   INT8 statique ne peut pas être 11x plus lent que le FP32 qu'il remplace : le
   signal est du bruit de contention, pas une propriété de la quantification. La
   valeur `11.3x slower` apparaît d'ailleurs telle quelle dans la colonne
   `Justification` du rapport, ce qui suffit à disqualifier la comparaison.

## Hypothèse la plus probable

La machine était occupée pendant l'exécution — un build ou un autre traitement
en parallèle. Un corpus réduit à 285 texts amplifie le bruit : moins de
documents à moyenner, et un warmup de 15 texts qui ne stabilise pas le
thread pool d'ONNX Runtime.

## Règle à retenir

Comme le rappelle `docs/archive/benchmarks/quantization-2026-09-27/README.md`,
une exécution n'est publiable que si elle est **complète** et réalisée sur une
**machine au repos**. Vérifier le bloc `environment`, le nombre de textes
chronométrés et la présence de tous les batchs attendus avant de citer un
chiffre.

## Si vous devez publier un nouveau benchmark

```
python benchmarks/quantization/bench_quantization.py --num-texts 1000
```

Machine au repos, puis archiver le rapport sous
`docs/archive/benchmarks/quantization-<date>/` et y pointer depuis le README.
Les chiffres publiés doivent être reconductibles ligne à ligne au rapport
archivé.