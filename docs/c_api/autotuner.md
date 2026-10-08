---
title: Auto-tuneur — C API
nav_order: 18
---

# Auto-tuneur C API

Ce module fournit l'auto-tuning complet pour trouver la configuration optimale (workers, threads, batch_size).

## Types

| Type | Description |
|------|-------------|
| `lembed_tuning_result_t` | Résultat de l'autotuning |
| `lembed_unified_tuning_result_t` | Résultat de l'autotuning unifié |
| `lembed_model_selection_t` | Résultat de sélection de modèle |

## Constantes

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `LEMBED_AUTOTUNE_QUICK` | `0` | Mode rapide (5-15s) |
| `LEMBED_AUTOTUNE_FULL` | `1` | Mode exhaustif (30-120s) |

## Fonctions d'autotuning

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_autotune(model_name, mode, out)` | `lembed_status_t` | Auto-tuner un modèle texte |
| `lembed_autotune_custom(model_name, texts, n, mode, out)` | `lembed_status_t` | Auto-tuner avec corpus custom |
| `lembed_autotune_unified(task, model_name, mode, out)` | `lembed_status_t` | Auto-tuner unifié (tous types) |
| `lembed_autotune_unified_config(task, model_name, target_latency_ms, out)` | `lembed_status_t` | Auto-tuner unifié avec budget de latence |
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Sélectionner le meilleur modèle pour la machine |
| `lembed_sparse_autotune(model_name, mode, out)` | `lembed_status_t` | Auto-tuner un modèle sparse |
| `lembed_image_autotune(model_name, mode, out)` | `lembed_status_t` | Auto-tuner un modèle image |
| `lembed_reranker_autotune(model_name, mode, objective, out)` | `lembed_status_t` | Auto-tuner un reranker |
| `lembed_reranker_autotune_custom(model_name, texts, n, mode, objective, out)` | `lembed_status_t` | Auto-tuner un reranker sur un corpus fourni |
| `lembed_reranker_autotune_constrained(model_name, mode, objective, min_tokens, max_latency_ms, out)` | `lembed_status_t` | Auto-tune avec contraintes |
| `lembed_reranker_auto_config(model_name, target_latency_ms, objective, out)` | `lembed_status_t` | Auto-configurer selon latence |
| `lembed_reranker_auto_config_profile(model_name, profile, out)` | `lembed_status_t` | Auto-configurer par profil |
| `lembed_quantization_auto_select(model, threads, batch_size, num_docs, dry_run, out)` | `lembed_status_t` | Choisir la variante de quantification la plus rapide sur cette machine |

## Sélection automatique de la quantification

`lembed_quantization_auto_select()` répond à une question que l'appelant ne peut pas
poser seul : **sur cette machine, quel fichier de poids est le plus rapide pour ce
modèle ?** Les variantes quantifiées sont des entrées de registre distinctes avec
leur propre `model_file`, donc choisir un mode, c'est choisir un fichier.

Seules les variantes **réellement livrées** sont considérées : un modèle sans
variante `static` n'est jamais comparé à une variante `static` fantôme. Les
variantes dont les poids sont absents sont **sautées**, jamais téléchargées pour
être mesurées.

### Règle de décision

1. sans baseline FP32 mesurable → FP32 ;
2. une variante doit battre FP32 de **plus de 5 %** — en dessous, l'écart est dans
   le bruit d'un benchmark court et un échange de poids ne se justifie pas ;
3. **biais vers l'INT8 dynamique** : une autre variante doit le battre de **plus de
   15 %** pour le remplacer, car c'est l'option quantifiée la plus sûre côté
   qualité.

### Cache

La clé de cache **est** l'empreinte matérielle : elle est inscrite dans le nom du
fichier (CPU, version ONNX Runtime, version de libembedding) et l'identité de
l'entrée est re-vérifiée à la lecture. Une décision mesurée sur une autre machine
ou une autre version ne peut donc jamais être servie. Emplacement dédié
`quantization/`, pour qu'une décision de quantification n'écrase jamais un réglage
threads/batch.

Coût : **~3,5 s au tout premier chargement** d'un modèle, puis **gratuit**.

### Deux phases de mesure

Les variantes n'ont pas le même prix : sur la même machine, FP16 s'est mesuré
**6x plus lent** que l'INT8 dynamique, et aurait donc dominé toute la sélection
sans rien apprendre que quelques centaines de millisecondes n'auraient pas appris.

| Phase | Ce qu'elle fait | Bornes |
|---|---|---|
| 1 — sonde | Rejette les variantes manifestement perdantes. **4 documents max, 250 ms max** par variante. | Bornée à ~750 ms pour 3 variantes, quelle que soit leur vitesse relative |
| 2 — mesure | Corpus complet, warmup séparé, uniquement sur la baseline et les survivantes. | 16 documents, warmup de 2 |

Le budget **en temps** est la moitié importante : une sonde limitée en nombre de
documents paie encore plein pot sur une variante pathologique, ce qui est
précisément le cas qu'on veut éviter.

La sonde est un **rejeteur, pas un classement** : son premier document est hors
chronomètre (le premier `embed` paie la sélection de kernels et l'allocation), et
elle n'élimine que si la variante est plus de 3x plus lente que la meilleure vue.

### Mesures réelles (bge-small-en-v1.5, i7-1065G7)

| Variante | docs/s | Poids |
|---|---|---|
| FP32 | 15.1 | 126.9 MB |
| **INT8 dynamique** | **25.6** (1.7x) | **32.2 MB** (3.9x plus petit) |
| FP16 | 2.5 (6x plus lent) | 63.4 MB |

> **Note de mesure.** La taille rapportée est celle **des poids sur disque**, pas
> le RSS résident : le pic RSS est un high-water mark **par process**, il ne peut
> donc jamais décroître et ne peut pas être attribué à une variante mesurée après
> une autre. Compter le graphe seul serait pire : un modèle ONNX au-delà de la
> limite protobuf est un petit fichier graphe + un sidecar de poids, et le
> `model_quantized.onnx` INT8 de bge-small fait 413 Ko pour 33 Mo de poids. Le
> sidecar est donc compté, sinon l'INT8 ressortirait **plus petit** que le FP32.

### Exemple

```c
#include <libembedding/autotuner.h>

lembed_quantization_choice_t choice;
if (lembed_quantization_auto_select(LEMBED_TEXT_BGE_SMALL_EN_V15,
                                    /*num_threads*/ 0,
                                    /*batch_size*/  32,
                                    /*num_docs*/    0,   /* 0 = défaut */
                                    /*dry_run*/     0,
                                    &choice) == LEMBED_OK) {
    printf("quantization = %s\n", choice.reason);   /* ex. "1.69x the fp32 variant ..." */
    printf("from_cache   = %d\n", choice.from_cache);

    /* Ou laisser create_v2 résoudre : LEMBED_QUANTIZATION_AUTO dans les options
     * v2 déclenche exactement la même sélection, et persiste la décision. */
    lembed_text_options_v2_t opts = lembed_text_options_v2_default();
    opts.base.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.quantization = LEMBED_QUANTIZATION_AUTO;
    lembed_text_embedding_t* ctx = NULL;
    lembed_text_embedding_create_v2(&opts, &ctx);
}
```

`lembed_quantization_choice_t` est déclaré dans `types.h` et non dans
`autotuner.h` : le chemin de création doit résoudre `AUTO` sans charger la
machinerie d'autotune.

La règle de décision est une fonction **pure**, séparée de la mesure, ce qui la rend
testable en quelques millisecondes sans modèle ni réseau
(`tests/test_quantization_auto.cpp`).

> **Les deux formes du nom sont acceptées.** Le registre donne à chaque modèle
> deux chaînes différentes — le dépôt HuggingFace (`model_code`,
> `Qdrant/all-MiniLM-L6-v2-onnx`) et le nom canonique (`model_name`,
> `sentence-transformers/all-MiniLM-L6-v2`). Tous les points d'entrée de
> l'autotune résolvent l'une ou l'autre via `lembed_resolve_text_model()` /
> `lembed_resolve_reranker_model()`, et l'identité de cache est toujours la
> forme `model_code`. Un appel avec l'une des deux trouve le même modèle et purge
> les mêmes entrées.
>
> Résolution : correspondance exacte sur `model_code` d'abord, puis sur
> `model_name`. Conséquence utile : un dépôt appartient à une seule entrée, donc
> `Xenova/all-MiniLM-L6-v2` (le dépôt INT8) sélectionne l'entrée quantifiée,
> tandis que le nom canonique, qui ne porte pas de mode, sélectionne la première
> entrée le déclarant — la version FP32. Un nom inconnu renvoie `-1` avec une
> erreur, jamais l'entrée 0.
>
> Ce comportement aligne le C sur ce que les bindings Python font déjà dans
> `resolve_text_model()`, et sur ce que le benchmark reranker faisait déjà de son
> côté. Auparavant, `lembed_autotune` exigeait le dépôt HF alors que la note
> annonçait le nom canonique : les deux exemples se contredisaient.
>
> `lembed_find_text_model_variant()` (plus bas) reste volontairement stricte :
> elle apparie sur le **nom canonique**, parce que c'est le seul moyen de
> désigner « ce modèle, dans ce mode de quantification » sans ambiguïté.
>
> Les deux champs sont décrits par `lembed_model_info_t` dans
> `include/libembedding/model_registry.h`.

## Effacement du cache

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_autotune_clear_cache(model_name)` | `void` | Effacer le cache d'autotune texte ; `NULL` = tout le cache |
| `lembed_reranker_autotune_clear_cache(model_name)` | `void` | Effacer le cache d'autotune reranker ; `NULL` = tout le cache reranker |
| `lembed_autotune_unified_clear_cache(task, model_name)` | `void` | Dispatcher vers le cache de la tâche |

## Tâches unifiées

| Constante | Description |
|-----------|-------------|
| `LEMBED_TASK_EMBEDDING` | Tâche d'embedding texte |
| `LEMBED_TASK_RERANKING` | Tâche de reranking |
| `LEMBED_TASK_IMAGE` | Tâche d'embedding image |
| `LEMBED_TASK_SPARSE` | Tâche d'embedding sparse |

## Objectifs

| Constante | Description |
|-----------|-------------|
| `LEMBED_OBJECTIVE_LATENCY` | Minimiser la latence |
| `LEMBED_OBJECTIVE_THROUGHPUT` | Maximiser le débit |
| `LEMBED_OBJECTIVE_BALANCED` | Compromis |
| `LEMBED_OBJECTIVE_MEMORY` | Minimiser la mémoire |

## Cache d'autotuning

Les résultats sont persistés sous `%LOCALAPPDATA%\libembedding\autotune\`
(`$HOME/.cache/libembedding/autotune` hors Windows, et
`./libembedding_autotune_cache` en dernier recours si aucune de ces variables
n'est définie). Les rerankers ont un sous-répertoire `reranker/` séparé.

Une entrée n'est réutilisée que si son identité correspond **intégralement** :
modèle, variante, empreinte du corpus, objectif et mode. Un résultat obtenu sur
un corpus n'est donc jamais renvoyé pour un autre, et un fichier illisible ou
corrompu est traité comme un cache miss.

| Libellé de variante | Utilisé par |
|---------------------|-------------|
| `synthetic` | Tuning texte par défaut, sur corpus synthétique |
| `custom` | Tuning texte ou reranker sur un corpus fourni |
| `default` | Tuning reranker par défaut, sur corpus synthétique |

Attention : le libellé par défaut **n'est pas le même** pour le texte
(`synthetic`, dans `autotune_bench_text.hpp`) et pour le reranker (`default`, via
`default_reranker_identity()` dans `autotune_bench_reranker.hpp`). Rien dans la
sélection de variante ne dépend de ces libellés : `lembed_find_text_model_variant()`
compare le mode de quantification et le `model_name`, pas `synthetic`/`custom`.

L'identité inclut aussi une empreinte matérielle et logicielle (marque CPU,
cœurs logiques et physiques, version majeure/mineure d'ONNX Runtime, version
majeure/mineure de libembedding), de sorte qu'un résultat mesuré sur une autre
machine ou après un changement de version n'est jamais resservi.

`lembed_autotune_clear_cache(model_name)` supprime toutes les entrées d'un modèle
(toutes variantes confondues) ; `lembed_reranker_autotune_clear_cache(model_name)`
fait de même pour les rerankers. Passer `NULL` vide le cache correspondant en
entier. Les entrées sont appariées sur le champ `model` lu **dans** le fichier,
et non sur le nom de fichier, qui est un hash de l'empreinte matérielle.

> **À ne pas confondre** avec le cache bas niveau `tune_cache.json` et l'API
> `lembed_tune_cache_*` (`lembed_tune_cache_load`, `lembed_tune_cache_save`,
> `lembed_tune_cache_clear`, voir [Autotune cache](autotune_cache.md)), qui sont
> indexés par empreinte matérielle + logiciel + modèle + backend et ne sont pas
> pilotés par `lembed_autotune_clear_cache`.

## Fonction connexe : résolution de variante de quantification

`lembed_find_text_model_variant(model_name, quantization)` n'est **pas** une
fonction d'autotuning : elle appartient à l'API registre
(`include/libembedding/model_registry.h`), et n'a pas encore de page dédiée
dans cette documentation.

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_find_text_model_variant(model_name, quantization)` | `int` | Résout une demande de quantification vers l'entrée de registre qui la fournit. Retourne l'index, ou `-1` avec `lembed_last_error()` nommant le mode demandé et les modes disponibles. |
| `lembed_resolve_text_model(model)` | `int` | Résout un modèle donné par son dépôt HF **ou** son nom canonique vers son entrée. `-1` + erreur si inconnu. |
| `lembed_resolve_reranker_model(model)` | `int` | Idem pour les rerankers. |

Ces deux résolveurs sont ceux qu'utilisent les fonctions d'autotune ci-dessus.

La comparaison se fait sur `model_name` (nom canonique), pas sur `model_code` :
passer un dépôt HuggingFace ne correspond à aucune entrée.

`LEMBED_QUANTIZATION_AUTO` n'est pas une sélection de poids : aucune entrée du
registre ne porte ce mode, donc la fonction retourne toujours `-1` — **et pose
malgré tout une erreur thread-local**, comme n'importe quel mode introuvable.
`lembed_text_embedding_create_v2` n'appelle cette fonction que pour un mode
explicitement demandé, jamais pour `AUTO`.

## Exemple

```c
#include <libembedding/autotuner.h>

lembed_tuning_result_t result;
lembed_status_t status = lembed_autotune(
    "Qdrant/all-MiniLM-L6-v2-onnx",
    LEMBED_AUTOTUNE_QUICK,
    &result
);
if (status == LEMBED_OK) {
    printf("Workers: %d, Threads: %d, Batch: %d\n",
        result.workers, result.threads, result.batch_size);
}
```

## Voir aussi

- [Cache d'autotune C API](autotune_cache.html) — Cache fingerprinting
- [Worker Auto-Tune C API](worker_autotune.html) — Détection workers
