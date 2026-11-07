---
title: Auto-tuneur — C API
nav_order: 18
---

# Auto-tuneur C API

Ce module fournit l'auto-tuning pour trouver la configuration optimale sur la machine courante : threads, batch_size, et selon la tâche workers (texte) ou `max_tokens` (reranker). Les quatre tâches (texte, reranker, sparse, image) tournent sur **un même moteur de mesure** (voir [méthodologie de mesure](../performance_tuning.html#méthodologie-de-mesure)) et partagent un seul magasin de cache.

> **Disponibilité.** Comme tous les points d'entrée de l'auto-tuning, ces fonctions sont compilées dans la **bibliothèque partagée** (`python/csrc/libembedding_shlib.cpp` : DLL sous Windows, wheel Python), pas dans le build header-only `LIBEMBEDDING_IMPLEMENTATION`.

## Types

| Type | Description |
|------|-------------|
| `lembed_tuning_result_t` | Résultat de l'autotuning texte (historique) |
| `lembed_unified_tuning_result_t` | Résultat de l'autotuning unifié (toutes tâches) |
| `lembed_autotune_options_t` | Options de `lembed_autotune_ex()` : tâche, mode, objectif, workload, budget, `min_tokens` |
| `lembed_model_selection_t` | Résultat de sélection de modèle |

## Constantes

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `LEMBED_AUTOTUNE_QUICK` | `0` | Mode rapide : grille réduite, 1 warmup + 3 appels chronométrés par configuration |
| `LEMBED_AUTOTUNE_FULL` | `1` | Mode exhaustif : grille étendue, 2 warmups + 10 appels chronométrés par configuration |

Durées **indicatives**, à froid (cache vide), mesurées en QUICK sur un portable i7-1065G7 (4C/8T) : texte ~13 s, reranker ~47 s, sparse ~33 s, image ~28 s (voir [Performance](../performance_tuning.html#mesures-du-banc-bench_autotune)). Elles dépendent du CPU, du modèle et du workload ; le mode FULL est plus long, sans durée garantie. Un appel répété (cache chaud) rend en quelques millisecondes.

## Point d'entrée recommandé : `lembed_autotune_ex`

`lembed_autotune_ex()` donne accès, pour **toutes** les tâches, à l'objectif, à un workload personnalisé et aux contraintes. Les fonctions historiques (liste plus bas) restent disponibles et sont des façades du même code ; pour un nouveau code, préférer `lembed_autotune_ex()`. (Dépréciation purement documentaire : aucun attribut compilateur n'est posé.)

```c
lembed_autotune_options_t lembed_autotune_options_default(lembed_task_t task);

lembed_status_t lembed_autotune_ex(
    const char* model_name,
    const lembed_autotune_options_t* options,   /* NULL = défauts de LEMBED_TASK_EMBEDDING */
    lembed_unified_tuning_result_t* result);
```

Toujours partir de `lembed_autotune_options_default(task)`, puis surcharger les champs :

| Champ | Défaut | Sémantique |
|-------|--------|------------|
| `task` | argument de `_default()` | `LEMBED_TASK_EMBEDDING`, `_RERANKING`, `_IMAGE` ou `_SPARSE` |
| `mode` | `LEMBED_AUTOTUNE_QUICK` | QUICK ou FULL |
| `objective` | `THROUGHPUT` (texte, sparse, image) ; `BALANCED` (reranking) | Critère d'optimisation (voir [Objectifs](#objectifs)) |
| `texts`, `n_texts` | `NULL`, `0` | Workload personnalisé ; `NULL`/`0` = corpus synthétique intégré. Texte, sparse : les textes à encoder. **Reranking : le dernier texte est la requête**, les autres sont les documents. **Image : refusé** (`LEMBED_ERROR_INVALID_ARGUMENT`). |
| `max_latency_ms` | `0` (aucun) | Budget **p95 par appel de tuning**. Un appel = une requête sur tout le workload : les documents d'une requête de reranking, ou le batch texte/image complet. |
| `min_tokens` | `0` (aucun) | Reranking uniquement : plus petit `max_tokens` acceptable. |

Comportement :

- **Budget et repli.** Seules les configurations dont le p95 respecte `max_latency_ms` sont éligibles. Si aucune ne convient, la meilleure configuration **sans contrainte** est renvoyée avec `LEMBED_OK` : comparer `result.p95_latency_ms` au budget pour détecter ce cas. Pour le reranker, ce repli conserve le plancher `min_tokens`.
- **Latences.** `latency_ms` et `p95_latency_ms` sont des latences **par appel**, sauf pour `LEMBED_TASK_EMBEDDING` où elles sont amorties **par texte** (comme dans `lembed_tuning_result_t`).
- **Cache.** Le résultat est mis en cache (voir [Cache d'autotuning](#cache-dautotuning)) ; un appel répété rend en quelques millisecondes.
- **Erreurs.** `LEMBED_ERROR_MODEL_NOT_FOUND` (modèle inconnu), `LEMBED_ERROR_INVALID_ARGUMENT` (options incohérentes), `LEMBED_ERROR_ONNX_RUNTIME` (aucune configuration n'a pu s'exécuter). `lembed_last_error()` est renseigné sur tous les chemins d'erreur de l'autotune.

Exemple : reranker avec budget p95 et plancher de troncature.

```c
#include <libembedding/libembedding.h>
#include <stdio.h>

int main(void) {
    /* Workload réel : documents candidats, la requête en DERNIER. */
    const char* texts[] = {
        "Document candidat 1 ...",
        "Document candidat 2 ...",
        "Document candidat 3 ...",
        "ma requête utilisateur",          /* dernier texte = requête */
    };

    lembed_autotune_options_t opts = lembed_autotune_options_default(LEMBED_TASK_RERANKING);
    opts.mode           = LEMBED_AUTOTUNE_QUICK;
    opts.objective      = LEMBED_OBJECTIVE_LATENCY;
    opts.texts          = texts;
    opts.n_texts        = 4;
    opts.max_latency_ms = 300.0;   /* budget p95 pour une requête complète */
    opts.min_tokens     = 128;     /* ne pas tronquer les documents sous 128 tokens */

    lembed_unified_tuning_result_t r;
    lembed_status_t st = lembed_autotune_ex(
        "jinaai/jina-reranker-v1-turbo-en-quantized", &opts, &r);
    if (st != LEMBED_OK) {
        fprintf(stderr, "autotune: %s\n", lembed_last_error());
        return 1;
    }
    printf("threads=%d batch=%d max_tokens=%d p95=%.1f ms\n",
           r.threads, r.batch_size, r.max_tokens, r.p95_latency_ms);
    if (r.p95_latency_ms > opts.max_latency_ms)
        printf("budget non tenable : meilleure configuration sans contrainte renvoyée\n");
    return 0;
}
```

### Ce que le tuner règle, par tâche

| Tâche | Grille | Notes |
|-------|--------|-------|
| `LEMBED_TASK_EMBEDDING` | Découpages des cœurs physiques *P* entre workers et threads : 1×P, 2×P/2, P×1 en QUICK (+ 1×logiques et 4×P/4 en FULL) × batch (32 en QUICK ; 16/32/64/128 en FULL) | Mesure bien le modèle demandé. Corpus synthétique : 32 textes (QUICK) / 256 (FULL). |
| `LEMBED_TASK_RERANKING` | threads = cœurs physiques et logiques en QUICK (+ 1, 2, P/2 en FULL) × `max_tokens` (256/64 en QUICK ; 256/128/64/32 en FULL) × batch | `max_tokens` **tronque les documents** : c'est un compromis qualité/vitesse, à encadrer avec `min_tokens`. À égalité (5 %), la configuration qui tronque le moins l'emporte. |
| `LEMBED_TASK_SPARSE` | threads × batch | `top_k`, `min_weight` et `storage_format` ne sont **plus** réglés : ils changent les vecteurs produits, pas seulement la vitesse. Le résultat renvoie les valeurs par défaut du modèle (champs informatifs). |
| `LEMBED_TASK_IMAGE` | threads × batch | Workload personnalisé non supporté. |

## Fonctions d'autotuning

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_autotune_ex(model_name, options, out)` | `lembed_status_t` | **Recommandée.** Toutes tâches, objectif, workload, budget p95, `min_tokens` |
| `lembed_autotune_options_default(task)` | `lembed_autotune_options_t` | Options par défaut d'une tâche |
| `lembed_autotune(model_name, mode, out)` | `lembed_status_t` | Historique : auto-tuner un modèle texte |
| `lembed_autotune_custom(model_name, texts, n, mode, out)` | `lembed_status_t` | Historique : auto-tuner avec corpus custom |
| `lembed_autotune_unified(task, model_name, mode, out)` | `lembed_status_t` | Façade de `lembed_autotune_ex` avec options par défaut |
| `lembed_autotune_unified_config(task, model_name, target_latency_ms, out)` | `lembed_status_t` | Façade avec budget de latence ; honore le budget pour **toutes** les tâches, sparse inclus (il renvoyait auparavant `UNSUPPORTED` pour sparse) |
| `lembed_auto_select_model(use_case, out)` | `lembed_status_t` | Sélectionner le meilleur modèle pour la machine |
| `lembed_sparse_autotune(model_name, mode, out)` | `lembed_status_t` | Historique : auto-tuner un modèle sparse (threads × batch) |
| `lembed_image_autotune(model_name, mode, out)` | `lembed_status_t` | Historique : auto-tuner un modèle image |
| `lembed_reranker_autotune(model_name, mode, objective, out)` | `lembed_status_t` | Historique : auto-tuner un reranker |
| `lembed_reranker_autotune_custom(model_name, texts, n, mode, objective, out)` | `lembed_status_t` | Historique : auto-tuner un reranker sur un corpus fourni |
| `lembed_reranker_autotune_constrained(model_name, mode, objective, min_tokens, max_latency_ms, out)` | `lembed_status_t` | Historique : auto-tune avec contraintes |
| `lembed_reranker_auto_config(model_name, target_latency_ms, objective, out)` | `lembed_status_t` | Historique : auto-configurer selon latence (respecte maintenant `objective`) |
| `lembed_reranker_auto_config_profile(model_name, profile, out)` | `lembed_status_t` | Auto-configurer par profil |
| `lembed_quantization_auto_select(model, threads, batch_size, num_docs, dry_run, out)` | `lembed_status_t` | Choisir la variante de quantification la plus rapide sur cette machine |

Les fonctions marquées « Historique » sont conservées pour la compatibilité source et binaire (aucune signature n'a changé) ; elles exposent moins de réglages que `lembed_autotune_ex()`.

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
| `lembed_autotune_unified_clear_cache(task, model_name)` | `void` | Purger le cache d'une tâche (texte, reranker, sparse **et** image) pour un modèle ; `NULL` = tous les modèles de cette tâche (pour `LEMBED_TASK_EMBEDDING`, les entrées texte de la racine uniquement). |

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
| `LEMBED_OBJECTIVE_LATENCY` | Minimiser la latence (coût = P50 + ½(P95 − P50)) |
| `LEMBED_OBJECTIVE_THROUGHPUT` | Maximiser le débit |
| `LEMBED_OBJECTIVE_BALANCED` | Compromis : moyenne des coûts latence et débit normalisés |
| `LEMBED_OBJECTIVE_MEMORY` | Minimiser la mémoire (delta de RSS, plancher de 16 Mo) |

Les coûts sont **normalisés sans unité** (chaque terme est divisé par la meilleure valeur observée sur la grille), de sorte que ni des docs/s ni des millisecondes ne se mélangent. Défauts de `lembed_autotune_options_default()` : `THROUGHPUT` pour texte, sparse et image ; `BALANCED` pour le reranking. Le détail est dans [Performance — méthodologie de mesure](../performance_tuning.html#méthodologie-de-mesure).

## Cache d'autotuning

Un **seul magasin** pour les quatre tâches : `%LOCALAPPDATA%\libembedding\autotune\`
(`$HOME/.cache/libembedding/autotune` hors Windows, et
`./libembedding_autotune_cache` en dernier recours si aucune de ces variables
n'est définie). Les entrées du **texte** sont à la racine ; les autres tâches ont
leur sous-répertoire : `reranker/`, `sparse/`, `image/`. Les décisions de
quantification sont dans `quantization/`.

Une entrée n'est réutilisée que si son identité correspond **intégralement** :
modèle canonique, empreinte matérielle et logicielle, empreinte du workload,
objectif, mode et contraintes (budget p95, `min_tokens`). Un résultat obtenu sur
un workload n'est donc jamais renvoyé pour un autre, et un fichier illisible ou
corrompu est traité comme un cache miss. Les résultats **contraints** sont
eux aussi mis en cache, de même que ceux de sparse et d'image (ils ne l'étaient
pas avant : un appel répété relançait un tuning de 95 à 170 s).

L'identité de modèle est toujours la forme résolue (pour le texte, le `model_code` ;
voir la note sur les deux formes du nom plus haut). L'empreinte matérielle et logicielle comprend la
marque CPU, les cœurs logiques et physiques, la version majeure/mineure
d'ONNX Runtime et la version majeure/mineure de libembedding, de sorte qu'un
résultat mesuré sur une autre machine ou après un changement de version n'est
jamais resservi.

> **Invalidation à la mise à jour.** Les anciennes entrées de cache du tuner texte
> sont invalidées : elles avaient été mesurées sur le modèle par défaut quel que
> soit le modèle demandé.

`lembed_autotune_clear_cache(model_name)` supprime toutes les entrées d'un modèle
dans le magasin racine et ses sous-répertoires (un nom inconnu est sans effet) ;
`lembed_reranker_autotune_clear_cache(model_name)` fait de même pour les rerankers ;
`lembed_autotune_unified_clear_cache(task, model_name)` cible une tâche (sparse et
image compris). Passer `NULL` vide le cache de cette tâche seulement : pour
`LEMBED_TASK_EMBEDDING`, seules les entrées texte de la racine sont supprimées.
Seul `lembed_autotune_clear_cache(NULL)` supprime **tout** le répertoire
d'autotune, sous-répertoires compris. Les entrées sont appariées sur le champ `model` lu **dans** le fichier,
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

## Exemple (API historique, texte)

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
