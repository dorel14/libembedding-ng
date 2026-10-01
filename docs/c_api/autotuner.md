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
