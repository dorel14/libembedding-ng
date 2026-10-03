---
title: Registre de modèles GGUF — C API
nav_order: 21
---

# Registre de modèles GGUF C API

Ce module gère la liste des modèles GGUF recommandés et leur configuration.

## Types

| Type | Description |
|------|-------------|
| `lembed_gguf_model_info_t` | Informations sur un modèle GGUF |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_list_gguf_models(out, count)` | `lembed_status_t` | Liste des modèles GGUF |
| `lembed_find_gguf_model(name)` | `const lembed_gguf_model_info_t*` | Chercher un modèle GGUF (NULL si inconnu) |
| `lembed_default_gguf_model()` | `const lembed_gguf_model_info_t*` | Modèle GGUF recommandé par défaut |

> Les deux fonctions de recherche retournent un **pointeur vers les données
> statiques** — ne pas libérer. Il n'existe pas de
> `lembed_get_gguf_model_info()` : on itère directement sur le tableau renvoyé
> par `lembed_list_gguf_models()`. `lembed_resolve_gguf_path()` est déclarée
> dans `downloader.h`, pas ici.

## Champs de `lembed_gguf_model_info_t`

| Champ | Type | Description |
|-------|------|-------------|
| `name` | `const char*` | Nom lisible (ex. `"Snowflake-XS-Q4"`) |
| `gguf_url` | `const char*` | URL de téléchargement du `.gguf` |
| `model_code` | `const char*` | Code HuggingFace d'origine |
| `description` | `const char*` | Description courte |
| `dim` | `int` | Dimension d'embedding |
| `params_m` | `int` | Paramètres en millions |
| `file_size_mb` | `int` | Taille approximative en Mo |
| `quality_mteb` | `float` | Score MTEB Retrieval (NDCG@10), 0 si inconnu |
| `recommended_sessions` | `int` | Nombre de sessions recommandé |

> Le champ s'appelle `name`, **pas** `model_name`.

## Exemple

```c
#include <libembedding/gguf_registry.h>

const lembed_gguf_model_info_t *models;
int count;
lembed_status_t status = lembed_list_gguf_models(&models, &count);
if (status == LEMBED_OK) {
    printf("GGUF models available: %d\n", count);
    for (int i = 0; i < count; i++) {
        printf("  %s (%s, %d dim)\n",
            models[i]->name,
            models[i]->model_code,
            models[i]->dim);
    }
}
```

## Voir aussi

- [Modèles](models.html) — Liste des modèles (Python)
- [Téléchargement C API](downloader.html) — Téléchargement de modèles
