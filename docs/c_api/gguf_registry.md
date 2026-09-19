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
| `lembed_find_gguf_model_by_name(name)` | `int` | Trouver un modèle GGUF par nom |
| `lembed_get_gguf_model_info(model_idx, out)` | `lembed_status_t` | Infos d'un modèle GGUF |
| `lembed_validate_gguf_path(path)` | `lembed_status_t` | Valider un chemin GGUF |
| `lembed_resolve_gguf_path(model_name, out_path)` | `lembed_status_t` | Résoudre un chemin GGUF |

## Exemple

```c
#include <libembedding/gguf_registry.h>

const lembed_model_info_t **models;
int count;
lembed_status_t status = lembed_list_gguf_models(&models, &count);
if (status == LEMBED_OK) {
    printf("GGUF models available: %d\n", count);
    for (int i = 0; i < count; i++) {
        printf("  %s (%s)\n",
            models[i]->model_name,
            models[i]->model_code);
    }
}
```

## Voir aussi

- [Modèles](models.html) — Liste des modèles (Python)
- [Téléchargement C API](downloader.html) — Téléchargement de modèles
