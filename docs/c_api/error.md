---
title: Codes d'erreur — C API
nav_order: 15
---

# Codes d'erreur C API

Ce module définit les codes de statut et la gestion des erreurs thread-local.

## `lembed_status_t`

| Valeur | Nom | Description |
|--------|-----|-------------|
| `0` | `LEMBED_OK` | Succès |
| `1` | `LEMBED_ERROR_INVALID_ARGUMENT` | Argument invalide |
| `2` | `LEMBED_ERROR_OUT_OF_MEMORY` | Mémoire insuffisante |
| `3` | `LEMBED_ERROR_ONNX_RUNTIME` | Erreur ONNX Runtime |
| `4` | `LEMBED_ERROR_TOKENIZER` | Erreur de tokenizer |
| `5` | `LEMBED_ERROR_DOWNLOAD` | Erreur de téléchargement |
| `6` | `LEMBED_ERROR_IO` | Erreur d'entrée/sortie |
| `7` | `LEMBED_ERROR_MODEL_NOT_FOUND` | Modèle non trouvé |
| `8` | `LEMBED_ERROR_UNSUPPORTED` | Fonctionnalité non supportée |
| `9` | `LEMBED_ERROR_BATCH_SIZE` | Taille de batch invalide |
| `10` | `LEMBED_ERROR_LLAMA` | Erreur llama.cpp |
| `11` | `LEMBED_ERROR_CACHE_MISS` | Cache miss |

> `LEMBED_ERROR_BATCH_SIZE` est conservé dans l'enum mais **plus aucune ligne de
> code ne le retourne** : la garde qui le produisait a été retirée, la
> quantification dynamique accepte de nouveau le batching. Il reste dans l'API
> pour ne pas rompre l'ordre des valeurs (l'ABI et le mapping Python
> `_STATUS_MAP` en dépendent).

## Fonctions de gestion d'erreur

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_last_error()` | `const char*` | Dernier message d'erreur (thread-local) |
| `lembed_status_message(status)` | `const char*` | Message du code d'erreur |
| `lembed_version()` | `const char*` | Version de la bibliothèque |

> La fonction porte le nom `lembed_status_message()`, **pas**
> `lembed_status_to_string()`. Le préfixe `LEMBED_ERROR_` des constantes ne se
> retrouve pas non plus dans les noms de fonctions.

## Exemple

```c
#include <libembedding/error.h>

lembed_status_t status = lembed_text_embedding_create(...);
if (status != LEMBED_OK) {
    printf("Error: %s\n", lembed_last_error());
    printf("Code: %d (%s)\n", status, lembed_status_message(status));
}
```

## Voir aussi

- [Configuration C API](config.html) — Version et macros
