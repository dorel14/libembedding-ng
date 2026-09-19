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
| `3` | `LEMBED_ERROR_RUNTIME` | Erreur runtime (ONNX Runtime) |
| `4` | `LEMBED_ERROR_TOKENIZER` | Erreur de tokenizer |
| `5` | `LEMBED_ERROR_DOWNLOAD` | Erreur de téléchargement |
| `6` | `LEMBED_ERROR_IO` | Erreur d'entrée/sortie |
| `7` | `LEMBED_ERROR_MODEL_NOT_FOUND` | Modèle non trouvé |
| `8` | `LEMBED_ERROR_UNSUPPORTED` | Fonctionnalité non supportée |
| `9` | `LEMBED_ERROR_BATCH_SIZE` | Taille de batch invalide |
| `10` | `LEMBED_ERROR_LLAMA` | Erreur llama.cpp |

## Fonctions de gestion d'erreur

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_last_error()` | `const char*` | Dernier message d'erreur (thread-local) |
| `lembed_status_to_string(status)` | `const char*` | Nom du code d'erreur |

## Exemple

```c
#include <libembedding/error.h>

lembed_status_t status = lembed_text_embedding_create(...);
if (status != LEMBED_OK) {
    printf("Error: %s\n", lembed_last_error());
    printf("Code: %s\n", lembed_status_to_string(status));
}
```

## Voir aussi

- [Configuration C API](config.html) — Version et macros
