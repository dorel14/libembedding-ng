---
title: Configuration et version — C API
nav_order: 16
---

# Configuration et version C API

Ce module gère la version de la bibliothèque et les macros de configuration.

## Version

| Macro | Description |
|-------|-------------|
| `LIBEMBEDDING_VERSION_MAJOR` | Version majeure |
| `LIBEMBEDDING_VERSION_MINOR` | Version mineure |
| `LIBEMBEDDING_VERSION_PATCH` | Version de correctif |
| `LIBEMBEDDING_VERSION_STRING` | Version complète (ex: "1.6.0") |

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_version()` | `const char*` | Version de la bibliothèque |

## Macros de configuration

| Macro | Description |
|-------|-------------|
| `LIBEMBEDDING_IMPLEMENTATION` | Active l'implémentation header-only (Linux/macOS) |
| `LIBEMBEDDING_NO_DOWNLOAD` | Désactive le téléchargement de modèles |
| `LIBEMBEDDING_NO_IMAGE` | Désactive les images (stb_image) |
| `LIBEMBEDDING_INTEGRATION_TESTS` | Active les tests d'intégration |

## Exemple

```c
#include <libembedding/config.h>
#include <stdio.h>

printf("libembedding v%s\n", LIBEMBEDDING_VERSION_STRING);
printf("Version: %d.%d.%d\n",
    LIBEMBEDDING_VERSION_MAJOR,
    LIBEMBEDDING_VERSION_MINOR,
    LIBEMBEDDING_VERSION_PATCH);
```

## Voir aussi

- [Codes d'erreur C API](error.html) — Gestion des erreurs
