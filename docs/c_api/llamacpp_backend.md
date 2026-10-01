---
title: Backend llama.cpp — C API
nav_order: 14
---

# Backend llama.cpp C API

Ce module permet d'interagir avec le backend llama.cpp pour les modèles GGUF.

## Fonctions

L'en-tête n'expose que deux fonctions :

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_llama_backend_available()` | `int` (1=yes, 0=no) | Vérifier la disponibilité |
| `lembed_llama_version()` | `const char*` | Identifiant du backend |

> `lembed_llama_version()` ne renvoie pas un numéro de version : l'implémentation
> actuelle retourne la chaîne littérale `"llama.cpp enabled"`. Ne l'utilisez pas
> pour détecter une version.
>
> Il n'existe **pas** de `lembed_llama_set_logging()` ni de
> `lembed_llama_get_n_gpu_layers()` dans cette API. Le nombre de couches GPU se
> règle via les options du modèle (`LEMBED_BACKEND_LLAMACPP` et
> `n_gpu_layers`), et les logs via la configuration de llama.cpp elle-même.

## Exemple

```c
#include <libembedding/llamacpp_backend.h>

if (lembed_llama_backend_available()) {
    printf("llama.cpp: %s\n", lembed_llama_version());
} else {
    printf("llama.cpp backend not available\n");
}
```

## Voir aussi

- [Auto-tuneur C API](autotuner.html) — Auto-tuning avec llama.cpp
- [Worker Auto-Tune C API](worker_autotune.html) — Configuration workers
