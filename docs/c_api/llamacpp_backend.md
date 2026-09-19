---
title: Backend llama.cpp — C API
nav_order: 14
---

# Backend llama.cpp C API

Ce module permet d'interagir avec le backend llama.cpp pour les modèles GGUF.

## Fonctions

| Fonction | Retour | Description |
|----------|--------|-------------|
| `lembed_llama_backend_available()` | `int` (1=yes, 0=no) | Vérifier la disponibilité |
| `lembed_llama_version()` | `const char*` | Version de llama.cpp |
| `lembed_llama_set_logging(enable)` | `void` | Activer/désactiver les logs |
| `lembed_llama_get_n_gpu_layers(model_name)` | `int` | Nombre de couches GPU |

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
