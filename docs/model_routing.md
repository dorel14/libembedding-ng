---
title: Routage des modèles
nav_order: 10
---

# Routage des modèles

> **English:** [Model routing](en/model_routing.html)

Comment la bibliothèque choisit le backend (ONNX Runtime ou llama.cpp / GGUF) pour un modèle donné, et comment vérifier ce qui tourne réellement.

## Principe

**Le backend suit le format du modèle**, pas un paramètre (`include/libembedding/text_embedding.h`, doc de `lembed_text_embedding_backend_name`) :

- un fichier `.gguf` est servi par le backend GGUF ;
- un fichier ou répertoire ONNX, un modèle du registre (identifiant HuggingFace) ou tout autre nom passe par ONNX Runtime.

Il n'y a **pas de paramètre `backend=`** sur `TextEmbedding` ni sur `SparseTextEmbedding` en Python. Deux exceptions de surface, décrites plus bas : le reranker (`backend=`) et la fonction utilitaire `detect_backend()`.

## Règles par point d'entrée

### Texte dense

| Entrée | Règle |
|---|---|
| Python `TextEmbedding(nom)` | Nom finissant par `.gguf` (casse ignorée) : llama.cpp. Soit un fichier local existant, soit `<repo>/<fichier>.gguf` (téléchargé). Sinon : chemin local (existant, ou finissant par `.onnx` / `.json`) ou modèle du registre : ONNX Runtime. Un nom inconnu lève `ModelNotFoundError` (`python/src/libembedding/text_embedding.py`, `models.py`). |
| C `lembed_text_embedding_create()` | Modèle du registre (énumération) : ONNX Runtime. |
| C `lembed_text_embedding_create_from_path()` | Chemin finissant par `.gguf` (en minuscules exactement) : llama.cpp, via `lembed_text_embedding_create_from_gguf_path()`. Sinon répertoire ONNX contenant `model.onnx` et `tokenizer.json` (`config.json` facultatif) (`detail/text_embedding_impl.hpp`). |
| C `..._create_from_gguf_path()` / `..._create_from_gguf_model()` | llama.cpp, explicitement. |

### Sparse

`lembed_sparse_text_embedding_create_from_path()` envoie un chemin finissant par `.gguf` (casse ignorée) vers le **runtime ggml maison**, pas vers llama.cpp, dont le chargeur ne sait pas lire ces fichiers (`sparse_text_embedding.h`). Tout le reste passe par ONNX Runtime. Voir [Runtime sparse GGUF](gguf_sparse_runtime.html).

### Reranker

Le reranker est le seul à exposer un sélecteur : champ `backend` de `lembed_reranker_options_t` (`LEMBED_BACKEND_ONNX`, `LEMBED_BACKEND_LLAMACPP`, `LEMBED_BACKEND_AUTO`, défaut `AUTO` dans `lembed_reranker_options_default()`) et argument `backend="auto"` de `Reranker` en Python (`"auto"`, `"onnx"`, `"llama"`).

- `LLAMACPP` : toujours GGUF.
- `ONNX` : toujours ONNX Runtime.
- `AUTO` : ONNX Runtime d'abord ; **seulement si les fichiers ONNX du modèle sont introuvables** (`LEMBED_ERROR_MODEL_NOT_FOUND`), le reranker bascule sur l'équivalent GGUF. Un fournisseur indisponible (`LEMBED_ERROR_UNSUPPORTED`) ou un modèle ONNX cassé reste une erreur : il ne provoque pas de bascule (`detail/reranker_impl.hpp`, `lembed_reranker_create`).
- `lembed_reranker_create_from_path()` envoie un chemin finissant par `.gguf` (casse ignorée) vers GGUF, et accepte aussi un répertoire ONNX (`reranker.h`).

`backend` est une option officielle du reranker **uniquement**. Le champ existe aussi dans `lembed_text_options_t`, mais `lembed_text_embedding_create()` n'en vérifie que la validité puis l'**ignore** (`detail/text_embedding_onnx_impl.hpp`). Pour le texte dense, choisir le format par le point d'entrée comme ci-dessus. Voir [Conventions de l'API](api_conventions.html).

### `detect_backend()` (Python)

`libembedding.detect_backend(model_name, backend="auto")` (`python/src/libembedding/backend.py`) est une fonction utilitaire autonome : elle renvoie `"onnx"` ou `"llama"` à partir de l'extension (`.gguf`, `.onnx`), du contenu d'un répertoire local, ou `"onnx"` par défaut. Les constructeurs Python ne l'appellent pas ; ce n'est donc pas ce qui décide du backend d'un `TextEmbedding`. Détails : [Auto-détection de backend](python/backend.html).

## Fournisseur d'exécution et backend

Le fournisseur (`provider`) est indépendant du backend : `"llama"` / `"llamacpp"` (valeur 5) n'est **pas** un fournisseur ONNX Runtime. Le passer avec un modèle ONNX échoue avec `LEMBED_ERROR_UNSUPPORTED` et le message `it is not an ONNX Runtime execution provider; load a .gguf model to use the llama.cpp backend` (`detail/onnx_session_impl.hpp`). Voir [Dépannage](troubleshooting.html).

## Vérifier le backend utilisé

### C

| Fonction | Valeurs |
|---|---|
| `lembed_text_embedding_backend_name(ctx)` | `"onnxruntime"` ou `"llama.cpp"` |
| `lembed_sparse_text_embedding_backend_name(ctx)` | `"onnxruntime"` ou `"ggml-sparse"` |
| `lembed_reranker_backend_name(ctx)` | `"onnxruntime"` ou `"llama.cpp"` |
| `lembed_image_embedding_backend_name(ctx)` | toujours `"onnxruntime"` |
| `lembed_*_effective_provider(ctx)` | ONNX Runtime : `"CPU"`, `"CUDA"`, `"CoreML"`, `"DirectML"` ou `"TensorRT"` ; llama.cpp : `"CPU"`, ou `"GPU"` si `llama_n_gpu_layers != 0` et qu'un périphérique GPU existe dans ggml (les builds livrées n'en ont pas) ; `ggml-sparse` : toujours `"CPU"` |

Ces fonctions renvoient une chaîne statique à ne pas libérer, ou `NULL` si `ctx` est `NULL`. Un fournisseur autre que CPU qui ne peut pas être attaché fait échouer la création : `effective_provider` ne rapporte donc jamais un fournisseur inutilisé.

### Python

```python
from libembedding import TextEmbedding

emb = TextEmbedding("BAAI/bge-small-en-v1.5")
print(emb.backend)   # "onnxruntime" ou "llama.cpp"
print(emb.provider)  # "cpu", "cuda"... (minuscules)
info = emb.info()    # descripteur : inclut backend et effective_provider
```

`TextEmbedding`, `SparseTextEmbedding`, `ImageEmbedding` et `Reranker` exposent les propriétés `backend` et `provider`. Pour llama.cpp, `provider` vaut `"cpu"`, ou `"gpu"` si des couches sont déchargées sur un périphérique GPU ggml (`text_embedding.py`).

### Journal

Avec `LIBEMBEDDING_VERBOSE=1` (lu une fois au démarrage du processus), la bibliothèque journalise sur stderr, notamment la bascule ONNX vers GGUF du reranker en mode `AUTO` (`reranker.h`, `detail/log.hpp`).
