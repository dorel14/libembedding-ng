---
title: Usage avancé
nav_order: 6
---

# Usage avancé

## Modèles locaux (Bring Your Own ONNX)

Vous pouvez utiliser vos propres modèles ONNX sans les télécharger. Placez simplement vos fichiers dans un répertoire local :

```
mon_modele/
├── model.onnx
├── tokenizer.json
└── config.json       # optionnel mais recommandé
```

Puis chargez-le en passant le chemin :

```python
from libembedding import TextEmbedding

model = TextEmbedding("/path/to/mon_modele")
embeddings = model.embed(["Hello world"])
```

### Paramètres pour modèles locaux

Si votre modèle n'a pas de `config.json`, vous devez spécifier manuellement :

```python
model = TextEmbedding(
    "/path/to/mon_modele",
    dim=768,           # dimension de l'embedding
    pooling="mean",    # "cls" ou "mean"
    max_length=512,    # longueur max en tokens
)
```

## Providers d'exécution

Le paramètre `provider` contrôle le backend ONNX Runtime utilisé pour l'inférence :

```python
from libembedding import TextEmbedding

# CPU (défaut, toujours disponible)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="cpu")

# NVIDIA CUDA
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="cuda", device_id=0)

# Apple CoreML (macOS/iOS)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="coreml")

# DirectML (Windows)
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="directml")

# NVIDIA TensorRT
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="tensorrt")
```

**Providers supportés :**

| Provider | Backend | Plateforme |
|----------|---------|-----------|
| `cpu` | CPU | Toutes |
| `cuda` | CUDA | NVIDIA GPU |
| `coreml` | CoreML | macOS, iOS |
| `directml` | DirectML | Windows |
| `tensorrt` | TensorRT | NVIDIA GPU |
| `llamacpp` | llama.cpp | Toutes |

### Modèles GGUF (llama.cpp)

Chargez des modèles `.gguf` quantifiés via le backend llama.cpp :

```python
from libembedding import TextEmbedding

# Charger un modèle GGUF depuis HuggingFace
model = TextEmbedding(
    "Xenova/all-MiniLM-L6-v2-GGUF/all-MiniLM-L6-v2-Q4_K_M.gguf"
)

# Ou charger depuis un fichier local
model = TextEmbedding("/path/to/model.Q4_K_M.gguf")

embeddings = model.embed(["Hello world"])
```

Le routage se fait dans le constructeur : `_is_gguf_model()` reconnaît l'extension
`.gguf` et bascule sur le backend llama.cpp. Il n'existe **pas** de méthode
`TextEmbedding.supports_llamacpp()` côté Python ; pour savoir si le backend est
disponible à l'exécution, utilisez `detect_backend(model_name)` ou l'appel C
`lembed_llama_backend_available()`.

### Auto-tuning workers (llama.cpp)

Pour le backend llama.cpp, le nombre optimal de sessions/workers peut être détecté automatiquement :

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,      # détecte automatiquement les sessions optimales
    cache_size=4096,        # cache LRU optionnel
)
```

## Cache LRU d'embeddings

Activez un cache LRU pour éviter les inférences répétées sur les mêmes textes :

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_size=8192,  # 8192 entrées max, 0 = désactivé
)

# Les embeddings sont mis en cache automatiquement
embeddings = model.embed(["Hello world", "Hello world"])  # 2e appel = cache hit
```

### Cas d'usage du cache

- **Benchmarks** : éviter de re-calculer les mêmes embeddings
- **Indexation incrémentale** : détecter les doublons
- **Développement** : accélérer les itérations

### Répertoire de cache par défaut

Les modèles sont téléchargés et mis en cache dans `~/.cache/libembedding` par défaut.

### Changer le répertoire de cache

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    cache_dir="/custom/cache/dir"
)
```

### Variables d'environnement

| Variable | Description |
|----------|-------------|
| `LIBEMBEDDING_CACHE_DIR` | Redéfinit le répertoire de cache globalement |
| `FASTEMBED_CACHE_DIR` | Répertoire de cache alternatif (compatibilité fastembed) |
| `HF_ENDPOINT` | URL personnalisée pour le HuggingFace Hub (ex: miroir) |

```bash
# Exemples d'utilisation
export LIBEMBEDDING_CACHE_DIR=/data/models/libembedding
export HF_ENDPOINT=https://hf-mirror.com
```

## Mode hors-ligne

Utilisez le mode hors-ligne pour fonctionner sans accès réseau :

```python
from libembedding import TextEmbedding

# Tente uniquement de charger depuis le cache local
model = TextEmbedding("BAAI/bge-small-en-v1.5", offline=True)
```

Ou définissez la variable d'environnement :

```bash
export LIBEMBEDDING_NO_DOWNLOAD=1
```

## Context managers

Toutes les classes supportent le protocole context manager pour une gestion automatique des ressources C :

```python
from libembedding import TextEmbedding

with TextEmbedding("BAAI/bge-small-en-v1.5") as model:
    embeddings = model.embed(["Hello", "World"])
# Les ressources sont libérées automatiquement à la sortie du bloc
```

## Gestion manuelle des ressources

Si vous n'utilisez pas de context manager, appelez explicitement `close()` :

```python
from libembedding import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
try:
    embeddings = model.embed(["Hello", "World"])
finally:
    model.close()
```

## Batch processing

Contrôlez la taille de batch pour équilibrer vitesse et mémoire :

```python
from libembedding import TextEmbedding

# Taille de batch au niveau du constructeur
model = TextEmbedding("BAAI/bge-small-en-v1.5", batch_size=64)

# Ou au niveau de l'appel embed()
model = TextEmbedding("BAAI/bge-small-en-v1.5", batch_size=256)
embeddings = model.embed(texts, batch_size=128)  # override temporaire
```

## Limite de tokens

Certains modèles ont une longueur maximale de tokens :

```python
from libembedding import TextEmbedding

# Limiter à 128 tokens (par défaut 512 pour BGE-small)
model = TextEmbedding("BAAI/bge-small-en-v1.5", max_length=128)
```

## Recherche de modèles par code

libembedding recherche automatiquement les modèles par nom HuggingFace ou par code de repo ONNX :

```python
from libembedding import TextEmbedding

# Par nom HuggingFace complet
model = TextEmbedding("BAAI/bge-small-en-v1.5")

# Par code de repo ONNX
model = TextEmbedding("Xenova/bge-small-en-v1.5")

# Par chemin local
model = TextEmbedding("/path/to/local/model")
```

## Intégration avec numpy

Les embeddings retournés sont des tableaux numpy `float32` L2-normalisés :

```python
from libembedding import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["doc1", "doc2", "doc3"])

# Similarité cosinus (produit scalaire car embeddings L2-normalisés)
similarity = embeddings @ embeddings.T

# Recherche du plus proche voisin
query = model.embed(["search query"])
scores = embeddings @ query.T
best_idx = scores.argmax()
```

## Recherche sémantique complète

```python
from libembedding import TextEmbedding
import numpy as np

class SemanticSearch:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list[str]):
        self.documents.extend(documents)
        new_embeddings = self.model.embed(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
        query_emb = self.model.embed([query])
        scores = (self.embeddings @ query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i]), self.documents[i]) for i in top_indices]

    def close(self):
        self.model.close()
```

## Comparaison avec fastembed

libembedding est conçu comme un remplacement drop-in de fastembed :

```python
# fastembed
from fastembed import TextEmbedding as FastEmbed
model = FastEmbed("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["Hello"]))

# libembedding — même API
from libembedding import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = model.embed(["Hello"])  # retourne np.ndarray directement
```

**Différences notables :**
- `embed()` retourne directement un `np.ndarray` au lieu d'un générateur
- Pas de méthode `passage_embed()` séparée — tout passe par `embed()`
- Classes plus légères, moins de dépendances Python

## Performance

Pour des performances optimales :

1. **Réutilisez le modèle** — créez-le une fois et embeds plusieurs batchs
2. **Ajustez `batch_size`** — augmentez jusqu'à saturation mémoire
3. **Utilisez un provider GPU** — `cuda`, `coreml`, ou `directml`
4. **Activez la quantification** — choisissez les modèles `_Q` pour diviser la taille sur disque par 2 à 4 (l'effet sur le débit dépend du modèle : mesurez-le, voir [Performance](performance_tuning.html#quantification--fp32-vs-int8))
5. **Désactivez la progression** — `show_download_progress=False` pour les scripts

## Scheduler de batching dynamique (llama.cpp)

Pour le backend llama.cpp, le scheduler regroupe dynamiquement les requêtes en lots pour améliorer le débit :

```python
from libembedding import TextEmbedding

model = TextEmbedding(
    "BAAI/bge-small-en-v1.5-GGUF",
    auto_workers=True,
)

# Le scheduler est activé automatiquement selon la configuration
embeddings = model.embed(texts)
```

### Stratégie de batching

| Stratégie | Description | Cas d'usage |
|-----------|-------------|-------------|
| `naive` | Traitement séquentiel | Textes de longueur similaire |
| `length_bucket` | Tri par longueur puis batch | Corpus hétérogène (short + long) |

```python
# Force le bucketing pour du texte hétérogène
model = TextEmbedding(
    "BAAI/bge-small-en-v1.5",
    batch_strategy="length_bucket",
)
```

> **Note** : Le bucketing est actuellement supporté sur le backend ONNX. Le support llama.cpp est en cours d'évaluation.
