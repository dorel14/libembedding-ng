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

# Apple CoreML (macOS/iOS) : UnsupportedError s'il ne peut pas être attaché
model = TextEmbedding("BAAI/bge-small-en-v1.5", provider="coreml")
print(model.backend, model.provider)  # onnxruntime coreml
```

**Providers supportés :**

| Provider | Backend | Plateforme |
|----------|---------|-----------|
| `cpu` | CPU | Toutes |
| `coreml` | CoreML | macOS, iOS |
| `llamacpp` | llama.cpp (CPU), modèles GGUF uniquement | Toutes |
| `cuda`, `directml`, `tensorrt` | réservés : `UnsupportedError` | — |

L'accélération GPU est hors périmètre avant la 1.x. Les valeurs `cuda`,
`directml` et `tensorrt` existent pour garder l'API stable, mais aucune build ne
les active. Un provider autre que `cpu` demandé mais indisponible lève
`UnsupportedError` (`LEMBED_ERROR_UNSUPPORTED` en C) au lieu de tourner sur CPU
sans le dire ; les propriétés `backend` et `provider` indiquent ce qui sert
réellement le modèle (voir [Backend et provider effectifs](api_reference.html#backend-et-provider-effectifs)).

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
| `LIBEMBEDDING_DOWNLOAD_STALL_TIMEOUT` | Secondes sans aucun octet reçu avant d'abandonner un téléchargement et de le relancer (30 par défaut ; 3 tentatives au plus) |
| `ORT_DISABLE_TELEMETRY` | Télémétrie d'ONNX Runtime : libembedding la met à `1` si elle est absente (aucune donnée envoyée à Microsoft) ; `0` conserve la télémétrie d'ORT |
| `LIBEMBEDDING_VERBOSE` | `1` affiche les diagnostics sur stderr (progression de l'autotune, sélection automatique du modèle, simulation du nettoyage du cache) ; silencieux par défaut |
| `LIBEMBEDDING_ALLOW_RESTRICTED_LICENSES` | `1` charge les modèles du registre dont la licence n'est pas compatible MIT (`jina-reranker-v2-base-multilingual`, CC-BY-NC-4.0 ; `embeddinggemma-300m`, Gemma), refusés par défaut |

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

## Fidélité du tokenizer (HuggingFace `tokenizers`)

Le tokenizer intégré est comparé à la bibliothèque HuggingFace `tokenizers` sur un
corpus multilingue (grec, cyrillique, vietnamien, CJK, arabe, hébreu, hindi, thaï,
coréen, emoji, latin accentué, casse mixte, ponctuation, mots très longs). Les
références sont des fixtures JSON versionnées dans `tests/data/tokenizer/`, produites
par `scripts/gen_tokenizer_fixtures.py` ; le test `test_tokenizer_fidelity` n'a besoin
ni de Python ni du paquet `tokenizers`.

| Famille | Modèles | Statut |
|---|---|---|
| WordPiece + BertNormalizer (uncased) | MiniLM, bge-*-en, snowflake, gte-en, nomic, mxbai, SPLADE | identique à HF, 0/54 textes différents |
| WordPiece, vocabulaire chinois | bge-*-zh | identique à HF, 0/54 |
| WordPiece, `<s>`/`</s>` | all-mpnet-base-v2 | identique à HF, 0/54 |
| Unigram (SentencePiece, `nmt_nfkc`) | multilingual-e5, paraphrase-multilingual, bge-m3, bge-reranker | identique à HF, 0/54 (charsmap précompilé, WhitespaceSplit + Metaspace, Viterbi, fusion des inconnus, byte fallback) ; embeddings à cosinus 1,000 de la référence. Paires du reranker : pas encore couvertes |
| BPE byte-level (`Split` + `ByteLevel`, suffixe `</w>`) | CLIP texte | identique à HF, 0/54 ; embeddings à cosinus 1,000 de la référence (avant : min 0,29, moyenne 0,76). NFC limité au latin (lettres + diacritiques U+0300-036F), tables Unicode 15.0 pour `\p{L}`/`\p{N}` |
| BPE byte-level, autres (ModernBERT, GPT-2) | ModernBERT | reconnu si le JSON utilise les mêmes briques, sinon repli sur l'ancien chemin ; non mesuré |

Couverture de la normalisation (strip accents + minuscules), vérifiée point de code par
point de code jusqu'à U+2FFFF : Latin (Latin-1, Étendu-A/B, Étendu additionnel dont le
vietnamien), IPA, diacritiques combinants, grec et copte, grec étendu (polytonique),
cyrillique, arménien, hébreu, arabe, devanagari, thaï, jamo et syllabes hangul
(décomposition algorithmique), ponctuation et symboles généraux (jusqu'à U+2BFF),
kana, CJK, formes pleine chasse, emoji : 0 écart. Reste hors table (1,0 % des points
de code testés, 1 835 sur 183 376) : formes de compatibilité CJK (U+F900-FAFF et
U+2F800-2FA1F), capitales cherokee, glagolitique, copte, cyrillique étendu B, deseret
et autres alphabets rares, lettres indiennes précomposées hors devanagari, formes de
présentation hébraïques. Ces points de code traversent le tokenizer inchangés, ce qui
diffère de HF.

Le même travail a corrigé trois écarts de pré-tokenisation avec HF : un mot dont une
partie est inconnue donne un seul `[UNK]` (et non les morceaux déjà trouvés), un mot
de plus de 100 caractères donne `[UNK]`, et les ponctuations Unicode, les idéogrammes
CJK et les caractères de format (espace de largeur nulle, etc.) sont traités comme
dans HF.

Les options du `BertNormalizer` sont lues indépendamment, comme dans HF : `lowercase`,
`strip_accents` (absent ou `null` : suit `lowercase`), `clean_text` et
`handle_chinese_chars`. Un normaliseur `Sequence` est pris en charge pour `Lowercase`,
`StripAccents` et `NFD`/`NFKD` ; les accents ne sont retirés que si `StripAccents` et une
décomposition sont tous deux présents. Les modes « minuscules seules » et « accents
seuls » sont aussi vérifiés point de code par point de code. Un BPE byte-level dont le
`tokenizer.json` n'est pas entièrement compris garde son comportement historique (non
conforme à HF) ; ses ids sont figés par `test_tokenizer`. Les tests d'ids par famille ne s'exécutent que si le
`tokenizer.json` exact est dans le cache du poste (sinon la famille est ignorée).

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

## Thread safety

La règle : **un contexte peut être partagé entre threads, mais les appels sur
ce contexte sont sérialisés** (texte, sparse, reranker). Les contextes image
exécutent les appels en parallèle. Pour du parallélisme, utiliser un contexte
par thread ou `TextEmbeddingPool`. Ne jamais libérer un contexte pendant qu'un
autre thread l'utilise, ni rappeler le même contexte depuis le callback de
`embed_stream` (il tient le verrou du contexte : interblocage).

| Type | Partageable entre threads ? | Pourquoi (source) |
|---|---|---|
| Texte, ONNX et GGUF | Oui, appels sérialisés | `call_mutex` dans `detail/text_embedding_impl.hpp:125,169` ; le tokenizer écrit des chaînes temporaires et le backend GGUF n'a qu'un `llama_context` |
| Sparse, ONNX et GGUF | Oui, appels sérialisés | `call_mutex` dans `sparse_text_embedding.h:171,348` ; les workers GGUF réutilisent leurs graphes ggml |
| Reranker, ONNX et GGUF | Oui, appels sérialisés | `call_mutex` dans `detail/reranker_impl.hpp:91,341` |
| Image | Oui, appels en parallèle | pas d'état temporaire par contexte, `OrtApi::Run` est thread-safe ; seuls les compteurs sont verrouillés (`image_embedding.h:116`) |
| Stats (`*_stats`, `*_stats_v2`) | Oui, n'attendent jamais une inférence | `stats_mutex` séparé (mêmes lignes) |
| `TextEmbeddingPool` | Oui | chaque worker possède un contexte (sérialisé), les appels de plusieurs threads s'enfilent par worker (`cpp/embedding_pool.hpp`) |
| Cache LRU (`lembed_cache_*`) | Oui | un seul mutex (`detail/embedding_cache_impl.hpp:142`). Utiliser `lembed_cache_get_copy()` : `lembed_cache_get()` rend un pointeur emprunté (déprécié) |
| Entrées du cache d'autotune | Oui dans un process | nom temporaire unique puis renommage atomique (`detail/autotune_cache.hpp:392`) ; plusieurs process : le dernier écrivain gagne |
| Ancien cache de tuning (`lembed_tune_cache_*`) | Oui dans un process | mutex global au process (`detail/autotune_cache_impl.hpp:91`) ; plusieurs process : le dernier écrivain gagne |
| Téléchargeur de modèles | Oui | fichier temporaire unique par téléchargement, publication atomique (`detail/downloader_impl.hpp:339`). Deux threads peuvent télécharger deux fois le même fichier |
| `lembed_last_error()` | Oui | tampon thread-local (`error.h:79`) : le lire sur le thread qui a reçu l'échec |
| `create` / `free` | Créer depuis n'importe quel thread ; libérer depuis un seul thread, sans appel en cours | |

Tests : `tests/test_concurrency.cpp` (aussi compilé sous ThreadSanitizer par le
workflow `tsan.yml`) et `tests/test_repeated_calls.cpp` (10 000 appels répétés,
croissance de RSS bornée, `-DLIBEMBEDDING_STABILITY_TESTS=ON`).

## Performance

Pour des performances optimales :

1. **Réutilisez le modèle** — créez-le une fois et embeds plusieurs batchs
2. **Ajustez `batch_size`** — augmentez jusqu'à saturation mémoire
3. **Essayez CoreML sur Apple** — `provider="coreml"` (mesurez : le gain dépend du modèle)
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
