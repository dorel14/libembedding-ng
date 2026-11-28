---
title: Requêtes concurrentes
nav_order: 12
---

# Requêtes concurrentes : `TextEmbeddingRouter`

> **English:** [Concurrent requests](en/concurrency.html)

Un contexte natif n'exécute **qu'un appel à la fois** (`call_mutex`). Un serveur qui embarque des textes depuis plusieurs threads ou tâches `asyncio` a donc besoin de plusieurs sessions et d'un ordonnanceur. `TextEmbeddingRouter` fournit les deux : N sessions `TextEmbedding`, une file FIFO stricte bornée, un délai d'attente, des préréglages issus de mesures, et une API `asyncio`.

Disponible en Python uniquement (aucune API C), pour le **texte dense** (ONNX et GGUF), à partir de la 1.17.

## Quand l'utiliser

| Besoin | Outil |
|---|---|
| Plusieurs requêtes courtes (1 à 8 textes) en parallèle : serveur web, API | **`TextEmbeddingRouter`** |
| Un seul gros lot à répartir sur plusieurs cœurs | `TextEmbeddingPool` (inchangé) |
| Un seul appelant | `TextEmbedding` |

`TextEmbeddingPool` découpe *un* appel en morceaux. Le router, lui, garde chaque appel entier et en exécute plusieurs en même temps.

## Démarrage rapide

```python
from libembedding import TextEmbeddingRouter

with TextEmbeddingRouter("BAAI/bge-small-en-v1.5") as router:
    vectors = router.embed(["une requête"])      # depuis n'importe quel thread
    print(vectors.shape, router.sessions, router.threads)
```

Depuis `asyncio` :

```python
import asyncio
from libembedding import TextEmbeddingRouter

async def main():
    async with TextEmbeddingRouter("BAAI/bge-small-en-v1.5") as router:
        vectors = await router.aembed(["une requête"])

asyncio.run(main())
```

## Paramètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `model_name` | `BAAI/bge-small-en-v1.5` | Comme `TextEmbedding` |
| `policy` | `"latency"` | Préréglage (voir ci-dessous) : `"latency"`, `"balanced"`, `"indexing"` |
| `sessions` | `None` | Nombre de sessions explicite (désactive `policy`) |
| `threads` | `None` | Threads intra-op par session |
| `max_queue` | `64` | Attentes maximales ; la suivante lève `RouterBusyError` tout de suite. `0` = illimité |
| `acquire_timeout` | `None` | Attente maximale d'une session en secondes (`TimeoutError`) ; `None` = indéfiniment |
| `**session_kwargs` | | Transmis à chaque `TextEmbedding` (`batch_size`, `cache_size`, `quantization`, `backend`…) |

`embed(texts, *, timeout=None, **kwargs)` et `aembed(...)` acceptent `batch_size`, `dtype` et `normalized` comme `TextEmbedding.embed`. `timeout` remplace `acquire_timeout` pour cet appel et ne couvre que l'**attente d'une session**, pas l'appel natif.

## Préréglages

P est le nombre de **cœurs physiques** (`detect_hardware().physical_cores` ; à défaut, la moitié des cœurs logiques). Les sessions sont plafonnées à 8.

| Policy | ONNX (sessions × threads) | Usage |
|---|---|---|
| `latency` (défaut) | 1 × P | Serveur peu chargé, latence minimale |
| `balanced` | 2 × ⌊P/2⌋ | Serveur chargé |
| `indexing` | min(P, 8) × 1 | Indexation par lots, débit maximal |

Pour les modèles **GGUF**, chaque session utilise 1 thread et le nombre de sessions vaut 1, ⌈P/2⌉ ou min(P, 8) selon le préréglage.

### Ce qui a été mesuré

Banc T32 : i7-1065G7 (4 cœurs physiques), ONNX Runtime 1.31.0, bge-small et MiniLM-L6 FP32, 3 passes entrelacées. Références prises par rapport à 1×4 (une session, 4 threads) :

| Configuration | Débit à concurrence ≥ 4 | p95, batch 1, bge | RSS (bge) |
|---|---|---|---|
| 1×4 (`latency`) | référence | 9 ms | 243 Mo |
| 2×2 (`balanced`) | +25 à 27 % (batch 1) | 13 à 15 ms | 382 Mo |
| 4×1 (`indexing`) | +50 à 70 % (batch 1), +21 à 34 % (batch 8) | 24 ms | 658 Mo |

- 1×8 (plus de threads que de cœurs physiques) **perd 25 à 30 %** : c'est pourquoi les préréglages ne dépassent jamais P.
- Les sessions donnent des vecteurs identiques (écart 0).
- La mémoire croît à peu près linéairement avec le nombre de sessions (ORT ne partage pas les poids).

> **Limites de ces chiffres.** Une seule machine, 4 cœurs. Les valeurs pour P ≠ 4, l'ARM et les préréglages GGUF sont **supposées, pas mesurées**. Utilisez `sessions=` / `threads=` pour imposer votre configuration, et mesurez sur votre matériel.

### Budget de threads

Si `sessions × threads` dépasse P, le constructeur émet un `RuntimeWarning` (pas une erreur). Les préréglages ne le dépassent jamais.

## File d'attente et contre-pression

- **FIFO strict** : une session libérée est remise **directement** au plus ancien en attente, sans repasser par la liste libre. Un thread qui rend sa session ne peut pas la reprendre avant celui qui attendait (pas de famine).
- **Bornée** : avec `max_queue` requêtes déjà en attente, la suivante lève `RouterBusyError` immédiatement, sans bloquer.
- **Délai** : `TimeoutError` standard si aucune session n'est libre à temps. Aucune session n'est perdue dans ce cas.
- **Erreurs natives** (`LembedError`) : propagées telles quelles, la session est rendue.

`RouterBusyError` et `RouterClosedError` héritent de `LembedError`. Pour un serveur HTTP : `RouterBusyError` → 503, `TimeoutError` → 504.

## `asyncio`

`aembed` attend une session **dans la boucle** (aucun thread parqué, FIFO préservée) puis exécute l'appel natif sur un `ThreadPoolExecutor` **dédié au router** (jamais l'executor par défaut de la boucle, qui reste libre même router saturé). Compatible Python 3.9+.

### Annulation : ce qui est possible

| Moment | Comportement |
|---|---|
| En attente d'une session | La requête est retirée proprement de la file. Si une session lui était attribuée à cet instant précis, elle est rendue et passe au suivant. |
| **Pendant l'appel natif** | `CancelledError` est levée **immédiatement**, mais l'appel natif **ne peut pas être interrompu** : il se termine dans l'executor, son résultat est jeté, et la session est rendue seulement ensuite. |

Conséquence : une annulation ne libère pas la session tout de suite. **Bornez `batch_size` et la taille de `texts` côté serveur**, sinon un gros lot abandonné monopolise une session jusqu'à sa fin. Le délai `timeout` ne couvre pas non plus l'appel natif.

## Fermeture

`close(wait=True)` (ou `await aclose()`) :

1. refuse les nouvelles requêtes (`RouterClosedError`) ;
2. réveille les requêtes en attente avec `RouterClosedError` ;
3. attend la fin des appels en vol ;
4. ferme les sessions. Une session occupée n'est **jamais** fermée.

Dans une coroutine, utilisez `aclose()` (ou `async with`) : `close()` bloque le thread. `close(wait=False)` rend la main aussitôt ; le dernier appel terminé ferme les sessions restantes.

## Observabilité

`router.stats()` renvoie un `RouterStats` : `sessions`, `in_flight`, `queued`, `completed`, `rejected`, `timeouts`, `max_wait_ms`.

## Autres classes : `asyncio.to_thread`

Le router ne couvre que le texte dense. Pour `Reranker`, `SparseTextEmbedding` ou `MultiVectorEmbedding`, le contournement est :

```python
scores = await asyncio.to_thread(reranker.rerank, query, documents)
```

Limites : l'executor par défaut est partagé avec toute l'application ; pas de FIFO stricte, pas de contre-pression, pas de délai ; et un seul appel à la fois par instance (le contexte natif est verrouillé). Pour du parallélisme, créez plusieurs instances et gérez la file vous-même.

## Exemple : fournisseur d'embeddings pour un moteur de recherche (style Whoosh-NG)

```python
import asyncio
from fastapi import FastAPI, HTTPException
from libembedding import RouterBusyError, TextEmbeddingRouter

router = TextEmbeddingRouter(
    "BAAI/bge-small-en-v1.5",
    policy="balanced",
    max_queue=64,
    acquire_timeout=2.0,
    batch_size=32,          # borné : une requête annulée finit son appel natif
)

class LibembeddingProvider:
    """Protocole EmbeddingProvider : embed, embed_batch, dimension."""

    @property
    def dimension(self) -> int:
        return router.dim

    def embed(self, text: str):
        return router.embed([text])[0]

    def embed_batch(self, texts: list[str]):
        return router.embed(texts)

    async def aembed_batch(self, texts: list[str]):
        return await router.aembed(texts)

app = FastAPI()
provider = LibembeddingProvider()

@app.post("/embed")
async def embed(texts: list[str]):
    try:
        vectors = await provider.aembed_batch(texts[:64])
    except RouterBusyError:
        raise HTTPException(503, "busy")
    except TimeoutError:
        raise HTTPException(504, "timeout")
    return {"dim": router.dim, "vectors": vectors.tolist()}

@app.on_event("shutdown")
async def shutdown():
    await router.aclose()
```

Pour une indexation hors ligne, créez une seconde instance en `policy="indexing"` le temps du traitement : la mémoire double.

## Points d'attention

- **Cache LRU** : `cache_size` est **par session**. Le cache n'est pas partagé, le taux de succès est divisé par le nombre de sessions.
- **Mémoire** : `indexing` coûte environ 2,7 fois la mémoire de `latency` (bge-small, mesure T32).
- **Conteneurs et CPU hybrides** : la détection des cœurs ne tient pas compte des quotas cgroup. Imposez `sessions=` / `threads=`.
- **Contexte après une erreur native** : supposé sain (le verrou est libéré, aucun état partiel) ; non vérifié sur toutes les erreurs.
