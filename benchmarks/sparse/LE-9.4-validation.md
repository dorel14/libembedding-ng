# LE-9.4 — Faisabilité du backend sparse GGUF

> **Verdict : la route *llama.cpp* est BLOCKED. Le concept *SPLADE + GGUF +
> quantification* n'est pas mort — c'est le choix du runtime d'exécution qui
> était prématuré.**
>
> Ce document ne supprime donc pas P1.5 : il le transforme en spike de
> faisabilité (§ 6), avec deux routes d'implémentation distinctes selon ce que le
> spike établit.

Date de la validation : 2026-10-03
llama.cpp evalué : v0.3.0 (snapshot vendorisé dans `third_party/llama.cpp/`)

---

## 1. Ce qui change par rapport à la roadmap initiale

La roadmap v1 (LE-9.4) spécifiait :

```
llama_encode()
    ↓
récupère [batch, seq_len, vocab_size]
    ↓
sparse_postprocess_aggregate()
```

**Cette étape « récupère `[batch, seq_len, vocab_size]` » n'est pas garantie et
ne l'est pas en pratique.** llama.cpp possède bien `llama_get_logits()` /
`llama_get_logits_ith()`, de largeur `n_vocab` — mais un `BertForMaskedLM`
converti en GGUF ne dispose pas pour autant de la tête de prédiction MLM dans le
graphe d'inférence exécuté. C'est précisément la partie manquante.

Ce que la roadmap avait bien vu juste : la question à poser d'abord est *« le
moteur sait-il exécuter la tête ? »*, pas « le transport fonctionne-t-il ? ».

---

## 2. Méthode

Lecture directe des en-têtes GGUF (magic + `n_tensors` + `n_kv` + bloc de
métadonnées + descripteurs de tenseurs) par requêtes HTTP partielles
(`Range: bytes=0-…`), sans téléchargement complet. Les en-têtes ont ensuite été
comparés au chargeur de tenseurs BERT de llama.cpp v0.3.0, et non à la fiche
modèle : les chiffres publiés par les convertisseurs sont **recoupés**, pas
repris.

---

## 3. Ce que llama.cpp v0.3.0 ne fait pas

Trois vérifications indépendantes dans le snapshot vendorisé :

**(a) Le chargeur ne charge aucun tenseur `mlm_*`**

`third_party/llama.cpp/src/models/bert.cpp:23-74` —
`llama_model_bert::load_arch_tensors()` charge exactement :

```
tok_embd, type_embd, pos_embd, cls, cls_b, cls_out, cls_out_b,
tok_norm, tok_norm_b, par couche : qkv, wo(+b), attn_out_norm(+b),
ffn_up(+b), ffn_down(+b), layer_out_norm(+b)
```

Aucun `mlm_transform`, `mlm_ln`, `mlm_bias`.

**(b) Le graphe ne publie aucun tenseur de logits**

`third_party/llama.cpp/src/models/bert.cpp:227-232` — fin de
`llama_model_bert::graph::graph()` :

```cpp
cur = inpL;

cb(cur, "result_embd", -1);
res->t_embd = cur;

ggml_build_forward_expand(gf, cur);
```

`res->t_logits` — le tenseur que lit `llama_get_logits_ith()` — **n'est jamais
assigné** pour l'architecture BERT. Le graphe BERT se termine sur le pooling
(`llama-graph.cpp:3637-3706`, `LLAMA_POOLING_TYPE_NONE` → `cur = inp`), donc la
seule sortie exploitable est un vecteur dense `[n_embd]` ou `[n_embd, n_tokens]`.

**(c) Aucune trace du MLM dans tout llama.cpp**

```
grep -ri "mlm_head|has_mlm_head|output_head|build_mlm" third_party/llama.cpp
→ 0 occurrence
```

La clé `bert.has_mlm_head` n'est lue par aucun fichier du projet llama.cpp.

**Conséquence.** `LLAMA_POOLING_TYPE_NONE` donne bien les états cachés par token
(`res->t_embd`, forme `[768, n_tokens]`), mais la projection vers le vocabulaire
exige `mlm_transform` (768×768), `mlm_ln` et `mlm_bias` (30522) — **aucun n'est
chargé**, donc aucun n'est accessible depuis le contexte. `cls_out` existe
(`bert.cpp:38-39`) mais vaut `[n_embd, n_cls_out]` avec `output_dim = 768` :
tête de classification/reranking, pas projection 30522.

---

## 4. Candidats examinés

### 4.1 `cstr/splade-pp-en-v1-GGUF` — Apache-2.0

`splade-pp-en-v1-q8_0.gguf` (111 Mo, Q8_0), 202 tenseurs.

| Clé | Valeur |
|---|---|
| `general.architecture` | `bert` |
| `bert.vocab_size` | `30522` |
| `bert.hidden_size` | `768` |
| `bert.num_hidden_layers` | `12` |
| `bert.output_dim` | `768` |
| **`bert.has_mlm_head`** | **`1`** |

Descripteurs de tenseurs hors couches `enc.*` :

```
token_embd.weight      dims=[768, 30522]
position_embd.weight   dims=[768, 512]
...
mlm_transform.weight   dims=[768, 768]     <- projection dense
mlm_transform.bias     dims=[768]
mlm_ln.weight          dims=[768]          <- LayerNorm
mlm_ln.bias            dims=[768]
mlm_bias               dims=[30522]        <- biais vocabulaire
```

La matrice 30522 × 768 n'est pas un tenseur distinct : elle est **tied** à
`token_embd.weight`, comportement standard d'une tête MLM BERT.

**Verdict : la tête sparse est dans le fichier, llama.cpp la jette.** Ce n'est pas
un problème de conversion mais de moteur.

### 4.2 `cstr/splade-v3-GGUF` — CC-BY-NC-SA-4.0 ⚠️

`splade-v3-q8_0.gguf`, 202 tenseurs. Structure de tête **identique** à 4.1,
vérifiée indépendamment :

```
mlm_transform.weight   dims=[768, 768]
mlm_transform.bias     dims=[768]
mlm_ln.weight          dims=[768]
mlm_ln.bias            dims=[768]
mlm_bias               dims=[30522]
```

+ `bert.has_mlm_head = 1`, `bert.vocab_size = 30522`.

La fiche modèle publie une fidélité « sparse-cos vs HF fp32 » :

| Fichier | Quant | Taille | Sparse-cos |
|---|---|---|---|
| `splade-v3-iq4_xs.gguf` | IQ4_XS + imatrix | 68 Mo | 0.9971 |
| `splade-v3-q8_0.gguf` | Q8_0 | 111 Mo | 1.0000 |
| `splade-v3-f16.gguf` | F16 | 256 Mo | 1.0000 |

Ces chiffres sont ceux du convertisseur (CrispEmbed), pas une mesure
indépendante de libembedding. Ils indiquent un **fort potentiel** de la
quantification GGUF pour le sparse : ~4x plus petit que F16, fidélité 0.9971.

**Contrainte bloquante** : `splade-v3` est sous **CC-BY-NC-SA-4.0**, soit
*non commercial*. libembedding est **MIT**. Ce modèle **ne peut pas** entrer dans
le registre de libembedding. Le bon choix reste `splade-pp-en-v1` (Apache-2.0),
qui pointe sur le même modèle amont `prithivida/Splade_PP_en_v1` que l'entrée
ONNX existante.

### 4.3 `mradermacher/opensearch-neural-sparse-encoding-doc-v2-mini-GGUF`

`opensearch-neural-sparse-encoding-doc-v2-mini.Q8_0.gguf`, 101 tenseurs.

| Clé | Valeur |
|---|---|
| `general.architecture` | `bert` |
| `general.source.url` | `opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini` |
| `mradermacher.convert_type` | `hf` |
| `bert.block_count` | `6` |
| `bert.embedding_length` | `384` |

Descripteurs hors couches `enc.*` :

```
position_embd.weight      dims=[384, 512]
token_embd.weight         dims=[384, 30522]
token_embd_norm.weight    dims=[384]
token_embd_norm.bias      dims=[384]
token_types.weight        dims=[384, 2]
```

**Aucun tenseur `mlm_*`, aucune clé `has_mlm_head`.** La conversion HF → GGUF a
supprimé la tête de projection sparse. Non rattrapable par un patch moteur.

---

## 5. Ce que l'investigation établit

| Candidat | Licence | Tête sparse dans le GGUF | Exécutée par llama.cpp v0.3.0 | Verdict |
|---|---|---|---|---|
| `cstr/splade-pp-en-v1-GGUF` | Apache-2.0 | oui (`mlm_*`) | **non** | BLOCKED (moteur) |
| `cstr/splade-v3-GGUF` | CC-BY-NC-SA-4.0 | oui (`mlm_*`) | **non** | BLOCKED (moteur) + **licence non commerciale** |
| `mradermacher/opensearch-…-mini-GGUF` | Apache-2.0 | **non** | non | NON VIABLE (fichier) |

Trois enseignements, tous exploitables :

1. **Le format GGUF transporte parfaitement la tête sparse.** Vérifié dans deux
   dépôts indépendants : `mlm_transform` + `mlm_ln` + `mlm_bias` sont présents et
   intacts. Ce n'est pas un problème de sérialisation.
2. **Le blocage est dans le graphe d'inférence, pas dans le format.** La piste
   `bert.has_mlm_head` existe déjà dans les fichiers ; il manque seulement le
   chargement des tenseurs et une branche de graphe dans llama.cpp.
3. **Le runtime doit être choisi par les besoins, pas par l'héritage existant.**
   llama.cpp reste le bon choix pour dense et reranker (où il est validé, voir
   `benchmarks/reranking/LE-8.10-results.md`). Le sparse est un besoin
   différent et peut justifier un runtime différent.

---

## 6. Nouvelle structure : P1.5 en spike de faisabilité

P1.5 n'est pas supprimé. Il est découpé, et **aucun backend n'est exposé dans
l'API publique avant que le spike ait conclu.**

### P1.5a — Valider `BertForMaskedLM` dans llama.cpp

Objectif : déterminer si une contribution upstream rend llama.cpp capable
d'exécuter la tête MLM, ou si le blocage est structurel.

- Vérifier si `bert.has_mlm_head` et les tenseurs `mlm_*` sont supportés dans une
  version de llama.cpp plus récente que v0.3.0 (le snapshot est pinné ; un bump
  est à évaluer, pas à exclure).
- Si oui : rejouer les critères LE-9.6 (§ 7) et, seulement s'ils passent, spikes
  P1.5c.
- Si non : documenter et passer à P1.5b.

**Critère de sortie** : soit les critères § 7 sont atteignables via llama.cpp,
soit le blocage est confirmé comme structurel pour cette version.

### P1.5b — Évaluer CrispEmbed comme runtime sparse GGUF

`CrispStrobe/CrispEmbed` est un runtime d'embedding C++ **MIT**, sans dépendance,
qui tourne sur ggml et expose un mode sparse (`crispembed_encode_sparse`,
`--sparse`). C'est le runtime pour lequel les GGUF `cstr/splade-*` ont été
produits.

Points à auditer avant toute adoption :

| Question | Pourquoi elle décide |
|---|---|
| Licence MIT + compatibilité | ✅ MIT (vérifié). Pas de friction. |
| Intégrable en `third_party/` ou en DLL séparée ? | Détermine le coût de build. |
| Format `.sprs` séparé obligatoire ? | Si oui, le routage par extension `.gguf` seul ne suffit plus. |
| maintenance / activité du dépôt | 66 étoiles, dernière push 2026-10-03. À surveiller. |
| Fiabilité mesurée par nous | Les 0.9971 sont du convertisseur, pas de nous. |

Le point « licence » est le point le plus easy à valider : CrispEmbed est **MIT**,
contrairement à `splade-v3` qui est CC-BY-NC-SA-4.0. C'est ce qui rend la piste
crédible là où `splade-v3` est hors jeu.

**Critère de sortie** : décision documentée d'intégrer, de ne pas intégrer, ou
d'attendre.

### P1.5c — Backend sparse GGUF

À n'ouvrir **qu'après** P1.5a ou P1.5b ayant conclu qu'un runtime est
disponible. Le nommage doit alors refléter le runtime, pas supposer llama.cpp :

```
sparse_text_embedding_onnx_impl.hpp   <- existant
sparse_text_embedding_gguf_impl.hpp   <- nouveau, runtime agnostique
```

Le backend dense garde `text_embedding_llama_impl.hpp` : llama.cpp reste le
runtime dense/reranker. Le sparse devient une voie distincte, ce qui évite
d'imposer le MLM à tous les autres types.

---

## 7. Critères de validation, à rejouer quel que soit le runtime

| # | Critère | Seuil |
|---|---|---|
| 1 | Le tenseur de sortie complet `[batch, seq_len, vocab_size]` est exposé | obligatoire |
| 2 | Cosine similarity sortie sparse ONNX vs sparse GGUF | ≥ 0.90 |
| 3 | Sparsité (nnz / vocab) GGUF cohérente avec ONNX | ± 20 % |
| 4 | Recall@K du GGUF (K ∈ {20, 50, 100}) | ≥ 95 % de l'ONNX |
| 5 | **FIDélité de la quantification** : cosine sparse vs PyTorch de référence | ≥ 0.99 pour Q8_0 ; ≥ 0.99 pour IQ4_XS |

Le critère 5 est ajouté : c'est celui qui distingue une vraie quantification
d'une perte silencieuse. Les 0.9971 annoncés pour IQ4_XS par le convertisseur
servent de point de comparaison, pas de résultat acquis.

`benchmarks/sparse/bench_sparse_backend_comparison.py` implémente déjà les
critères 2 à 5 : il calcule le produit scalaire SPLADE exact sur le vocabulaire
pour le recall (pas une approximation), et il signale explicitement les colonnes
non mesurables plutôt que de produire un chiffre.

---

## 8. État de l'API publique aujourd'hui

 inchangé, et c'est délibéré :

- **Aucun** `lembed_sparse_text_embedding_create_from_gguf_path()`.
- **Aucun** routage `.gguf` sur `lembed_sparse_text_embedding_create_from_path()`.
- **Aucune** entrée sparse ajoutée à `detail/gguf_registry_impl.hpp` (la règle
  LE-9.7 — 2 modèles, pas 29 — reste satisfaite : on en a ajouté **0**).
- Le sparse est un backend **ONNX unique**, et `docs/` doit l'indiquer sans
  ambiguïté.

Si P1.5b conclut à l'intégration de CrispEmbed, la seule entrée de registre
candidate est **`SPLADE-PP-En-v1`** (`cstr/splade-pp-en-v1-GGUF`, Apache-2.0),
pour rester aligné sur le modèle sparse ONNX déjà présent. `splade-v3` est exclu
pour sa licence non commerciale. `opensearch-…-mini-GGUF` est écarté (§ 4.3).

### 8.1 Noms de modèle sparse : ce que le registry expose réellement

Vérifié par lecture du registry **et** par exécution du résolveur, parce que les
deux champs ne sont pas le même identifiant.

`include/libembedding/model_registry.h:302-311` — layout de
`lembed_model_info_t` = `{ model_name, model_code, model_file, … }` :

| Champ | Valeur | Rôle |
|---|---|---|
| `model_name` | `prithivida/Splade_PP_en_v1` | **nom canonique** — celui du modèle amont |
| `model_code` | `Qdrant/Splade_PP_en_v1` | **repo de téléchargement** (repack optimisé Qdrant) |

C'est la convention du projet pour toute entrée, pas une exception : CLIP suit la
même règle (`model_name` = `openai/clip-vit-base-patch32`, `model_code` =
`Qdrant/clip-ViT-B-32-vision`, `model_registry.h:319`).

Le dossier de cache `models--Qdrant-Splade_PP_en_v1` vient donc du **repo**, pas
du nom canonique. Ce n'est pas une indication que le registry pointe
« uniquement » sur Qdrant.

Résolution réelle (`resolve_sparse_model`, `python/src/libembedding/models.py:95`) :

| Chaîne passée | Index |
|---|---|
| `prithivida/Splade_PP_en_v1` | 0 |
| `Qdrant/Splade_PP_en_v1` | 0 |
| `Splade_PP_en_v1` | 0 |
| `splade-pp-en-v1` | 0 |
| `BAAI/bge-m3` | 1 |

Les deux noms fonctionnent : `lembed_find_sparse_model_by_code()` teste
`model_code`, puis `_matches_model()` retombe sur `model_name` **et** sur le
dernier segment du chemin.

**Conséquence pour les benchmarks et la doc** : garder
`prithivida/Splade_PP_en_v1` (nom canonique), qui est déjà ce qu'utilisent
`benchmarks/sparse/bench_sparse.py`, `python/tests/test_sparse_text_embedding.py`
et les docstrings d'autotune. Passer à `Qdrant/...` fonctionnerait, mais
introduirait le repo de téléchargement là où le reste du projet utilise le nom
canonique.

### 8.2 Défaut orthographié dans l'API Python (constat, hors périmètre)

Trois valeurs par défaut ne correspondent pas au nom canonique, et ne résolvent
que par accident :

| Emplacement | Valeur | Problème |
|---|---|---|
| `sparse_text_embedding.py:47` (`SparseTextEmbedding.__init__`) | `prithvida/SPLADE_PP_en_v1` | `prithvida` : le `i` de `prithivida` manque |
| `sparse_text_embedding.py:290` (`sparse_best_config`) | `prithivida/SPLADE_PP_en_v1` | casse différente du canonical |
| `sparse_text_embedding.py:47` | — | la casse est tolérée, l'org ne l'est pas |

Les trois résolvent vers l'index 0 **uniquement** parce que `_matches_model`
compare en dernier recours le seul segment final du chemin
(`Splade_PP_en_v1`), ce qui masque l'org mal orthographiée. LeSparse défaut de
`SparseTextEmbedding()` fonctionne donc, mais par filet de sécurité : resserrer le
matcher, ou ajouter une entrée de même basename depuis une autre org, transformerait
le défaut en panne.

À corriger dans un commit dédié (3 lignes + une assertion de test), hors du
périmètre de ce document.

---

## 9. Reproduction

Le contrôle des en-têtes GGUF ne demande aucun outil du projet :

```bash
python - <<'PY'
import urllib.request, struct
url = ("https://huggingface.co/cstr/splade-pp-en-v1-GGUF/"
       "resolve/main/splade-pp-en-v1-q8_0.gguf")
req = urllib.request.Request(url, headers={"Range": "bytes=0-8388607"})
blob = urllib.request.urlopen(req).read()
n_tensors, = struct.unpack_from("<Q", blob, 8)
n_kv,      = struct.unpack_from("<Q", blob, 16)
print("n_tensors", n_tensors, "n_kv", n_kv)
PY
```

Côté source llama.cpp :

```bash
rg -i "mlm_head|has_mlm_head|build_mlm|output_head" third_party/llama.cpp   # 0 résultat
rg -n "res->t_logits" third_party/llama.cpp/src/models/bert.cpp            # 0 résultat
```