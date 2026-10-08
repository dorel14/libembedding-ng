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

**(d) Correction ajoutée le 2026-10-04 — le chargeur ne trouve *aucun* tenseur
d'encodeur, pas seulement pas de tête**

Les vérifications (a) à (c) portent sur la tête. En implémentant le runtime
(P1.5c), un constat plus fort est apparu : **le fichier ne se charge pas du
tout**, parce que les noms de tenseurs et les clés de métadonnées ne sont pas
ceux que llama.cpp attend. Ce n'est plus « la tête est perdue », c'est « le
modèle est absent ».

`src/llama-arch.cpp:407-453` définit les noms attendus ; `create_tensor` lève
`missing tensor 'blk.N.attn_q.weight'` pour un tenseur requis absent
(`src/llama-model-loader.cpp:1103-1106`), et les clés hparam lues avec
`required = true` lèvent de la même façon (`:1225-1233`).

| llama.cpp attend | `cstr/splade-pp-en-v1-q8_0.gguf` contient |
|---|---|
| `blk.N.attn_q.weight` | `enc.N.attn.q.weight` |
| `blk.N.ffn_up.weight` | `enc.N.ffn.fc1.weight` |
| `blk.N.layer_output_norm.weight` | `enc.N.ln2.weight` |
| `blk.N.attn_output_norm.weight` | `enc.N.ln1.weight` |
| `token_embd_norm.weight` | `embd_ln.weight` |
| `token_types.weight` | `token_type_embd.weight` |
| `%s.embedding_length` | `bert.hidden_size` |
| `%s.block_count` | `bert.num_hidden_layers` |
| `%s.attention.head_count` | `bert.num_attention_heads` |
| `%s.feed_forward_length` | `bert.intermediate_size` |
| `%s.attention.layer_norm_epsilon` | `bert.layer_norm_eps` |
| `%s.context_length` | `bert.max_position_embeddings` |

Seuls `token_embd.weight` et `position_embd.weight` correspondent. Les
conventions de nommage divergent, et il n'existe aucun mécanisme de repli par
plusieurs préfixes côté llama.cpp (le seul repli multi-noms de
`create_tensor_qkv` couvre `attn_qkv` contre `attn_q`/`attn_k`/`attn_v`, tous
sous `blk.N.`).

Cela ne rend pas (a)-(c) fausses : la tête est bien absente du chargeur. Cela les
contraint : la route llama.cpp n'était pas « partially blocked », elle était
inutilisable, et le bloqueur est plus large que le MLM.

**Conséquence.** `LLAMA_POOLING_TYPE_NONE` donnerait bien les états cachés par token
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

**Verdict : la tête sparse est dans le fichier, llama.cpp ne sait pas le lire.**
Ce n'est pas un problème de conversion mais de moteur — et §3(d) précise que le
moteur ne sait pas lire *le fichier* : ni la tête, ni l'encodeur.

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

## 6. Structure : P1.5 en phases

> ⚠️ **Section caduque, conservée pour la traçabilité.** Le découpage ci-dessous
> (a = valider llama.cpp, b = évaluer CrispEmbed) a été remplacé le 2026-10-03
> après la lecture d'architecture de CrispEmbed : **llama.cpp n'est plus une
> piste**, et CrispEmbed n'est pas une dépendance. Le découpage en vigueur est
> **P1.5a → P1.5b → P1.5c → P1.5d → P1.5e**, décrit au § 9 et dans
> `roadmap-2.md`.

P1.5 n'est pas supprimé. Il est découpé, et **aucun backend n'est exposé dans
l'API publique avant que les phases ait conclu.**

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
| Fiabilité mesurée par nous ? | Les 0.9971 sont du convertisseur, pas de nous. |

Le point « licence » est le point le plus easy à valider : CrispEmbed est **MIT**,
contrairement à `splade-v3` qui est CC-BY-NC-SA-4.0. C'est ce qui rend la piste
crédible là où `splade-v3` est hors jeu.

**Critère de sortie** : décision documentée d'intégrer, de ne pas intégrer, ou
d'attendre.

### P1.5c — Backend sparse GGUF

À n'ouvrir **qu'après P1.5a ou P1.5b ayant conclu qu'un runtime est
disponible**. Le nommage doit alors refléter le runtime, pas supposer llama.cpp :

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

## 9. Architecture de référence : lecture de CrispEmbed

CrispEmbed (`CrispStrobe/CrispEmbed`, C++, MIT, ~66 étoiles, poussé le
2026-10-03) est aujourd'hui l'implémentation open source la plus avancée de
sparse/ColBERT en GGUF. Lecture du dépôt effectuée le 2026-10-03 (tree SHA
`28d5a61c`). **Objectif : en tirer des choix de conception, pas en devenir une
dépendance.**

### 9.1 Quatre affirmations de la roadmap initiale, corrigées

| Affirmation | Verdict | Source |
|---|---|---|
| Les GGUF sparse exigent un fichier `.sprs` séparé | ❌ **inexistant** — 0 chemin sur 2054 blobs, 0 occurrence dans le source, 0 sur HF | `git/trees?recursive=1` |
| Le dispatch se fait via des clés GGUF `has_sparse` / `has_colbert` | ⚠️ **mal documenté** — ce sont des `bool` du struct runtime, déduits de la **présence des tenseurs** ; rien n'est écrit dans le fichier | `src/crispembed.cpp:854-887` |
| IQ4_XS atteint 0.9971 de fidélité | ⚠️ chiffre de la fiche HF `cstr`, **introuvable dans CrispEmbed** ; le dépôt mesure 0.963074 (CPU) / 0.965709 (CUDA) sur `fireredpunc` | `PERFORMANCE.md:288-289` |
| llama.cpp est le runtime sparse | ❌ CrispEmbed ne dépend **ni de llama.cpp ni d'onnxruntime** — uniquement de son **fork** de ggml | `.gitmodules` |

Le point « `.sprs` » mérite d'être noté : c'est un mécanisme réel, mais il
appartient à **llama.cpp**, pas à CrispEmbed. `models/add-st-dense-to-gguf.py:6-16`
documente que les exports SPM de llama.cpp (`gemma-embedding`) omettent la tête
Dense post-pooling, et que « sans le fichier
`--sentence-transformers-dense-modules` l'embedding est le mean-pool du backbone,
**orthogonal à la vraie sortie du modèle** (cos ≈ 0) ». L'erreur est totale, pas
une dérive — c'est un problème de **convertisseur**, pas de runtime.

### 9.2 Ce qui est directement transposable

**a) Le dispatch par présence de tenseur.** C'est l'idée la plus réutilisable, et
elle est élégante : pas d'enum, pas de métadonnée, pas de convention de nom de
fichier — la capacité est inférée des tenseurs présents.

```cpp
// src/crispembed.cpp:877-887
m.mlm_transform_w     = get("mlm_transform.weight");
m.has_mlm_head       = m.mlm_transform_w != nullptr;
m.has_sparse         = m.sparse_linear_w != nullptr || m.has_mlm_head;
m.has_colbert        = m.colbert_linear_w != nullptr;
m.is_reranker        = m.classifier_dense_w && m.classifier_out_w;
```

Pour libembedding, l'équivalent est déjà implicite : `lembed_resolve_sparse_model`
+ la présence de `mlm_*`. C'est une **convention à documenter**, pas une table à
maintenir.

**b) Les clés d'hparams dérivées de l'architecture** (`src/core/hparam_keys.h`) :
lire `general.architecture` et dériver le préfixe `<arch>.<field>`, avec une
résolution *qui signale si la clé a réellement été trouvée*. Le commentaire du
dépôt énonce exactement le piège qu'il faut éviter :

> « une architecture dont les noms de tenseurs se résolvent, **un mauvais défaut
> produit silencieusement un embedding poubelle avec un code de sortie 0**. »

C'est la leçon à retenir pour `libembedding` : **un défaut silencieux est pire
qu'un échec**. Voir P1.5b.

**c) La formule SPLADE**, avec le masquage d'attention (§9.3).

**d) La piste ColBERT** : sortie plate `[n_tokens, dim]` F32, chaque vecteur de
token normalisé en L2, score MaxSim exporté. `bert.colbert_dim` (défaut 128)
dans les métadonnées. C'est le prolongement naturel si le sparse GGUF aboutit.

### 9.3 SPLADE : la formule, et les deux pièges à ne pas copier

```cpp
// src/crispembed.cpp:3109-3116
for (int v = 0; v < V; v++) {
    float logit = mlm_b[v];
    for (int j = 0; j < H; j++) logit += emb_w[v * H + j] * h[j];
    if (logit > 0.0f) {                              // ReLU
        float sv = logf(1.0f + logit);               // log(1 + ReLU(x))
        if (sv > max_logits[v]) max_logits[v] = sv; // max-pool sur les tokens
    }
}
```

Le transformer MLM est `GELU(W·h + b)` avec un **GELU tanh approché codé en
dur** (`:3093`), puis LayerNorm (`:3097-3106`), puis projection vocabulary —
le poids du décodeur est **tied à `token_embd.weight`**.

**Piège 1 — filtre de jetons spéciaux codé en dur :**

```cpp
// src/crispembed.cpp:3122-3127
if (max_logits[v] > 0.0f && v != 0 && v != 101 && v != 102) {   // [PAD] [CLS] [SEP] BERT
```

Le chargeur **lit** les bons ids (`tokenizer.ggml.cls_token_id`,
`separator_token_id`, `pad_token_id`) mais ce filtre les ignore. Sur un vocabulaire
XLM-R (250k), les spéciaux sont `<s>`=0, `</s>`=2, `<pad>`=1 : **1 et 2 fuient
dans le vecteur sparse**. À ne pas copier — utiliser les ids des métadonnées.

**Piège 2 — coût de la projection vocabulary.** La boucle `O(T·H·V)` en scalaire
CPU est correcte uniquement parce que `H·V` est petit pour un SPLADE de classe
MiniLM. Sur un MLM head XLM-R (`V` = 250k), c'est inacceptable. libembedding doit
passer par un **matmul ggml**, pas une boucle scalaire.

**Deux chemins sparse distincts** — à ne pas confondre :

| Chemin | Condition | Formule |
|---|---|---|
| SPLADE (MLM head) | `has_mlm_head` | max-pool puis `log(1+x)` |
| BGE-M3, `out_dim == 1` | un scalaire par token | scatter sur `input_ids`, max par id, **poids brut, sans `log(1+x)`** |
| BGE-M3, `out_dim == V` | projection vocab complète | max-pool puis `log(1+x)` |

**Aucun top-k, aucun seuil** en sortie : la taille n'est bornée que par `V`. Pour
XLM-R, jusqu'à ~250k entrées par document. Le `top_k` / `min_weight` déjà livré
par P0.1 côté ONNX est donc **indispensable** côté GGUF, pas optionnel.

### 9.4 Quantification : la leçon sur les lectures CPU

CrispEmbed ne déquantifie jamais dans le graphe — ggml s'en charge. Le danger est
ailleurs, et le dépôt le documente trois fois :

```cpp
// src/crispembed.cpp:3068-3071
// mlm_transform_w and token_embd are 2-D weight matrices that the quantizer may
// store as Q8_0/F16/Q4_K, so read them via to_f32 (dequant-safe) — a raw
// n*sizeof(float) get would overrun ggml_nbytes and abort.
```

**À retenir pour libembedding** : toute tête qui doit descendre en scalaire CPU
(MLM, reranker, ColBERT) exige une lecture **déquant-sûre**. Une lecture
`n * sizeof(float)` brute sur un tenseur Q8_0 déborde `ggml_nbytes` et **abort le
processus**. Notre `detail/` n'a pas encore d'équivalent de `core_cpu::to_f32()`.

Deux autres points :

- le pré-merge QKV n'est appliqué qu'aux tenseurs **F32** (`:913`), donc un modèle
  Q8_0 perd silencieusement cette optimisation — à savoir avant de modéliser les
  perfs ;
- la fidélité est validée **hors ligne** (CI + `PERFORMANCE.md`), jamais au
  runtime. `PERFORMANCE.md:304-308` avertit que la cos_min de `q4_k` dépend du
  matériel (0.957795 chez eux, 0.935078 sur un VPS, **octet pour octet le même
  fichier**) : ne jamais citer un chiffre Q4_K comme une propriété de l'artefact.

### 9.5 Structure et coût de l'intégration

| Fichier | Lignes |
|---|---|
| `src/crispembed.cpp` | 6230 |
| `src/crispembed.h` (ABI C publique) | 1149 |
| `src/core/gguf_loader.cpp` | 732 |
| `examples/cli/main.cpp` | 2693 |

Dépendances : **fork `CrispStrobe/ggml` @ `sync/upstream-v0.17`**, pas
`ggml-org/ggml`. C'est le principal frein au vendor : le dépôt documente un
workaround de régression ggml v0.10.0 (Metal « residency sets », désactivés par
défaut via un kill-switch ggml). Brancher du ggml amont non patché n'est pas
sûr.

**C'est l'argument décisif pour un runtime maison** : libembedding vendorise déjà
`third_party/llama.cpp` **avec son ggml**, snapshot pinné et élagué selon une
politique de version explicite (AGENTS.md § 10). Un runtime sparse maison s'appuie
sur **notre** ggml déjà présent, avec une seule politique de mise à jour. Adopter
le fork CrispEmbed nous ferait hériter d'une seconde chaîne ggml.

Le dépôt fait ~230 Mo, ~60 moteurs sans rapport avec l'embedding (OCR, SR, NER,
KIE, LID…), un `CMakeLists.txt` de 52 Ko et ~470 Ko de tables Unicode générées.
Il faudrait en extraire ~10 % — ce qui confirme « source d'idées » plutôt que
« dépendance ».

### 9.6 Ce que nous ne copions pas

Trois défauts relevés dans le source, à corriger chez nous dès le départ :

1. filtre des spéciaux codé en dur `{0, 101, 102}` (§9.3, piège 1) ;
2. `n_vocab` par défaut à 30522 avec le mode strict **désactivé** par défaut
   (`CRISPEMBED_STRICT_HPARAMS=0`) — un défaut silencieusement faux sur une autre
   architecture ;
3. chemin sparse `out_dim == 1` sans `log(1+x)` : à documenter explicitement
   plutôt qu'à copier sans le savoir.

**État au 2026-10-04 (P1.5c livré).** Les trois ont été évités :

1. les ids de tokens spéciaux viennent **uniquement** des métadonnées
   (`tokenizer.ggml.{pad,bos,eos,cls,separator}_token_id`), sans liste codée en
   dur — voir `detail/gguf/gguf_weights.hpp`, `read_special_ids()` ;
2. aucune dimension n'est devinée : `n_embd`, `n_head`, `n_layer`, `n_ff`,
   `n_vocab`, `n_pos` et l'epsilon sont tous **exigés**, avec les synonymes de la
   convention comme second essai ;
3. seul SPLADE est implémenté ; `sparse_linear` est refusé avec un message
   plutôt que traité par une formule approximative.

Point notable : la tête est calculée par `ggml_mul_mat`, pas par la boucle
scalaire de CrispEmbed. La déquantisation Q8_0 y est gratuite, ce qui **rend le
piège des lectures déquant-sûres sans objet** — aucun poids quantifié n'est lu
élément par élément dans le runtime livré.

---

## 10. Reproduction

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

CrispEmbed (lecture du dépôt) :

```bash
gh api repos/CrispStrobe/CrispEmbed/git/trees/main?recursive=1 \
  --jq '.tree[].path' | rg -i 'sprs'          # 0 résultat
gh api repos/CrispStrobe/CrispEmbed/contents/.gitmodules --jq .content | base64 -d
```