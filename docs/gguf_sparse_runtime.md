# Runtime sparse GGUF

Version: 1.11.0
Status: implémenté, couvert par `tests/test_sparse_gguf.cpp` (67 assertions)
et `tests/test_sparse_gguf_vs_onnx.cpp` (intégration, réseau requis)

Décrit **comment un fichier sparse GGUF est exécuté**. La
[convention GGUF sparse](gguf_sparse_convention.md) décrit, elle, ce qu'un
fichier *contient*. Les deux sont normatives et distinctes : la convention décide
de l'acceptation d'un fichier, cette page du calcul.

---

## 1. Point d'entrée

```c
/* Un fichier local */
lembed_sparse_text_embedding_ctx_t* ctx = NULL;
lembed_sparse_options_t opts = lembed_sparse_options_default();
opts.top_k = 256;

lembed_status_t s = lembed_sparse_text_embedding_create_from_gguf_path(
    "splade-pp-en-v1-q8_0.gguf", &opts, &ctx);

/* Ou un dépôt HuggingFace, comme le backend dense */
s = lembed_sparse_text_embedding_create_from_gguf_model(
    "cstr/splade-pp-en-v1-GGUF", "splade-pp-en-v1-q8_0.gguf", &opts, &ctx);

lembed_sparse_embeddings_t out;
lembed_sparse_text_embedding_embed(ctx, texts, n, 8, NULL, &out);
lembed_sparse_embeddings_free(&out);
lembed_sparse_text_embedding_free(ctx);
```

`lembed_sparse_text_embedding_create_from_path()` **route sur l'extension** : un
chemin terminé par `.gguf` va directement au runtime GGUF. Un appelant qui ne
connaît qu'un chemin n'a donc pas à connaître le backend.

En Python :

```python
from libembedding import SparseTextEmbedding

# Chemin local
model = SparseTextEmbedding.from_gguf("splade-pp-en-v1-q8_0.gguf")

# Dépôt HuggingFace
model = SparseTextEmbedding.from_gguf_model(
    "cstr/splade-pp-en-v1-GGUF", "splade-pp-en-v1-q8_0.gguf"
)

vectors = model.embed(["a fast brown fox"])
```

Et `SparseTextEmbedding("chemin/vers/modele.gguf")` fonctionne aussi : le
routage est fait en C.

Le nommage est **agnostique du runtime** : pas de `_llama_` dans les
identifiants. llama.cpp n'intervient pas (voir §2).

---

## 2. Pourquoi llama.cpp n'est pas utilisé

C'est le point non négociable de cette implémentation, et il est vérifiable dans
le snapshot vendorisé.

llama.cpp v0.3.0 **ne peut pas charger** cette famille de fichiers. Il cherche
`blk.N.attn_q.weight`, `blk.N.ffn_up.weight`, `blk.N.layer_output_norm.weight`,
`blk.N.attn_output_norm.weight`, `token_embd_norm.weight` et
`token_types.weight` (`third_party/llama.cpp/src/llama-arch.cpp:407-453`), et
il lit les métadonnées `%s.attention.head_count`, `%s.feed_forward_length`,
`%s.embedding_length`, `%s.attention.layer_norm_epsilon`. Or les exports SPLADE
écrivent :

| llama.cpp attend | le fichier contient |
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

Un tenseur requis absent est fatal :
`create_tensor` lève `missing tensor 'blk.0.attn_q.weight'`
(`src/llama-model-loader.cpp:1103-1106`). Seuls `token_embd.weight` et
`position_embd.weight` correspondent.

Conséquence : `llama.cpp = n/a` dans
[`benchmarks/sparse/LE-9.6-results.md`](https://github.com/dorel14/libembedding-ng/blob/dev/benchmarks/sparse/LE-9.6-results.md)
reste le résultat *correct*, et non une lacune à combler.

Le runtime est donc le nôtre : un graphe ggml sur le backend CPU déjà
vendorisé, qui lit les tenseurs par `gguf_init_from_file` et calcule dans
`ggml_mul_mat`. C'est la conclusion de
[`LE-9.4-validation.md` §5.3](https://github.com/dorel14/libembedding-ng/blob/dev/benchmarks/sparse/LE-9.4-validation.md).

---

## 3. Le pipeline

```
texte
  → ids WordPiece            TokenizerWrapper (partagé avec le backend ONNX)
  → encodeur BERT            graphe ggml, post-LayerNorm
  → tête MLM                 graphe ggml, ggml_mul_mat contre token_embd
  → log(1 + relu) max-pool   sur l'hôte, tokens réels uniquement
  → retrait des ids spéciaux ids des métadonnées
  → top_k / min_weight / storage_format   sparse_prune_and_sort()
  → {indices, values}
```

### 3.1 Encodeur — l'ordre post-LN n'est pas négociable

BERT est **post-LayerNorm**, à l'inverse de presque tous les transformers que
llama.cpp exécute. Se tromper d'ordre produit un modèle qui charge, s'exécute et
renvoie des vecteurs plausibles :

- les projections QKV lisent l'entrée **normalisée** ;
- `ln1` s'applique **après** la résiduelle d'attention ;
- `ln2` s'applique **après** la résiduelle du FFN.

Le graphe est écrit dans cet ordre, calqué sur le constructeur BERT de
llama.cpp lui-même (`src/models/bert.cpp:114-225`), qui fait exactement cela.
L'implémentation est dans
`include/libembedding/detail/gguf/gguf_bert_graph.hpp`.

Deux détails vérifiés dans le ggml vendorisé :

- `ggml_gelu` **est** l'approximation tanh
  (`ggml-cpu/vec.h:968-970` : `0.5x(1+tanh(√(2/π)·x(1+0.044715x²)))`), ce
  contre quoi un export BERT est entraîné. `ggml_gelu_erf` est une autre
  fonction et ne doit pas la remplacer. `llama-graph.cpp:1806-1813` fait le même
  choix pour le FFN BERT.
- `ggml_norm` **soustrait la moyenne** (`ggml-cpu/ops.cpp:3721-3737`) : c'est un
  vrai LayerNorm, pas un RMSNorm. Il ne travaille qu'en F32, donc les échelles et
  biais de LayerNorm sont appliqués à part (`ggml_mul` puis `ggml_add`), ce qui
  tolère des poids de LayerNorm quantifiés.

### 3.2 Tête MLM — dans le graphe, jamais en boucle

```c
x    = ggml_add(ggml_mul_mat(mlm_transform_w, h), mlm_transform_b);
x    = ggml_gelu(x);
x    = layernorm(x, mlm_ln_w, mlm_ln_b, eps);
logits = ggml_add(ggml_mul_mat(token_embd_w, x), mlm_bias);   /* [vocab, tokens] */
```

La projection de décodage **est** la matrice d'embeddings de tokens : BERT lie
la tête MLM aux embeddings, donc il n'existe pas de matrice `vocab × hidden`
séparée dans le fichier, et il n'en est pas inventé une.

La tête est un `ggml_mul_mat`, pas une boucle scalaire. Une boucle serait
`O(tokens × hidden × vocab)` — inacceptable à V = 30522 — et l'opérateur
déquantifie le Q8_0 à notre place, ce qui règle au passage le piège des lectures
déquantisées : **aucun poids quantifié n'est jamais lu élément par élément**
(`ggml_backend_tensor_get` au-delà de `ggml_nbytes` fait avorter le processus).

### 3.3 Lots et masque

Tous les tokens d'un lot forment **une seule ligne**. Le masque d'attention
[T, T] est bloc-diagonal par document et à −∞ sur le padding ; c'est lui qui
sépare les documents, pas un second graphe. Un lot de deux et un lot de cinq
donnent donc le même vecteur — ce que `test_sparse_gguf.cpp` vérifie.

### 3.4 Tokenizer : partagé, pas dupliqué

Un GGUF ne contient qu'un vocabulaire (`tokenizer.ggml.tokens`), pas de
`tokenizer.json`. Le runtime alimente le **même** `TokenizerWrapper` que le
backend ONNX avec ces chaînes. Conséquence voulue : les deux backends
tokenisent de façon identique par construction, donc la similarité mesurée entre
eux mesure la quantification et le runtime, pas une dérive de tokenizer.

Le prix à payer est que les approximations connues du backend ONNX sont
partagées et non doublées : minuscule ASCII seulement, ponctuation découpée par
octet, pas de suppression d'accents. C'est un choix assumé, pas un oubli ; un
tokenizer HF complet exigerait des tables Unicode générées (~470 Ko) pour
diverger du backend de référence auquel nous comparons ce runtime.

---

## 4. Refus

Un fichier illisible, sans tête sparse, sans vocabulaire ou dont la tête ne
convient pas est **refusé avec une raison**, jamais accepté silencieusement.

| Situation | Code | Message |
|---|---|---|
| chemin nul / vide | `LEMBED_ERROR_INVALID_ARGUMENT` | — |
| fichier illisible ou non GGUF | `LEMBED_ERROR_UNSUPPORTED` | `cannot read GGUF file '...'` |
| pas de `general.architecture` | `LEMBED_ERROR_UNSUPPORTED` | `GGUF header has no general.architecture` |
| `<arch>.vocab_size` absent | `LEMBED_ERROR_UNSUPPORTED` | `GGUF has no <arch>.vocab_size to pool over` |
| epsilon de LayerNorm absent | `LEMBED_ERROR_UNSUPPORTED` | `GGUF header has no <arch>.attention.layer_norm_epsilon` |
| tenseur d'encodeur absent | `LEMBED_ERROR_UNSUPPORTED` | `missing encoder tensor '...'` (de `classify()`) |
| pas de `mlm_transform.weight` | `LEMBED_ERROR_UNSUPPORTED` | `GGUF carries no SPLADE head (mlm_transform.weight is absent)` |
| formule non SPLADE | `LEMBED_ERROR_UNSUPPORTED` | `GGUF sparse formula is not SPLADE` |
| tête SPLADE **et** `sparse_linear` | `LEMBED_ERROR_UNSUPPORTED` | `refusing to guess which one this file is meant for` |
| vocabulaire absent ou de mauvaise taille | `LEMBED_ERROR_UNSUPPORTED` | `GGUF has no tokenizer.ggml.tokens vocabulary array` |
| tokeniseur non BERT | `LEMBED_ERROR_UNSUPPORTED` | `GGUF declares tokenizer model '...', which this runtime does not implement` |
| erreur ggml | `LEMBED_ERROR_GGUF` | message de l'exception |

Le refus intervient **avant** toute allocation de poids : annoncer « pas de tête
sparse » à quelqu'un qui a un fichier dense ne doit pas coûter cent még-octets de
copie. `LEMBED_ERROR_GGUF` est un code *ajouté en fin* d'énumération, et non
`LEMBED_ERROR_LLAMA` : llama.cpp n'est pas en jeu.

`max_length = 0` signifie « aussi long que la table de positions le permet ». La
demande est **plafonnée** à la longueur de contexte du modèle, pas refusée : une
séquence plus longue indexerait hors de la table.

---

## 5. Mesures

Critère LE-9.6 : cosinus ≥ 0.90 entre le sparse GGUF et le sparse ONNX, sur les
mêmes textes.

```
sparse GGUF vs ONNX: min cos 0.9998, mean cos 0.9999 over 12 texts (bar 0.90)
```

Mesuré sur `cstr/splade-pp-en-v1-q8_0.gguf` (Q8_0, ~111 Mo) contre
`prithivida/Splade_PP_en_v1` (ONNX), 12 textes, `top_k = 0`. La marge est
d'un ordre de grandeur au-dessus de la barre : ce qui est mesuré, c'est le bruit
de quantification Q8_0, pas une divergence d'implémentation.

Les deux modèles dérivent du même export HuggingFace, ce qui rend la comparaison
légitime ; la licence du dépôt GGUF est Apache-2.0.

`tests/test_sparse_gguf.cpp` est hermétique : fixtures écrites dans un répertoire
temporaire, poids d'un générateur déterministe, F32 / F16 / Q8_0. Aucune réseau,
quelques millisecondes.

### Débit, et où il en est

`benchmarks/sparse/bench_sparse_backend_comparison.py`, 95 documents chronométrés,
lot 32, `top_k = 50` :

| Backend | docs/s | ms/doc | Charge (ms) | Pic RAM (MB) |
|---|---|---|---|---|
| Sparse ONNX | 9.7 | 102.80 | 1176 | 1368 |
| Sparse GGUF (ce runtime) | 2.8 | 357.85 | 109 | 905 |

Le runtime est donc **3,5× plus lent** qu'ONNX Runtime, pour **un tiers de moins de
RAM** et un chargement **dix fois plus rapide**. Le débit est le point faible, et il
est attendu : chaque appel reconstruit le graphe et réserve un buffer
d'activations neuf, là où ONNX Runtime garde un graphe fusionné et son pool de
mémoire. La correction est connue et bornée — voir « Pas de cache de lots » plus
bas — mais elle n'est pas faite dans P1.5c, dont le critère de sortie était la
fidélité.

---

## 6. Limites connues

- **Mémoire au chargement** : le blob GGUF est lu en RAM puis copié dans le
  buffer du backend, donc le pic vaut deux fois la taille du fichier (~222 Mo
  pour 111 Mo). Le régime établi est le fichier lui-même : le blob est libéré dès
  que le vocabulaire en est extrait.
- **Longueur de contexte** : plafonnée par `bert.max_position_embeddings` (512
  sur SPLADE PP v1). Au-delà, les positions n'existent pas.
- **Tokenisation** : voir §3.4. Sur du texte accentué, les deux backends
  divergent de HuggingFace de la même façon.
- **Pas de GPU** : backend CPU ggml uniquement. `provider` est rapporté
  `LEMBED_PROVIDER_CPU` et ignoré à la construction.
- **Pas de cache de lots** : le graphe est reconstruit à chaque appel. Sur un
  flux à lot unique et constant, la reconstruction se remarquerait ; ce n'est pas
  le profil d'usage visé.