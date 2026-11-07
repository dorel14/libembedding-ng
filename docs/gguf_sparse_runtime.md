# Runtime sparse GGUF

Version: 1.11.0
Status: implémenté, couvert par `tests/test_sparse_gguf.cpp` (96 assertions)
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
[`LE-9.6-results.md`](archive/LE-9.6-results.md) (archivé)
reste le résultat *correct*, et non une lacune à combler.

Le runtime est donc le nôtre : un graphe ggml sur le backend CPU déjà
vendorisé, qui lit les tenseurs par `gguf_init_from_file` et calcule dans
`ggml_mul_mat`. C'est la conclusion de
[`LE-9.4-validation.md` §5.3](archive/LE-9.4-validation.md) (archivé).

---

## 3. Le pipeline

```
texte
  → ids WordPiece            TokenizerWrapper (partagé avec le backend ONNX)
  → encodeur BERT            graphe ggml, post-LayerNorm
  → tête MLM                 graphe ggml, ggml_mul_mat contre token_embd
  → max-pool puis log(1 + relu)   sur l'hôte, tokens réels uniquement
  → retrait des ids spéciaux ids des métadonnées
  → top_k / min_weight / storage_format   sparse_prune_and_sort()
  → {indices, values}
```

Les textes d'un appel sont tokenisés une fois, triés par longueur décroissante,
puis découpés en lots bornés de deux façons : au plus `batch_size` textes, et au
plus 512 colonnes de tokens paddées (`batch × seq_len`, voir
`detail/gguf/sparse_batching.hpp`). Les résultats sont remis dans l'ordre
d'entrée. Le second plafond est celui qui borne le coût : le graphe et les
logits F32 `[vocab, batch × seq_len]` grandissent avec la largeur paddée, pas
avec le nombre de textes.
Sur `benchmarks/corpus.txt` (392 textes), batch 16 et 4 threads, une passe a
calculé 8 256 tokens avant le tri (6 830 réels + 1 426 paddés, soit 17,3 % de
padding) et 6 928 après (6 830 réels + 98 paddés, soit 1,4 %). Le travail
tokenisé baisse ainsi de 16,1 %. Les constructions de graphe par passe sont
passées de 20 à 9. Les médianes de latence observées sont passées de 27,76 s à
21,11 s, mais les répétitions avant/après ne sont pas appariées : ce chiffre de
débit reste indicatif et doit être confirmé par des mesures entrelacées.
Le changement de forme des graphes peut aussi modifier légèrement les arrondis
flottants. Sur ces 392 textes, avant/après tri à batch 16, les vecteurs complets
ont un écart de poids maximal de 0,00168 et un cosinus minimal de 0,99999984 ;
12 textes diffèrent par un terme de score quasi nul. Avec `top_k = 50`, deux
textes sélectionnent un terme différent à la frontière du classement (cosinus
minimal 0,99759). Le tri préserve l'ordre et les résultats usuels restent très
proches, mais l'identité exacte des termes n'est pas garantie pour ces quasi-
égalités numériques.

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

Le pooling hôte calcule d'abord le maximum du logit brut par terme, puis applique
`log(1 + relu(x))` une seule fois par terme. La transformation étant monotone,
c'est équivalent au max des logits transformés, avec beaucoup moins d'appels à
`log`. Le buffer des scores est réutilisé entre les documents d'un lot. Cette
optimisation réduit le post-traitement, mais ne supprime pas la projection sur le
vocabulaire complet ni le transfert de ses logits.
Le filtrage `min_weight` compacte également les vecteurs en place ; lorsque
`top_k` est appliqué, le résultat est déjà ordonné par poids et n'est pas retrié
une seconde fois en format dict.

### 3.3 Lots et masque

Tous les tokens d'un lot forment **une seule ligne**. Le masque d'attention
[T, T] est bloc-diagonal par document et à −∞ sur le padding ; c'est lui qui
sépare les documents, pas un second graphe. Le test synthétique
`test_sparse_gguf.cpp` vérifie l'égalité exacte entre batch 1 et les batchs 2 à 5
pour ses poids fixes ; cela ne garantit pas une égalité bit à bit pour un vrai
modèle quantifié, car la forme des GEMM change avec le batch et le padding.

Sur `splade-pp-en-v1-q8_0.gguf`, comparé au traitement individuel des 392 textes
de `benchmarks/corpus.txt` (4 threads, batch 16), les vecteurs complets ont un
cosinus minimal de 0,99976 sans tri et 0,99986 avec tri. Le cosinus minimal
calculé sur les seuls top 50 est respectivement 0,99807 et 0,99877. L'ensemble
des 50 termes diffère de la référence individuelle pour 52 textes sans tri,
contre 9 après tri par longueur. Le tri réduit donc la dérive mesurée, mais ne
garantit pas l'identité des termes retenus.

Exemple : pour le deuxième texte du corpus, le terme 2156 passe de 0,16757 en
batch 1 à 0,14872 en batch 16 trié, tandis que le terme 2235 passe de 0,16218
à 0,16151 ; la sélection top 50 change. Les scores de classement sur un jeu de
requêtes avec pertinences annotées n'ont pas été mesurés. Les applications qui
dépendent strictement des termes de tête doivent valider leur propre corpus et
leur configuration de batch.

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

Cette mesure est un résultat historique de la comparaison à lot 32. La mesure
plus récente consignée dans
[`LE-9.6-results.md`](archive/LE-9.6-results.md) utilise un lot 1 et
rapporte 8,5 docs/s (117,10 ms/doc) sur 91 textes ; les deux chiffres ne sont pas
directement comparables. Le débit demeure le point faible : contrairement à
ONNX Runtime, ggml exécute ici la projection complète sur le vocabulaire et le
runtime doit encore transférer ses logits pour le pooling.

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
- **Cache de graphe limité à une forme** : le graphe et son allocation sont
  réutilisés tant que longueur de séquence et taille de lot ne changent pas. Une
  forme différente remplace le cache et provoque une reconstruction ; les lots
  de longueurs variables peuvent donc encore payer ce coût.
- **Projection et logits complets** : `top_k` limite la sortie sparse, pas la
  projection MLM sur le vocabulaire. Le graphe produit toujours les logits de
  chaque token et les transfère à l'hôte avant le pooling.
- **Taille de lot** : le défaut GGUF est 16 textes par lot ; la valeur reste
  configurable. Sur le corpus de 392 textes mesuré sur un i7-1065G7, batch 2
  avec 4 threads a atteint 18,0 documents/s, contre 16,7 pour batch 16 et
  14,2 pour batch 64 dans un premier balayage. Une répétition alternée 16/2/2/16
  a mesuré respectivement 16,2 et 16,3 documents/s en moyenne : performances
  comparables dans la dispersion observée. Batch 2 utilisait environ 252 MiB
  de RSS pic et 5,3 % de padding, contre 334 MiB et 17,3 % pour batch 16 ;
  batch 16 réduit en revanche le nombre de lots et a montré une dispersion un
  peu plus faible. Il reste le défaut comme compromis, pas comme minimum mémoire
  ni comme optimum universel. Ces chiffres dépendent du processeur, du corpus et
  des longueurs de séquence.
- **Budget de tokens par lot** : 512 colonnes paddées au plus par graphe, quel
  que soit `batch_size`. Mesuré sur i7-1065G7, 4 threads, profileur
  `integration_sparse_gguf` :
  - 256 textes courts (`customers_sample.txt`, ~13 tokens), batch 256 : sans
    budget 12,6 documents/s et 1 642 MiB de RSS pic ; budget 128 / 256 / 512 /
    1024 : 25,5 / 26,1 / 26,4 / 24,0 documents/s, 276 / 325 / 415 / 596 MiB.
  - 96 dépêches Reuters (`reuters_clean.txt`, longues), batch 16 : sans budget
    1,51 documents/s et 2 208 MiB ; budget 512 / 1024 / 2048 : 1,83 / 1,61 /
    1,63 documents/s, 358 / 481 / 728 MiB.

  512 est le meilleur débit sur les deux profils et borne la mémoire.

  Le même budget (`kSparseMaxBatchTokens`, `detail/sparse_batching.hpp`)
  s'applique au chemin sparse ONNX, qui n'avait ni tri par longueur ni plafond :
  son batch par défaut de 256 réclamait 16 Go à ONNX Runtime sur des documents
  de 512 tokens (`bad allocation`). Balayage ONNX, batch 256 : 128 résumés
  SciFact 1,51 / 1,52 / 1,50 / 1,49 documents/s et 842 / 839 / 1 084 /
  1 478 MiB pour 256 / 512 / 1024 / 2048 tokens ; 256 textes courts 38,6 /
  41,3 / 37,9 / 33,2 documents/s, contre 22,6 avant correctif.
- **Workers** : une session GGUF exécute plusieurs graphes indépendants
  (`workers`) qui partagent une seule copie des poids, en lecture seule, et se
  répartissent les lots d'un appel. Un graphe seul cesse de profiter de threads
  supplémentaires (ggml synchronise tous les threads après chacune de ses ~460
  opérations) ; des graphes indépendants ne s'attendent pas. `num_threads` est
  le budget total (0 = tous les cœurs logiques) et `workers` le nombre de
  graphes (0 = `num_threads / 2`, soit un par cœur physique sur un processeur
  hyper-threadé). Un appel d'un seul lot, typiquement une requête, s'exécute sur
  un worker avec tout le budget. Mesuré sur i7-1065G7 (profileur, entrelacé) :
  256 textes courts 26,1 / 25,1 documents/s en 1 × 4, 23,5 en 1 × 8, 34,6 /
  31,4 en auto (4 × 2), RSS 279 → 320 MiB ; 64 résumés SciFact 0,92 / 0,93 en
  1 × 4, 0,98 en 1 × 8, 1,24 / 1,22 en auto, RSS 298 → 491 MiB. Les logits sont
  lus directement dans le tampon du graphe, sans copie par worker.

## 7. Profileur GGUF

Le profileur manuel `integration_sparse_gguf` mesure le runtime sur un GGUF
local, sans instrumentation dans la DLL ni modification de l'API publique. Il
est construit avec les tests :

```powershell
cmake --build build --config Release --target integration_sparse_gguf
build\tests\Release\integration_sparse_gguf.exe `
  --model "$HOME\.cache\libembedding\models--cstr-splade-pp-en-v1-GGUF\splade-pp-en-v1-q8_0.gguf" `
  --corpus benchmarks\corpus.txt `
  --batch 4 --threads 4 --warmups 2 --repetitions 30 --top-k 50
```

`--corpus` attend un document non vide par ligne ; sans cette option, le
profileur utilise douze textes intégrés. `--batch`, `--threads`, `--warmups`,
`--repetitions`, `--top-k` et `--max-length` règlent le lot, les threads ggml,
les passes non chronométrées, les passes mesurées, le pruning et la longueur
maximale. `--max-batch-tokens` remplace le budget de tokens par lot (0 le
désactive) ; `--workers` fixe le nombre de graphes (0 = auto) et `--threads` le
budget total de threads (0 = tous les cœurs logiques). `--mode per-text` fait
un appel à `embed()` par ligne au lieu d'un par passe sur le corpus : les
statistiques sont alors des latences de requête (une requête par ligne).

Pour chaque passe mesurée, le tableau rapporte médiane, moyenne, minimum,
p95 empirique nearest-rank, maximum et écart-type des étapes : tokenisation,
préparation des entrées, construction/allocation du graphe, masque,
téléversement, exécution ggml, récupération des logits, pooling, pruning/tri et
temps résiduel de l'enveloppe. Le résumé comprend aussi le débit documents/s,
les nnz moyens, RSS courant/pic, temps CPU processus / temps mural, compteurs de
graphes et padding, forme/nombre de nœuds ainsi que l'inventaire des opérateurs
ggml.

Les durées de scène agrègent tous les lots d'une passe ; elles ne sont pas des
latences par document ni des chronométrages par opérateur. `GraphExec` est le
temps de `ggml_backend_graph_compute` : il peut inclure le plan CPU et la
préparation/allocation de son espace de travail, et ne sépare pas le temps des
kernels individuels. L'inventaire final compte les opérations, sans leur
attribuer de temps. Utiliser au moins 30 répétitions pour une estimation de
queue plus informative ; avec peu de répétitions, le p95 peut être égal au
maximum. La première passe de chauffe crée notamment le graphe ; son coût est
rapporté séparément des passes chronométrées.

### Microbenchmark GEMM Q8_0

La cible `gguf_gemm_microbench` isole les multiplications matricielles ggml
avec des poids synthétiques quantifiés en Q8_0 et des activations F32. Elle
mesure les formes BERT `768x768`, `3072x768`, `768x3072` et la projection
vocabulaire `30522x768`, pour `N=32, 64, 80, 128, 256, 512`, à 1, 2 et 4
threads. Les pools de threads persistants sont mis en pause hors de leur mesure pour
éviter qu'un pool inactif ne consomme du CPU.

```powershell
cmake -S . -B build -DLIBEMBEDDING_BUILD_BENCHMARKS=ON
cmake --build build --config Release --target gguf_gemm_microbench
build\benchmarks\Release\gguf_gemm_microbench.exe `
  --warmups 2 --repetitions 5
```

`--shape-index 0..3` et `--n 32|64|80|128|256|512` limitent l'exécution à une
forme ou une largeur. `--threads 1|2|4` sélectionne le nombre de workers ;
`--cpumask <masque>` applique un masque d'affinité strict au pool ggml, par
exemple `--threads 4 --cpumask 0x55`. Ce dernier sert à comparer les placements
CPU dans le microbenchmark et ne change pas l'affinité du runtime libembedding.
Les GFLOPS utilisent `2*M*K*N` opérations par produit. Ce benchmark sert à
comparer les formes et l'échelle des threads ; les poids synthétiques, l'absence
d'attention et le graphe réduit ne reproduisent pas l'inférence complète ni ses
poids réels.