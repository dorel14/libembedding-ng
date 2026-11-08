/*
 * libembedding - detail/gguf/gguf_weights.hpp
 * Loads a SPLADE GGUF into a ggml backend buffer, ready to be used by a graph.
 *
 * Why this exists instead of llama.cpp's model loader
 * ----------------------------------------------------
 * llama.cpp cannot read this family of files at all (checked at v0.3.0, still
 * true of the vendored v0.6.0). It looks for
 * `blk.N.attn_q.weight`, `blk.N.ffn_up.weight`, `blk.N.layer_output_norm.weight`,
 * `token_embd_norm.weight` and `token_types.weight`
 * (src/llama-arch.cpp, LLM_TENSOR_NAMES), plus `%s.attention.head_count`,
 * `%s.feed_forward_length` and `%s.embedding_length`; the SPLADE exports carry
 * `enc.N.attn.q.weight`, `enc.N.ffn.fc1.weight`, `enc.N.ln1/ln2.weight`,
 * `embd_ln.weight`, `token_type_embd.weight` and `bert.hidden_size` /
 * `bert.num_attention_heads` / `bert.intermediate_size`. A missing required
 * tensor is fatal: `create_tensor` throws
 * "missing tensor 'blk.0.attn_q.weight'" (src/llama-model-loader.cpp, create_tensor).
 * Only `token_embd.weight` and `position_embd.weight` happen to match.
 *
 * So the weights are read here, through gguf's own reader, into a buffer this
 * library owns. Everything the graph does with them stays in ggml: the
 * dequantisation of Q8_0 weights happens inside `ggml_mul_mat`, which is also
 * why nothing in this runtime ever reads a quantised weight element by element
 * (`ggml_backend_tensor_get` past `ggml_nbytes` aborts the process).
 *
 * The GGUF blob is copied once into our buffer and the file is closed: with
 * `no_alloc = false` the blob is held in RAM, so peak usage is twice the file
 * size and steady-state usage is once.
 *
 * Auteur: David Orel
 * Version: 1.12.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_WEIGHTS_HPP
#define LIBEMBEDDING_GGUF_WEIGHTS_HPP

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <gguf.h>

#include <cstdio>
#include <string>
#include <vector>

#include "../ggml_cpu_backend.hpp"
#include "gguf_probe.hpp"

namespace lembed {
namespace gguf {

/* Upper bounds on the header's hyper-parameters.
 *
 * A GGUF is untrusted input: every dimension below sizes an allocation or a
 * loop before a single weight is read, so `block_count = 2^31 - 1` would build
 * thirty billion tensor names before failing. The bounds are far above any
 * BERT-family encoder (BERT-large: 1024 wide, 16 heads, 24 layers, 4096 FFN;
 * XLM-R: 250 002 tokens; long-context BERTs: 8192 positions) and only exist to
 * turn an absurd header into an immediate refusal. */
constexpr int kMaxEmbd      = 16384;
constexpr int kMaxHeads     = 256;
constexpr int kMaxLayers    = 256;
constexpr int kMaxFf        = 65536;
constexpr int kMaxVocab     = 1 << 21;
constexpr int kMaxPositions = 1 << 17;

/* Tensors per encoder layer, as listed by Weights::collect_required(). A file
 * with fewer tensors than its block_count implies is refused before any name
 * is built. Only the layers are counted: a file missing the head or the
 * embeddings is refused later, by the name of the tensor it lacks. */
constexpr int kTensorsPerLayer = 16;

/* Architecture hyper-parameters, resolved once at load time.
 *
 * None of these has a default. A guessed width or head count produces a model
 * that runs and returns nonsense, which is strictly worse than a refusal. */
struct SpladeHparams {
    int   n_embd   = 0;
    int   n_head   = 0;
    int   n_layer  = 0;
    int   n_ff     = 0;
    int   n_vocab  = 0;
    int   n_pos    = 0;   /* size of the position embedding table */
    int   n_type   = 0;   /* size of the token-type embedding table */
    float eps      = 0.0f;
    std::string arch;     /* general.architecture, "bert" in practice */
};

/* Per-layer weights of the BERT encoder. Held as pointers rather than looked up
 * by name at graph-build time: 12 layers means 168 name lookups per document
 * otherwise, and the layer count only changes when a model is loaded. */
struct LayerWeights {
    ggml_tensor* ln1_w = nullptr;
    ggml_tensor* ln1_b = nullptr;
    ggml_tensor* q_w   = nullptr;
    ggml_tensor* q_b   = nullptr;
    ggml_tensor* k_w   = nullptr;
    ggml_tensor* k_b   = nullptr;
    ggml_tensor* v_w   = nullptr;
    ggml_tensor* v_b   = nullptr;
    ggml_tensor* o_w   = nullptr;
    ggml_tensor* o_b   = nullptr;
    ggml_tensor* ln2_w = nullptr;
    ggml_tensor* ln2_b = nullptr;
    ggml_tensor* fc1_w = nullptr;
    ggml_tensor* fc1_b = nullptr;
    ggml_tensor* fc2_w = nullptr;
    ggml_tensor* fc2_b = nullptr;
};

/* Embedding-side weights plus the MLM/SPLADE head.
 *
 * `tok_embd` doubles as the decoder projection: BERT ties the MLM head to the
 * token embeddings, so a 30522x768 vocabulary matrix is stored once and reused.
 * That is also why the head costs almost nothing on disk. */
struct HeadWeights {
    ggml_tensor* tok_embd = nullptr;
    ggml_tensor* pos_embd = nullptr;
    ggml_tensor* type_embd = nullptr;
    ggml_tensor* embd_ln_w = nullptr;
    ggml_tensor* embd_ln_b = nullptr;

    ggml_tensor* mlm_transform_w = nullptr;
    ggml_tensor* mlm_transform_b = nullptr;
    ggml_tensor* mlm_ln_w = nullptr;
    ggml_tensor* mlm_ln_b = nullptr;
    ggml_tensor* mlm_bias = nullptr;
};

/* Owns the model weights and the parsed header -- and no backend.
 *
 * The weights live in a plain CPU buffer that nothing writes after loading,
 * so one Weights can be shared, read-only, by several sessions running at the
 * same time; each session brings its own backend and its own thread count.
 * That is what lets the sparse runtime run N workers for the memory of one
 * model.
 *
 * Non-copyable and non-movable: it holds raw ggml and gguf pointers, and a
 * silent double free in a destructor is not a trade worth making for a
 * convenience nobody needs. Construct it, load into it, keep it. */
class Weights {
public:
    Weights() = default;

    ~Weights() {
        if (ctx_) ggml_free(ctx_);
        /* ctx_ is freed before buf_ on purpose: the tensors point into buf_. */
        if (buf_) ggml_backend_buffer_free(buf_);
    }

    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;
    Weights(Weights&&) = delete;
    Weights& operator=(Weights&&) = delete;

    /* Reads the header and resolves the hyper-parameters. Nothing is allocated and
     * no weight is touched, so the caller can decide whether this file is one it
     * wants before paying for it.
     *
     * On failure err holds the reason, naming the missing key; nothing is left
     * half-open. */
    bool open(const char* path, std::string& err) {
        if (!path || !*path) {
            err = "empty GGUF path";
            return false;
        }

        ggml_context* blob_ctx = nullptr;
        gguf_init_params params{};
        params.no_alloc = false; /* the blob has to be in RAM to be copied out */
        params.ctx = &blob_ctx;

        gguf_context* raw = gguf_init_from_file(path, params);
        if (!raw) {
            err = std::string("cannot read GGUF file '") + path + "'";
            return false;
        }
        /* raw is non-null, so adopt() cannot fail; asking it would only produce
         * a branch that has already lost the pointer. */
        probe_.adopt(raw);
        blob_ctx_ = blob_ctx;

        if (!read_hparams(err)) return false;

        read_special_ids();
        return true;
    }

    /* Copies every required tensor into the backend buffer. Call only after
     * open(), and only if the file is one this runtime will accept: reporting
     * "no sparse head" costs nothing, reporting it after allocating a hundred
     * megabytes of weights costs a lot. */
    bool load_tensors(std::string& err) {
        if (ctx_) {
            err = "weights are already loaded";
            return false;
        }

        std::vector<RequiredTensor> required;
        collect_required(required);
        if (!missing_.empty()) {
            err = "missing tensor '" + missing_ + "'";
            return false;
        }
        /* Shapes and types are checked before anything is allocated: a wrong
         * shape would otherwise reach a GGML_ASSERT inside the graph and abort
         * the whole process, which is not a refusal anybody can catch. */
        if (!validate_tensors(required, err)) return false;

        std::vector<std::string> names;
        names.reserve(required.size());
        for (const RequiredTensor& r : required) names.push_back(r.name);

        /* Two spare object slots of headroom: ggml_new_tensor can fail, and a
         * half-built context would leak whatever it already holds. */
        const size_t mem_size =
            ggml_tensor_overhead() * (names.size() + 4) + ggml_graph_overhead();

        ggml_init_params ctx_params{};
        ctx_params.mem_size = mem_size;
        ctx_params.mem_buffer = nullptr;
        ctx_params.no_alloc = true; /* the backend buffer holds the data */
        ctx_ = ggml_init(ctx_params);
        if (!ctx_) {
            err = "out of memory while allocating the weight context";
            return false;
        }

        std::vector<ggml_tensor*> dst;
        dst.reserve(names.size());
        for (const std::string& name : names) {
            /* Find the tensor in the blob by name rather than by index: the
             * order of the file's tensor table is irrelevant to us and must not
             * become relevant. */
            ggml_tensor* src = ggml_get_tensor(blob_ctx_, name.c_str());
            if (!src || !src->data) {
                err = "tensor '" + name + "' carries no data";
                return false;
            }
            ggml_tensor* copy = ggml_dup_tensor(ctx_, src);
            if (!copy) {
                err = "out of memory while copying tensor '" + name + "'";
                return false;
            }
            ggml_set_name(copy, name.c_str());
            dst.push_back(copy);
        }

        ggml_backend_buffer_type_t buft = buffer_type();
        if (!buft) {
            err = lembed::detail::ggml_cpu_backend_missing_message();
            return false;
        }
        buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
        if (!buf_) {
            err = "cannot allocate the weight buffer";
            return false;
        }

        for (size_t i = 0; i < names.size(); i++) {
            ggml_tensor* src = ggml_get_tensor(blob_ctx_, names[i].c_str());
            ggml_backend_tensor_set(dst[i], ggml_get_data(src), 0,
                                    (size_t)ggml_nbytes(dst[i]));
        }

        bind(dst);
        return true;
    }

    /* Convenience for callers that have no reason to look at the header between
     * the two steps. */
    bool load(const char* path, std::string& err) {
        return open(path, err) && load_tensors(err);
    }

    /* Drops the parsed header and with it the whole GGUF blob, which
     * gguf_free() owns. Called once the vocabulary has been read out of it: the
     * steady-state footprint then is the backend buffer alone. */
    void release_header() { probe_.close(); }

    const Probe& probe() const { return probe_; }
    const SpladeHparams& hparams() const { return hp_; }
    const HeadWeights& head() const { return head_; }
    const std::vector<LayerWeights>& layers() const { return layers_; }
    /* Token ids to keep out of the sparse vector, read from
     * `tokenizer.ggml.*_token_id`. Never a hardcoded list: [CLS] is 101 on a
     * BERT vocabulary and 0 on an XLM-R one. */
    const std::vector<int32_t>& special_ids() const { return special_ids_; }

    /* The CPU device's buffer type, through the backend registry so the same
     * code works with a dynamically loaded CPU variant (ggml_cpu_backend.hpp).
     * nullptr when no CPU backend could be loaded. */
    ggml_backend_buffer_type_t buffer_type() const { return lembed::detail::ggml_cpu_buffer_type(); }

private:
    /* Tries each known spelling of a scalar field, first hit wins. Converters
     * disagree on the spelling and none of them is wrong. */
    static bool read_i32_any(const Probe& probe, const char* arch,
                             const char* const* fields, int n, int32_t& out) {
        for (int i = 0; i < n; i++) {
            if (probe.get_arch_i32(arch, fields[i], out)) return true;
        }
        return false;
    }

    bool read_hparams(std::string& err) {
        if (!probe_.get_str("general.architecture", hp_.arch) || hp_.arch.empty()) {
            err = "GGUF header has no general.architecture";
            return false;
        }

        static const char* kEmbeddingLength[] = {"embedding_length", "hidden_size"};
        static const char* kBlockCount[]     = {"block_count", "num_hidden_layers"};
        static const char* kHeadCount[]      = {"attention.head_count",
                                               "num_attention_heads",
                                               "attention.n_head"};
        static const char* kFeedForward[]    = {"feed_forward_length",
                                               "intermediate_size",
                                               "feed_forward_length"};
        static const char* kContextLength[]  = {"context_length",
                                               "max_position_embeddings"};

        const char* a = hp_.arch.c_str();
        struct Field { const char* const* names; int n; int32_t* out; const char* label; };
        const Field fields[] = {
            {kEmbeddingLength, 2, &hp_.n_embd,  "<arch>.embedding_length"},
            {kBlockCount,     2, &hp_.n_layer, "<arch>.block_count"},
            {kHeadCount,      3, &hp_.n_head,  "<arch>.attention.head_count"},
            {kFeedForward,    2, &hp_.n_ff,    "<arch>.feed_forward_length"},
            {kContextLength,  2, &hp_.n_pos,   "<arch>.context_length"},
        };
        for (const Field& f : fields) {
            if (!read_i32_any(probe_, a, f.names, f.n, *f.out)) {
                err = std::string("GGUF header has no ") + f.label + " for architecture '" +
                      hp_.arch + "'";
                return false;
            }
        }
        if (!probe_.get_arch_i32(a, "vocab_size", hp_.n_vocab)) {
            err = "GGUF header has no <arch>.vocab_size for architecture '" + hp_.arch + "'";
            return false;
        }

        /* The epsilon is a float, not a count, so it has no integer spelling to
         * fall back on. Required rather than defaulted: a wrong epsilon is
         * invisible in the output and shifts every layer. */
        static const char* kEps[] = {"attention.layer_norm_epsilon", "layer_norm_eps",
                                     "layer_norm_epsilon"};
        bool eps_ok = false;
        for (const char* key : kEps) {
            if (probe_.get_arch_f32(a, key, hp_.eps)) { eps_ok = true; break; }
        }
        if (!eps_ok) {
            err = std::string("GGUF header has no <arch>.attention.layer_norm_epsilon for "
                              "architecture '") + hp_.arch + "'";
            return false;
        }

        if (hp_.n_embd <= 0 || hp_.n_head <= 0 || hp_.n_layer <= 0 || hp_.n_ff <= 0 ||
            hp_.n_vocab <= 0 || hp_.n_pos <= 0) {
            err = "GGUF header carries a non-positive BERT dimension";
            return false;
        }
        struct Bound { int value; int max; const char* label; };
        const Bound bounds[] = {
            {hp_.n_embd,  kMaxEmbd,      "embedding_length"},
            {hp_.n_head,  kMaxHeads,     "attention.head_count"},
            {hp_.n_layer, kMaxLayers,    "block_count"},
            {hp_.n_ff,    kMaxFf,        "feed_forward_length"},
            {hp_.n_vocab, kMaxVocab,     "vocab_size"},
            {hp_.n_pos,   kMaxPositions, "context_length"},
        };
        for (const Bound& b : bounds) {
            if (b.value > b.max) {
                err = "GGUF <arch>." + std::string(b.label) + " = " +
                      std::to_string(b.value) + " exceeds the supported maximum of " +
                      std::to_string(b.max);
                return false;
            }
        }
        /* The tensor table is already parsed, so a block_count the file cannot
         * back with tensors is refused here, without building a single name. */
        const int64_t needed = (int64_t)hp_.n_layer * kTensorsPerLayer;
        if (probe_.n_tensors() < needed) {
            err = "GGUF declares " + std::to_string(hp_.n_layer) + " layers but holds only " +
                  std::to_string(probe_.n_tensors()) + " tensors (at least " +
                  std::to_string(needed) + " required)";
            return false;
        }
        if (hp_.n_embd % hp_.n_head != 0) {
            err = "GGUF hidden size is not divisible by the attention head count";
            return false;
        }
        if (hp_.eps <= 0.0f) {
            err = "GGUF LayerNorm epsilon is not positive";
            return false;
        }
        return true;
    }

    /* A tensor the runtime needs, with the shape the graph will use it at.
     * ggml order: ne0 is the row length. A vector is [ne0, 1]. */
    static constexpr int64_t kAnyRows = -1; /* ne1 >= 1, whatever its value */
    struct RequiredTensor {
        std::string name;
        int64_t ne0;
        int64_t ne1;
    };

    static void add(std::vector<RequiredTensor>& out, const char* name,
                    int64_t ne0, int64_t ne1 = 1) {
        out.push_back(RequiredTensor{name, ne0, ne1});
    }

    static void add_layer(std::vector<RequiredTensor>& out, int il, const char* suffix,
                          int64_t ne0, int64_t ne1 = 1) {
        out.push_back(RequiredTensor{"enc." + std::to_string(il) + "." + suffix, ne0, ne1});
    }

    /* Checks every required tensor against the shape and the type the graph
     * relies on (gguf_bert_graph.hpp and SparseSession::build_mlm_head).
     *
     * Types are limited to what this runtime is tested with -- F32, F16 and
     * Q8_0 -- rather than to what ggml can represent: a type nobody has run
     * through this graph is refused by name instead of being trusted. */
    bool validate_tensors(const std::vector<RequiredTensor>& required, std::string& err) const {
        for (const RequiredTensor& r : required) {
            const ggml_tensor* t = ggml_get_tensor(blob_ctx_, r.name.c_str());
            if (!t) {
                err = "missing tensor '" + r.name + "'";
                return false;
            }
            if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 &&
                t->type != GGML_TYPE_Q8_0) {
                err = "tensor '" + r.name + "' has type " + ggml_type_name(t->type) +
                      "; this runtime supports f32, f16 and q8_0";
                return false;
            }
            const bool rows_ok = (r.ne1 == kAnyRows) ? t->ne[1] >= 1 : t->ne[1] == r.ne1;
            if (t->ne[0] != r.ne0 || !rows_ok || t->ne[2] != 1 || t->ne[3] != 1) {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "tensor '%s' has shape [%lld, %lld, %lld, %lld], expected [%lld, %s]",
                         r.name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                         (long long)t->ne[2], (long long)t->ne[3], (long long)r.ne0,
                         r.ne1 == kAnyRows ? "n" : std::to_string(r.ne1).c_str());
                err = msg;
                return false;
            }
        }
        /* The encoder adds row 0 of this table to every token as a view, which
         * assumes single-precision rows. Refusing here is better than reading a
         * quantised row as floats. */
        const ggml_tensor* type_embd = ggml_get_tensor(blob_ctx_, "token_type_embd.weight");
        if (type_embd->type != GGML_TYPE_F32) {
            err = "token_type_embd.weight is not F32";
            return false;
        }
        return true;
    }

    /* Token ids that must never reach the sparse vector: padding, and the
     * sequence delimiters.
     *
     * Read from `tokenizer.ggml.*_token_id` and nothing else. The reference
     * implementation filters a hardcoded {0, 101, 102} *on top of* the metadata,
     * which is only correct for a BERT vocabulary: on an XLM-R export 101 and
     * 102 are ordinary tokens, while 0 (<pad>) and 1 (<s>) are the ones that
     * leak. Ids that the file does not declare are simply not filtered, which
     * is the honest behaviour: a file without the metadata has already been
     * refused for not declaring its capability.
     *
     * Ids are deduplicated: bos and eos are the same token on many
     * vocabularies, and a repeated id would only cost a comparison. */
    void read_special_ids() {
        static const char* kKeys[] = {
            "tokenizer.ggml.pad_token_id",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.cls_token_id",
            "tokenizer.ggml.separator_token_id",
        };
        special_ids_.clear();
        for (const char* key : kKeys) {
            int32_t id = 0;
            if (!probe_.get_i32(key, id)) continue;
            if (id < 0 || id >= hp_.n_vocab) continue; /* out of vocabulary: no-op */
            bool seen = false;
            for (int32_t seen_id : special_ids_) {
                if (seen_id == id) { seen = true; break; }
            }
            if (!seen) special_ids_.push_back(id);
        }
    }

    /* Every tensor the encoder or the head needs. A file missing any of them is
     * refused by name: a SPLADE vector computed from half an encoder is a
     * vector nobody can debug. */
    void collect_required(std::vector<RequiredTensor>& out) {
        const int64_t E = hp_.n_embd, F = hp_.n_ff;
        /* Projections are stored [in, out]: ggml_mul_mat(weight, x) needs
         * weight->ne[0] == x->ne[0]. */
        add(out, "token_embd.weight", E, hp_.n_vocab);
        add(out, "position_embd.weight", E, hp_.n_pos);
        add(out, "token_type_embd.weight", E, kAnyRows);
        add(out, "embd_ln.weight", E);
        add(out, "embd_ln.bias", E);
        for (int il = 0; il < hp_.n_layer; il++) {
            add_layer(out, il, "ln1.weight", E);
            add_layer(out, il, "ln1.bias", E);
            add_layer(out, il, "attn.q.weight", E, E);
            add_layer(out, il, "attn.q.bias", E);
            add_layer(out, il, "attn.k.weight", E, E);
            add_layer(out, il, "attn.k.bias", E);
            add_layer(out, il, "attn.v.weight", E, E);
            add_layer(out, il, "attn.v.bias", E);
            add_layer(out, il, "attn.o.weight", E, E);
            add_layer(out, il, "attn.o.bias", E);
            add_layer(out, il, "ln2.weight", E);
            add_layer(out, il, "ln2.bias", E);
            add_layer(out, il, "ffn.fc1.weight", E, F);
            add_layer(out, il, "ffn.fc1.bias", F);
            add_layer(out, il, "ffn.fc2.weight", F, E);
            add_layer(out, il, "ffn.fc2.bias", E);
        }
        add(out, "mlm_transform.weight", E, E);
        add(out, "mlm_transform.bias", E);
        add(out, "mlm_ln.weight", E);
        add(out, "mlm_ln.bias", E);
        add(out, "mlm_bias", hp_.n_vocab);

        /* Report the first absent tensor rather than crashing later on a null
         * pointer, and say which one. */
        for (const RequiredTensor& r : out) {
            if (!probe_.has_tensor(r.name.c_str())) {
                missing_ = r.name;
                return;
            }
        }
    }

    /* Maps the flat name list back onto the typed views, in the same order. */
    void bind(const std::vector<ggml_tensor*>& t) {
        size_t i = 0;
        head_.tok_embd = t[i++];
        head_.pos_embd = t[i++];
        head_.type_embd = t[i++];
        head_.embd_ln_w = t[i++];
        head_.embd_ln_b = t[i++];
        layers_.resize((size_t)hp_.n_layer);
        for (int il = 0; il < hp_.n_layer; il++) {
            LayerWeights& L = layers_[(size_t)il];
            L.ln1_w = t[i++]; L.ln1_b = t[i++];
            L.q_w   = t[i++]; L.q_b   = t[i++];
            L.k_w   = t[i++]; L.k_b   = t[i++];
            L.v_w   = t[i++]; L.v_b   = t[i++];
            L.o_w   = t[i++]; L.o_b   = t[i++];
            L.ln2_w = t[i++]; L.ln2_b = t[i++];
            L.fc1_w = t[i++]; L.fc1_b = t[i++];
            L.fc2_w = t[i++]; L.fc2_b = t[i++];
        }
        head_.mlm_transform_w = t[i++];
        head_.mlm_transform_b = t[i++];
        head_.mlm_ln_w = t[i++];
        head_.mlm_ln_b = t[i++];
        head_.mlm_bias = t[i++];

        const int64_t* ne = head_.type_embd->ne;
        hp_.n_type = (int)ne[1];
    }

/* Name of the first required tensor that was absent, empty when all were
     * present. Set by collect_required(). */
    const std::string& missing_tensor() const { return missing_; }

private:
    /* Declared first so it is destroyed last: the probe owns the GGUF blob the
     * weights were copied out of, and nothing may outlive its own buffer. */
    Probe probe_;
    SpladeHparams hp_;
    HeadWeights head_;
    std::vector<LayerWeights> layers_;
    std::vector<int32_t> special_ids_;
    std::string missing_;

    ggml_backend_buffer_t buf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    /* The gguf_context owns this: gguf_free() frees it. Never free it here. */
    ggml_context* blob_ctx_ = nullptr;
};

} /* namespace gguf */
} /* namespace lembed */

#endif /* LIBEMBEDDING_GGUF_WEIGHTS_HPP */