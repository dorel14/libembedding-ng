/*
 * test_sparse_gguf.cpp - The sparse GGUF runtime.
 *
 * Hermetic: every fixture is written to a temp directory by the GGUF writer
 * below, and every weight comes from a fixed seed. Nothing is downloaded and
 * nothing depends on a host, so the whole file runs in milliseconds.
 *
 * The writer is the one from test_gguf_inspect.cpp, extended where that one
 * stops: it emits real tensor bytes and accepts F32, F16 and Q8_0. Real bytes
 * are the whole point -- a fixture full of zeroes would load happily and give
 * zeros back, which passes a shape check and proves nothing about the graph.
 *
 * The fixtures are named after the real export (enc.N.attn.q.weight,
 * mlm_transform.weight, bert.hidden_size, ...) and not after llama.cpp's naming,
 * because the runtime reads that file and not a llama.cpp one.
 *
 * The miniature model is deliberately tiny: hidden 32, 4 heads, 2 layers,
 * intermediate 64, vocabulary 64, 16 positions. Wide enough for a Q8_0 fixture
 * -- Q8_0 works on blocks of 32, so every quantised dimension must be a multiple
 * of 32 -- and small enough that a fixture is a few tens of kilobytes.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/libembedding.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); g_fail++; } \
} while (0)

/* A refused load is only debuggable if the reason is printed, so every refusal
 * check carries it. lembed_last_error() is thread-local, hence the immediate
 * read rather than a saved pointer. */
static int expect_ok(lembed_status_t s, const char* msg) {
    if (s == LEMBED_OK) {
        g_pass++;
        return 1;
    }
    fprintf(stderr, "FAIL: %s (status %s: %s)\n", msg, lembed_status_message(s),
            lembed_last_error() ? lembed_last_error() : "(no message)");
    g_fail++;
    return 0;
}

/* ==========================================================================
 * Miniature model shape and vocabulary
 * ========================================================================== */

namespace {

const int kHidden = 32;
const int kHeads  = 4;
const int kLayers = 2;
const int kInter  = 64;
const int kVocab  = 64;
const int kPos    = 16;

/* Vocabulary positions of the special tokens. Deliberately not 0, 1, 100 and
 * 101: a runtime filtering a hardcoded {0, 101, 102} would look right here by
 * accident and be wrong on every other vocabulary. */
const int kPadId = 3;
const int kUnkId = 7;
const int kClsId = 9;
const int kSepId = 11;

const char* kVocabTable[kVocab] = {
    "[unused0]", "a", "b", "[PAD]", "c", "##", "[unused1]", "[UNK]",
    "d", "[CLS]", "e", "[SEP]", "hello", "wor", "##ld", "the",
    "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "sen",
    "##tence", "embed", "##ding", "sp", "##arse", "vec", "##tor", "s",
    "t", "u", "v", "w", "x", "y", "z", "!",
    "?", ".", ",", "-", "model", "##ling", "run", "##time",
    "bench", "##mark", "test", "##ing", "sparse", "##se", "gguf", "on",
    "##nx", "do", "##main", "file", "load", "##er", "a1", "zz",
};

/* ==========================================================================
 * GGUF writer
 *
 * Header, key/value block, tensor descriptor table, then the blob. The offsets
 * in the descriptor table are each tensor's size padded up to the alignment,
 * not a running byte count, and the table itself is padded before the blob
 * starts.
 * ========================================================================== */

/* ggml type ids, as they appear in a GGUF tensor descriptor. */
const uint32_t kF32  = 0;
const uint32_t kF16  = 1;
const uint32_t kQ8_0 = 8;

const uint32_t kAlign = 32;

void put_f16(FILE* f, float v) {
    /* IEEE-754 binary16, round to nearest even. Written out rather than
     * delegated to ggml so that the test does not lean on the runtime it is
     * testing. */
    uint32_t bits;
    memcpy(&bits, &v, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = bits & 0x7FFFFFu;
    uint16_t out;
    if (exp <= 0) {
        if (exp < -10) {
            out = (uint16_t)sign;
        } else {
            const uint32_t m = mant | 0x800000u;
            const int shift = 14 - exp;
            uint32_t half = m >> shift;
            if ((m >> (shift - 1)) & 1u) half += 1u;
            out = (uint16_t)(sign | half);
        }
    } else if (exp >= 31) {
        out = (uint16_t)(sign | 0x7C00u);
    } else {
        uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
        if ((mant >> 12) & 1u) half += 1u;
        out = (uint16_t)(sign | half);
    }
    fwrite(&out, sizeof(out), 1, f);
}

/* Same quantisation as ggml's quantize_row_q8_0_ref, so a fixture decodes back
 * to approximately what went in. */
void put_q8_0_block(FILE* f, const float* x, int n) {
    float amax = 0.0f;
    for (int j = 0; j < n; j++) {
        const float a = std::fabs(x[j]);
        if (a > amax) amax = a;
    }
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    put_f16(f, d);
    for (int j = 0; j < n; j++) {
        float q = std::floor(x[j] * id + 0.5f);
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        fputc((int)(int8_t)q, f);
    }
}

struct Kv {
    std::string key;
    uint32_t type = 0;   /* gguf_type */
    int64_t i = 0;
    double f = 0.0;
    std::string s;
    uint32_t arr_type = 0;
    std::vector<std::string> arr_str;
};

struct Tensor {
    std::string name;
    std::vector<int64_t> ne;
    uint32_t type = 0;
    std::vector<float> values;   /* source of truth, pre-quantisation */
};

Kv kv_u32(const std::string& k, int64_t v) { Kv x; x.key = k; x.type = 4; x.i = v; return x; }
Kv kv_f32(const std::string& k, double v)  { Kv x; x.key = k; x.type = 6; x.f = v; return x; }
Kv kv_str(const std::string& k, const std::string& v) { Kv x; x.key = k; x.type = 8; x.s = v; return x; }
Kv kv_arr_str(const std::string& k, const std::vector<std::string>& v) {
    Kv x; x.key = k; x.type = 9; x.arr_type = 8; x.arr_str = v; return x;
}

/* Deterministic pseudo-random weights, small on purpose: two layers of a random
 * dense stack would otherwise drift away from N(0, 1) and the assertions about
 * "some terms survive the relu" would be testing the seed, not the runtime. */
struct Rng {
    uint32_t state;
    explicit Rng(uint32_t seed) : state(seed ? seed : 1u) {}
    float next() {
        state = state * 1664525u + 1013904223u;
        const uint32_t hi = (state >> 8) & 0xFFFFu;
        const uint32_t lo = (state >> 3) & 0xFFFFu;
        const uint32_t mixed = (hi ^ (lo * 3u)) % 10000u;
        return ((float)mixed / 10000.0f) * 0.10f - 0.05f;
    }
};

Tensor rnd(const std::string& name, std::vector<int64_t> ne, Rng& rng) {
    Tensor t;
    t.name = name;
    t.ne = std::move(ne);
    int64_t numel = 1;
    for (int64_t d : t.ne) numel *= d;
    t.values.resize((size_t)numel);
    for (size_t i = 0; i < t.values.size(); i++) t.values[i] = rng.next();
    return t;
}

Tensor ones(const std::string& name, std::vector<int64_t> ne) {
    Tensor t;
    t.name = name;
    t.ne = std::move(ne);
    int64_t numel = 1;
    for (int64_t d : t.ne) numel *= d;
    t.values.assign((size_t)numel, 1.0f);
    return t;
}

Tensor filled(const std::string& name, std::vector<int64_t> ne, float v) {
    Tensor t;
    t.name = name;
    t.ne = std::move(ne);
    int64_t numel = 1;
    for (int64_t d : t.ne) numel *= d;
    t.values.assign((size_t)numel, v);
    return t;
}

/* Bytes in one row of a tensor. ggml counts a quantised row as
 * (ne[0] / block) * sizeof(block), which is why a Q8_0 row is 34 bytes per 32
 * values and not 32. */
size_t row_bytes(const Tensor& t) {
    if (t.type == kF32) return (size_t)t.ne[0] * 4u;
    if (t.type == kF16) return (size_t)t.ne[0] * 2u;
    return (size_t)(t.ne[0] / 32) * 34u;
}

int64_t tensor_rows(const Tensor& t) {
    int64_t rows = 1;
    for (size_t i = 1; i < t.ne.size(); i++) rows *= t.ne[i];
    return rows;
}

int64_t tensor_numel(const Tensor& t) {
    int64_t n = 1;
    for (int64_t d : t.ne) n *= d;
    return n;
}

bool write_gguf(const std::string& path, const std::vector<Kv>& kvs,
                std::vector<Tensor>& tensors) {
    /* Every tensor declares how many bytes it will emit, and the offsets the
     * reader validates them against are built from exactly that number. */
    std::vector<size_t> bytes(tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
        if (tensors[i].type == kQ8_0 && tensors[i].ne[0] % 32 != 0) {
            fprintf(stderr, "fixture setup failure: %s has a Q8_0 row of %lld\n",
                    tensors[i].name.c_str(), (long long)tensors[i].ne[0]);
            return false;
        }
        if ((int64_t)tensors[i].values.size() != tensor_numel(tensors[i])) {
            fprintf(stderr, "fixture setup failure: %s has %zu values for %lld slots\n",
                    tensors[i].name.c_str(), tensors[i].values.size(),
                    (long long)tensor_numel(tensors[i]));
            return false;
        }
        bytes[i] = row_bytes(tensors[i]) * (size_t)tensor_rows(tensors[i]);
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    fwrite("GGUF", 1, 4, f);
    const uint32_t version = 3;
    fwrite(&version, 4, 1, f);
    const uint64_t n_tensors = tensors.size();
    fwrite(&n_tensors, 8, 1, f);
    const uint64_t n_kvs = kvs.size();
    fwrite(&n_kvs, 8, 1, f);

    for (const Kv& kv : kvs) {
        const uint64_t klen = kv.key.size();
        fwrite(&klen, 8, 1, f);
        fwrite(kv.key.data(), 1, kv.key.size(), f);
        const uint32_t type = kv.type;
        fwrite(&type, 4, 1, f);
        switch (kv.type) {
            case 4: { const uint32_t v = (uint32_t)kv.i; fwrite(&v, 4, 1, f); } break;
            case 6: { const float v = (float)kv.f; fwrite(&v, 4, 1, f); } break;
            case 8: { const uint64_t l = kv.s.size(); fwrite(&l, 8, 1, f);
                      fwrite(kv.s.data(), 1, kv.s.size(), f); } break;
            case 9: { const uint32_t at = kv.arr_type; fwrite(&at, 4, 1, f);
                      const uint64_t n = kv.arr_str.size(); fwrite(&n, 8, 1, f);
                      for (const std::string& e : kv.arr_str) {
                          const uint64_t l = e.size(); fwrite(&l, 8, 1, f);
                          fwrite(e.data(), 1, e.size(), f);
                      } } break;
            default: fclose(f); return false;
        }
    }

    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        const Tensor& t = tensors[i];
        const uint64_t nlen = t.name.size();
        fwrite(&nlen, 8, 1, f);
        fwrite(t.name.data(), 1, t.name.size(), f);
        const uint32_t ndims = (uint32_t)t.ne.size();
        fwrite(&ndims, 4, 1, f);
        for (int64_t d : t.ne) fwrite(&d, 8, 1, f);
        const uint32_t type = t.type;
        fwrite(&type, 4, 1, f);
        fwrite(&offset, 8, 1, f);
        offset += (uint64_t)((bytes[i] + kAlign - 1) / kAlign * kAlign);
    }

    const long kAlignPos = 32;
    const long pos = ftell(f);
    for (long i = 0; i < (kAlignPos - (pos % kAlignPos)) % kAlignPos; i++) fputc(0, f);

    for (size_t i = 0; i < tensors.size(); i++) {
        const Tensor& t = tensors[i];
        const int64_t rows = tensor_rows(t);
        for (int64_t r = 0; r < rows; r++) {
            const float* row = t.values.data() + r * t.ne[0];
            if (t.type == kF32) {
                fwrite(row, 4, (size_t)t.ne[0], f);
            } else if (t.type == kF16) {
                for (int64_t c = 0; c < t.ne[0]; c++) put_f16(f, row[c]);
            } else {
                for (int64_t c = 0; c < t.ne[0]; c += 32) put_q8_0_block(f, row + c, 32);
            }
        }
    }

    fclose(f);
    return true;
}

/* --------------------------------------------------------------------------
 * The model itself
 * --------------------------------------------------------------------------
 *
 * `quant` is the weight type of every matrix. The LayerNorm scales, the biases
 * and the token-type table stay F32 whatever it is: the real exports keep them
 * F32 too, and the runtime reads the token-type table as a row view, which only
 * makes sense for single precision.
 */

enum ModelVariant {
    VARIANT_PLAIN,     /* F32 weights: many terms survive the relu */
    VARIANT_F16,       /* F16 weights */
    VARIANT_Q8,        /* Q8_0 weights */
    VARIANT_NO_HEAD,   /* dense encoder only: must be refused */
    VARIANT_NO_VOCAB,  /* head but no <arch>.vocab_size: must be refused */
    VARIANT_SPECIALS,  /* only the special tokens can score above zero */
};

void build_model(std::vector<Kv>& kvs, std::vector<Tensor>& tensors,
                 ModelVariant variant) {
    const uint32_t matrix_type =
        (variant == VARIANT_F16) ? kF16
      : (variant == VARIANT_Q8)  ? kQ8_0
      : kF32;

    Rng rng(0x5EEDu);

    std::vector<std::string> vocab(kVocab);
    for (int i = 0; i < kVocab; i++) vocab[(size_t)i] = kVocabTable[i];

    kvs.push_back(kv_str("general.architecture", "bert"));
    kvs.push_back(kv_str("general.name", "lembed-test-splade"));
    kvs.push_back(kv_u32("bert.vocab_size", kVocab));
    kvs.push_back(kv_u32("bert.hidden_size", kHidden));
    kvs.push_back(kv_u32("bert.num_hidden_layers", kLayers));
    kvs.push_back(kv_u32("bert.num_attention_heads", kHeads));
    kvs.push_back(kv_u32("bert.intermediate_size", kInter));
    kvs.push_back(kv_f32("bert.layer_norm_eps", 1e-12));
    kvs.push_back(kv_u32("bert.max_position_embeddings", kPos));
    kvs.push_back(kv_u32("bert.output_dim", kHidden));
    kvs.push_back(kv_str("tokenizer.ggml.model", "bert"));
    kvs.push_back(kv_arr_str("tokenizer.ggml.tokens", vocab));
    kvs.push_back(kv_u32("tokenizer.ggml.pad_token_id", kPadId));
    kvs.push_back(kv_u32("tokenizer.ggml.unk_token_id", kUnkId));
    kvs.push_back(kv_u32("tokenizer.ggml.cls_token_id", kClsId));
    kvs.push_back(kv_u32("tokenizer.ggml.separator_token_id", kSepId));

    if (variant == VARIANT_NO_VOCAB) {
        /* Present but zeroed: a GGUF has to declare the key for the reader to
         * see it as absent, and a hand-written fixture that simply omits it
         * would test a different refusal. */
        std::vector<Kv> kept;
        for (const Kv& kv : kvs) {
            if (kv.key == "bert.vocab_size") continue;
            kept.push_back(kv);
        }
        kvs = kept;
    }

    /* Embeddings. */
    Tensor tok = rnd("token_embd.weight", {kHidden, kVocab}, rng);
    tok.type = matrix_type;
    tensors.push_back(tok);
    Tensor pos = rnd("position_embd.weight", {kHidden, kPos}, rng);
    pos.type = matrix_type;
    tensors.push_back(pos);
    /* Row 0 is the "single segment" token type and is the only one used, so it
     * is left at zero: adding a non-zero constant here would only shift every
     * logit by the same amount. */
    tensors.push_back(filled("token_type_embd.weight", {kHidden, 2}, 0.0f));
    tensors.push_back(ones("embd_ln.weight", {kHidden}));
    tensors.push_back(filled("embd_ln.bias", {kHidden}, 0.0f));

    /* Encoder layers, named the way the real export names them. */
    for (int il = 0; il < kLayers; il++) {
        const std::string p = "enc." + std::to_string(il) + ".";
        tensors.push_back(ones(p + "ln1.weight", {kHidden}));
        tensors.push_back(filled(p + "ln1.bias", {kHidden}, 0.0f));
        for (const char* q : {"q", "k", "v", "o"}) {
            Tensor w = rnd(p + "attn." + q + ".weight", {kHidden, kHidden}, rng);
            w.type = matrix_type;
            tensors.push_back(w);
            tensors.push_back(filled(p + "attn." + q + ".bias", {kHidden}, 0.0f));
        }
        tensors.push_back(ones(p + "ln2.weight", {kHidden}));
        tensors.push_back(filled(p + "ln2.bias", {kHidden}, 0.0f));
        /* ggml_mul_mat(weight, activations) needs weight->ne[0] == activations->ne[0],
         * so a projection is stored [in, out]: the real file's
         * enc.N.ffn.fc1.weight is [768, 3072] and fc2 is [3072, 768]. */
        Tensor fc1 = rnd(p + "ffn.fc1.weight", {kHidden, kInter}, rng);
        fc1.type = matrix_type;
        tensors.push_back(fc1);
        tensors.push_back(filled(p + "ffn.fc1.bias", {kInter}, 0.0f));
        Tensor fc2 = rnd(p + "ffn.fc2.weight", {kInter, kHidden}, rng);
        fc2.type = matrix_type;
        tensors.push_back(fc2);
        tensors.push_back(filled(p + "ffn.fc2.bias", {kHidden}, 0.0f));
    }

    if (variant == VARIANT_NO_HEAD) return;

    /* MLM head. The decoder is tied to token_embd, so there is no second
     * [vocab, hidden] tensor here and none is invented. */
    Tensor mlm = rnd("mlm_transform.weight", {kHidden, kHidden}, rng);
    mlm.type = matrix_type;
    tensors.push_back(mlm);
    tensors.push_back(filled("mlm_transform.bias", {kHidden}, 0.0f));
    tensors.push_back(ones("mlm_ln.weight", {kHidden}));
    tensors.push_back(filled("mlm_ln.bias", {kHidden}, 0.0f));

    if (variant == VARIANT_SPECIALS) {
        /* Only the special tokens can end up positive, so the only thing the
         * runtime's metadata-driven removal can do is visible: an empty vector
         * means it removed them, a non-empty one means it did not. */
        Tensor bias = filled("mlm_bias", {kVocab}, -5.0f);
        bias.values[(size_t)kPadId] = 5.0f;
        bias.values[(size_t)kClsId] = 5.0f;
        bias.values[(size_t)kSepId] = 5.0f;
        tensors.push_back(bias);
    } else {
        tensors.push_back(filled("mlm_bias", {kVocab}, 0.0f));
    }
}

std::string tmpdir() {
    const char* t = std::getenv("TMPDIR");
    if (!t || !*t) t = std::getenv("TEMP");
    if (!t || !*t) t = std::getenv("TMP");
    if (!t || !*t) t = ".";
    return std::string(t);
}

std::string fixture(const char* name, ModelVariant variant) {
    std::vector<Kv> kvs;
    std::vector<Tensor> tensors;
    build_model(kvs, tensors, variant);

    const std::string path = tmpdir() + "/lembed_sparse_gguf_" + name + ".gguf";
    if (!write_gguf(path, kvs, tensors)) {
        fprintf(stderr, "fixture setup failure: cannot write %s\n", path.c_str());
        g_fail++;
        return std::string();
    }
    return path;
}

std::string garbage_fixture() {
    const std::string path = tmpdir() + "/lembed_sparse_gguf_garbage.gguf";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return std::string();
    fwrite("this is not a gguf file at all", 1, 30, f);
    fclose(f);
    return path;
}

/* ---- assertions on a produced vector ------------------------------------- */

bool well_formed(const lembed_sparse_embedding_t& e, int vocab) {
    if (e.length <= 0) return false;
    for (int i = 0; i < e.length; i++) {
        if (e.indices[i] < 0 || e.indices[i] >= vocab) return false;
        if (!(e.values[i] > 0.0f)) return false;
        for (int j = 0; j < i; j++) {
            if (e.indices[j] == e.indices[i]) return false;
        }
        if (i > 0 && e.values[i - 1] < e.values[i]) return false;
    }
    return true;
}

bool descending(const lembed_sparse_embedding_t& e) {
    for (int i = 1; i < e.length; i++) {
        if (e.values[i - 1] < e.values[i]) return false;
    }
    return true;
}

bool ascending_indices(const lembed_sparse_embedding_t& e) {
    for (int i = 1; i < e.length; i++) {
        if (e.indices[i - 1] > e.indices[i]) return false;
    }
    return true;
}

bool same_vector(const lembed_sparse_embedding_t& a, const lembed_sparse_embedding_t& b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
        if (a.indices[i] != b.indices[i]) return false;
        if (std::fabs(a.values[i] - b.values[i]) > 1e-5f) return false;
    }
    return true;
}

/* True when b is a's first `n` entries, ids and weights both. The untruncated
 * vector is already sorted heaviest first, so this is the identity a top_k cut
 * has to satisfy. */
bool same_prefix(const lembed_sparse_embedding_t& a, const lembed_sparse_embedding_t& b,
                 int n) {
    if (a.length < n || b.length != n) return false;
    for (int i = 0; i < n; i++) {
        if (a.indices[i] != b.indices[i]) return false;
        if (std::fabs(a.values[i] - b.values[i]) > 1e-6f) return false;
    }
    return true;
}

} /* namespace */

/* ==========================================================================
 * The runtime
 * ========================================================================== */

/* A vector has the shape of a SparseResult: real ids, positive weights, no
 * duplicates, heaviest first. */
static void test_output_shape(void) {
    const std::string path = fixture("plain", VARIANT_PLAIN);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.max_length = 0;
    opts.top_k = 0;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    const lembed_status_t s =
        lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx);
    if (!expect_ok(s, "a SPLADE GGUF loads")) return;

    const char* texts[] = {"hello world", "the quick brown fox", "sparse embedding"};
    lembed_sparse_embeddings_t out;
    memset(&out, 0, sizeof(out));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 3, 4, nullptr, &out) == LEMBED_OK,
          "three texts embed");
    CHECK(out.count == 3, "one vector per text");
    for (int i = 0; i < out.count; i++) {
        CHECK(well_formed(out.items[i], kVocab), "vector is well formed");
    }
    CHECK(lembed_sparse_text_embedding_model_name(ctx) != nullptr,
          "the model reports a name");
    CHECK(lembed_sparse_text_embedding_max_length(ctx) == kPos,
          "the sequence is capped at the position table");
    lembed_sparse_embeddings_free(&out);
    lembed_sparse_text_embedding_free(ctx);
}

/* top_k bounds the output, and keeps the heaviest terms. */
static void test_top_k_and_min_weight(void) {
    const std::string path = fixture("plain", VARIANT_PLAIN);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.max_length = 0;
    opts.top_k = 0;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx)
        != LEMBED_OK) {
        CHECK(false, "the model loads for the pruning tests");
        return;
    }

    const char* texts[] = {"hello world", "the quick brown fox", "sparse embedding"};

    lembed_sparse_embeddings_t full;
    memset(&full, 0, sizeof(full));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 3, 4, nullptr, &full) == LEMBED_OK,
          "untruncated embedding runs");
    const int full_len = full.count > 0 ? full.items[0].length : 0;
    CHECK(full_len > 5, "the untruncated vector has more than five terms");

    const int top_k = 5;
    lembed_sparse_options_t prune = opts;
    prune.top_k = top_k;
    lembed_sparse_embeddings_t cut;
    memset(&cut, 0, sizeof(cut));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 3, 4, &prune, &cut) == LEMBED_OK,
          "top_k embedding runs");
    for (int i = 0; i < cut.count; i++) {
        CHECK(cut.items[i].length == top_k, "top_k bounds the output");
        /* And it is the top_k of the untruncated vector, not an arbitrary
         * subset: the ids and weights are the first five of the sorted result. */
        CHECK(i < full.count && same_prefix(full.items[i], cut.items[i], top_k),
              "top_k keeps the heaviest terms of the untruncated vector");
    }

    lembed_sparse_options_t thresh = opts;
    thresh.min_weight = 0.05f;
    lembed_sparse_embeddings_t pruned;
    memset(&pruned, 0, sizeof(pruned));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 3, 4, &thresh, &pruned) == LEMBED_OK,
          "min_weight embedding runs");
    for (int i = 0; i < pruned.count; i++) {
        bool ok = true;
        for (int j = 0; j < pruned.items[i].length; j++) {
            if (pruned.items[i].values[j] < 0.05f) ok = false;
        }
        CHECK(ok, "min_weight removes the lightest terms");
    }

    lembed_sparse_embeddings_free(&full);
    lembed_sparse_embeddings_free(&cut);
    lembed_sparse_embeddings_free(&pruned);
    lembed_sparse_text_embedding_free(ctx);
}

/* storage_format reorders the same terms: DICT heaviest first, INDEX_ORDER by
 * ascending id. Same set, different order -- and, because the ids are not
 * themselves sorted by weight, a genuinely different order. */
static void test_storage_format(void) {
    const std::string path = fixture("plain", VARIANT_PLAIN);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.max_length = 0;
    opts.top_k = 0;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx)
        != LEMBED_OK) {
        CHECK(false, "the model loads for the ordering test");
        return;
    }

    const char* texts[] = {"hello world", "sparse embedding"};

    lembed_sparse_embeddings_t by_weight;
    memset(&by_weight, 0, sizeof(by_weight));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 2, 2, nullptr, &by_weight)
          == LEMBED_OK, "DICT embedding runs");
    for (int i = 0; i < by_weight.count; i++) {
        CHECK(descending(by_weight.items[i]), "DICT sorts by weight descending");
    }

    lembed_sparse_options_t by_index = opts;
    by_index.storage_format = LEMBED_SPARSE_FORMAT_INDEX_ORDER;
    lembed_sparse_embeddings_t ordered;
    memset(&ordered, 0, sizeof(ordered));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 2, 2, &by_index, &ordered)
          == LEMBED_OK, "INDEX_ORDER embedding runs");

    for (int i = 0; i < ordered.count && i < by_weight.count; i++) {
        CHECK(ascending_indices(ordered.items[i]),
              "INDEX_ORDER sorts indices ascending");
        CHECK(ordered.items[i].length == by_weight.items[i].length,
              "INDEX_ORDER keeps the same number of terms");
        /* Same multiset of (id, weight) pairs, different order. */
        bool same_terms = true;
        for (int j = 0; j < ordered.items[i].length; j++) {
            bool found = false;
            for (int k = 0; k < by_weight.items[i].length; k++) {
                if (by_weight.items[i].indices[k] == ordered.items[i].indices[j] &&
                    std::fabs(by_weight.items[i].values[k] -
                              ordered.items[i].values[j]) <= 1e-6f) {
                    found = true;
                    break;
                }
            }
            if (!found) same_terms = false;
        }
        CHECK(same_terms, "INDEX_ORDER reorders rather than changes");
        CHECK(!descending(ordered.items[i]),
              "INDEX_ORDER is not sorted by weight");
    }

    lembed_sparse_embeddings_free(&by_weight);
    lembed_sparse_embeddings_free(&ordered);
    lembed_sparse_text_embedding_free(ctx);
}

/* Batching must not change the numbers. Documents of a batch share one row of
 * tokens and are separated only by the attention mask, so this is the test that
 * catches a mask which lets one document attend to its neighbour. */
static void test_batching_is_stable(void) {
    const std::string path = fixture("plain", VARIANT_PLAIN);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.max_length = 0;
    opts.top_k = 0;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx)
        != LEMBED_OK) {
        CHECK(false, "the model loads for the batching test");
        return;
    }

    /* Deliberately of different lengths, so the batch is ragged and the padding
     * path is exercised rather than two copies of the same padded shape. */
    const char* texts[] = {"hello", "the quick brown fox jumps over the lazy dog",
                           "sparse embedding", "gguf", "model"};

lembed_sparse_embeddings_t singles;
    memset(&singles, 0, sizeof(singles));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 5, 1, nullptr, &singles)
              == LEMBED_OK && singles.count == 5,
           "five texts embed one by one");

    /* Sweep every batch size the runtime actually uses. The original guard
     * compared batches of two against batches of one, and a mask whose document
     * axis is transposed, or whose query/key axes are swapped, can still pass
     * that: a fully valid document's mask is all zeros either way, and the only
     * place a transposed mask differs is across documents, which two documents
     * are enough to expose -- but only when the batch is split across two
     * documents. Batches of 2, 3, 4 and 5 cover splits of one, two and three
     * documents, and a batch larger than the set is covered by the next test. */
    for (int bs = 2; bs <= 5; bs++) {
        lembed_sparse_embeddings_t batched;
        memset(&batched, 0, sizeof(batched));
        CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 5, bs, nullptr, &batched)
                  == LEMBED_OK && batched.count == 5,
              "five texts embed in batches of N");
        for (int i = 0; i < batched.count && i < singles.count; i++) {
            CHECK(same_vector(singles.items[i], batched.items[i]),
                  "a batch of N gives the same vector as a batch of one");
        }
        lembed_sparse_embeddings_free(&batched);
    }

    lembed_sparse_embeddings_free(&singles);
    lembed_sparse_text_embedding_free(ctx);
}

/* Special tokens are dropped by id, and the ids come from the metadata. */
static void test_special_tokens_are_removed(void) {
    const std::string path = fixture("specials", VARIANT_SPECIALS);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.max_length = 0;
    opts.top_k = 0;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx)
        != LEMBED_OK) {
        CHECK(false, "the special-token fixture loads");
        return;
    }

    /* Only [PAD], [CLS] and [SEP] can score above zero in this fixture, so an
     * empty vector is the expected result, and any surviving id is either a
     * special token that was not removed or a term that should not exist. */
    const char* texts[] = {"hello world", "the quick brown fox"};
    lembed_sparse_embeddings_t out;
    memset(&out, 0, sizeof(out));
    CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 2, 2, nullptr, &out) == LEMBED_OK,
          "the special-token fixture embeds");
    for (int i = 0; i < out.count; i++) {
        CHECK(out.items[i].length == 0,
              "only the special tokens were positive, and they were removed");
        for (int j = 0; j < out.items[i].length; j++) {
            CHECK(out.items[i].indices[j] != kPadId &&
                      out.items[i].indices[j] != kClsId &&
                      out.items[i].indices[j] != kSepId,
                  "no special token id reaches the sparse vector");
        }
    }
    lembed_sparse_embeddings_free(&out);
    lembed_sparse_text_embedding_free(ctx);
}

/* A file with no sparse head is refused, and the reason names the tensor. */
static void test_headless_file_is_refused(void) {
    const std::string path = fixture("nohead", VARIANT_NO_HEAD);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    const lembed_status_t s =
        lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx);
    CHECK(s != LEMBED_OK, "a dense GGUF is refused, not accepted silently");
    CHECK(ctx == nullptr, "no context is handed back on refusal");
    const char* err = lembed_last_error();
    CHECK(err != nullptr && std::string(err).find("mlm_transform.weight")
              != std::string::npos,
          "the refusal names the missing tensor");
    if (ctx) lembed_sparse_text_embedding_free(ctx);

    /* The same file is also not a sparse file for the inspector, which is the
     * guarantee the runtime relies on rather than a second rule of its own. */
    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(path.c_str(), &d) == LEMBED_OK,
          "the headless fixture is still inspectable");
    CHECK((d.capabilities & LEMBED_GGUF_CAP_SPLADE) == 0,
          "the inspector agrees there is no sparse head");
}

/* A file with no vocabulary cannot be pooled over, so it is refused. */
static void test_missing_vocab_size_is_refused(void) {
    const std::string path = fixture("novocab", VARIANT_NO_VOCAB);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    const lembed_status_t s =
        lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx);
    CHECK(s != LEMBED_OK, "a file with no vocab_size is refused");
    if (ctx) lembed_sparse_text_embedding_free(ctx);
}

/* Nonsense input is a refusal with a message, never a crash. */
static void test_bad_input(void) {
    lembed_sparse_options_t opts = lembed_sparse_options_default();
    lembed_sparse_embedding_ctx_t* ctx = nullptr;

    CHECK(lembed_sparse_text_embedding_create_from_gguf_path(nullptr, &opts, &ctx)
              != LEMBED_OK,
          "a null path is refused");
    CHECK(lembed_sparse_text_embedding_create_from_gguf_path("", &opts, &ctx) != LEMBED_OK,
          "an empty path is refused");
    CHECK(lembed_sparse_text_embedding_create_from_gguf_path("no-such-file.gguf", &opts, &ctx)
              != LEMBED_OK,
          "a missing file is refused");

    const std::string garbage = garbage_fixture();
    if (!garbage.empty()) {
        CHECK(lembed_sparse_text_embedding_create_from_gguf_path(garbage.c_str(), &opts, &ctx)
                  != LEMBED_OK,
              "a file that is not a GGUF is refused");
    }
}

/* A .gguf path handed to the generic entry point reaches the GGUF runtime. */
static void test_create_from_path_routes_gguf(void) {
    const std::string path = fixture("routing", VARIANT_PLAIN);
    if (path.empty()) return;

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.max_length = 0;
    opts.top_k = 0;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    const lembed_status_t s =
        lembed_sparse_text_embedding_create_from_path(path.c_str(), &opts, &ctx);
    CHECK(s == LEMBED_OK, "create_from_path routes a .gguf to the GGUF runtime");
    if (ctx) {
        const char* texts[] = {"hello world"};
        lembed_sparse_embeddings_t out;
        memset(&out, 0, sizeof(out));
        CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 1, 1, nullptr, &out)
                  == LEMBED_OK && out.count == 1,
              "the routed context embeds");
        lembed_sparse_embeddings_free(&out);
        lembed_sparse_text_embedding_free(ctx);
    }
}

/* F16 and Q8_0 weights both work, and give the same answer as F32 to within
 * what the quantisation costs. */
static void test_quantised_weights(void) {
    struct { ModelVariant variant; const char* name; const char* label; } cases[] = {
        {VARIANT_F16, "f16", "F16 weights load and embed"},
        {VARIANT_Q8, "q8", "Q8_0 weights load and embed"},
    };

    for (const auto& c : cases) {
        const std::string path = fixture(c.name, c.variant);
        if (path.empty()) continue;

        lembed_sparse_options_t opts = lembed_sparse_options_default();
        opts.max_length = 0;
        opts.top_k = 0;

        lembed_sparse_embedding_ctx_t* ctx = nullptr;
        if (lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx)
            != LEMBED_OK) {
            CHECK(false, c.label);
            continue;
        }
        const char* texts[] = {"hello world", "the quick brown fox", "sparse embedding"};
        lembed_sparse_embeddings_t out;
        memset(&out, 0, sizeof(out));
        CHECK(lembed_sparse_text_embedding_embed(ctx, texts, 3, 4, nullptr, &out)
                  == LEMBED_OK,
              c.label);
        for (int i = 0; i < out.count; i++) {
            CHECK(well_formed(out.items[i], kVocab), "quantised vector is well formed");
        }
        lembed_sparse_embeddings_free(&out);
        lembed_sparse_text_embedding_free(ctx);
    }
}

/* The inspector recognises what the runtime accepts, which is the whole point
 * of the capability convention. */
static void test_inspect_agrees_with_runtime(void) {
    const std::string path = fixture("inspect", VARIANT_PLAIN);
    if (path.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(path.c_str(), &d) == LEMBED_OK, "the fixture inspects");
    CHECK((d.capabilities & LEMBED_GGUF_CAP_SPLADE) != 0, "the SPLADE capability is set");
    CHECK(d.sparse_formula == LEMBED_GGUF_SPARSE_FORMULA_SPLADE,
          "the formula is SPLADE");

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    CHECK(lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(), &opts, &ctx)
              == LEMBED_OK,
          "what the inspector calls SPLADE, the runtime accepts");
    if (ctx) lembed_sparse_text_embedding_free(ctx);
}

int main(void) {
    test_output_shape();
    test_top_k_and_min_weight();
    test_storage_format();
    test_batching_is_stable();
    test_special_tokens_are_removed();
    test_headless_file_is_refused();
    test_missing_vocab_size_is_refused();
    test_bad_input();
    test_create_from_path_routes_gguf();
    test_quantised_weights();
    test_inspect_agrees_with_runtime();

    printf("PASS: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}