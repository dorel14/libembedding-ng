/*
 * test_gguf_inspect.cpp - The GGUF capability convention.
 *
 * Hermetic by construction: every fixture is written to a temp directory by the
 * GGUF writer below, shaped like one of the real models examined in
 * benchmarks/sparse/LE-9.4-validation.md. Nothing is downloaded, so the suite
 * runs in milliseconds and cannot break because a host moved.
 *
 * That matters here specifically. The convention exists because real converters
 * disagree with each other, so testing against one pinned file would only prove
 * it works for that file. The fixtures reproduce the *shapes* that were actually
 * observed:
 *
 *   splade_pp_v1   cstr/splade-pp-en-v1-q8_0.gguf  -- MLM head present
 *   splade_v3      cstr/splade-v3-q8_0.gguf         -- same shape, different model
 *   open_search    mradermacher/opensearch-...-GGUF -- sparse head DROPPED at conversion
 *   bge_m3_scalar  BGE-M3 out_dim == 1             -- a different formula
 *   bert_dense     a plain encoder                  -- no head at all
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/gguf_inspect.h>

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

/* ==========================================================================
 * Minimal GGUF writer
 *
 * Writes only what the reader looks at: the header, a key/value block and a
 * tensor descriptor table. No tensor data is emitted, which is legal -- gguf
 * allows the blob to be absent, and it keeps a fixture at a few hundred bytes.
 * ========================================================================== */

namespace {

struct Kv {
    std::string key;
    uint32_t type = 0;   /* gguf_type */
    /* Scalar payload, interpreted per `type`. */
    int64_t i = 0;
    double  f = 0.0;
    std::string s;

    /* Array payload (GGUF_TYPE_ARRAY == 9). */
    uint32_t arr_type = 0;
    std::vector<std::string> arr_str;
};

struct Tensor {
    std::string name;
    std::vector<int64_t> ne;   /* dimensions */
    uint32_t type = 0;         /* ggml_type */
};

void put_u32(FILE* f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
void put_u64(FILE* f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }
void put_i64(FILE* f, int64_t v) { fwrite(&v, sizeof(v), 1, f); }
void put_f32(FILE* f, float v) { fwrite(&v, sizeof(v), 1, f); }

void put_str(FILE* f, const std::string& s) {
    put_u64(f, (uint64_t)s.size());
    fwrite(s.data(), 1, s.size(), f);
}

bool write_gguf(const std::string& path, const std::vector<Kv>& kvs,
                const std::vector<Tensor>& tensors) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    /* GGML_TYPE_F32, and the matching element size. The reader validates each
     * tensor's offset against the cumulative size of the ones before it and then
     * reads the data, so both have to be right -- which is why the fixtures use
     * small shapes. The convention reads shapes only for sparse_linear.out_dim,
     * and takes the vocabulary from the metadata, so nothing else depends on the
     * dimensions matching a real model. */
    const uint32_t kType = 0;      /* GGML_TYPE_F32 */
    const size_t   kElem = 4;

    fwrite("GGUF", 1, 4, f);
    put_u32(f, 3);
    put_i64(f, (int64_t)tensors.size());
    put_i64(f, (int64_t)kvs.size());

    for (const Kv& kv : kvs) {
        put_str(f, kv.key);
        put_u32(f, kv.type);
        switch (kv.type) {
            /* Every declared type must emit its payload. A type with no case
             * here writes nothing while still advancing nothing, so the reader
             * desynchronises and reports a nonsense string length several keys
             * later -- which looks like a bug in the reader, not the writer. */
            case 0: fputc((int)(kv.i & 0xFF), f); break;   /* UINT8 */
            case 1: fputc((int)(kv.i & 0xFF), f); break;   /* INT8 */
            case 7: fputc((int)(kv.i & 0xFF), f); break;   /* BOOL, stored as int8 */
            case 2: { uint16_t v = (uint16_t)kv.i; fwrite(&v, sizeof(v), 1, f); } break;
            case 3: { int16_t v = (int16_t)kv.i; fwrite(&v, sizeof(v), 1, f); } break;
            case 4: put_u32(f, (uint32_t)kv.i); break;     /* UINT32 */
            case 5: { int32_t v = (int32_t)kv.i; fwrite(&v, sizeof(v), 1, f); } break;
            case 6: put_f32(f, (float)kv.f); break;        /* FLOAT32 */
            case 8: put_str(f, kv.s); break;               /* STRING */
            case 9: {                                     /* ARRAY */
                put_u32(f, kv.arr_type);
                put_u64(f, (uint64_t)kv.arr_str.size());
                for (const std::string& e : kv.arr_str) put_str(f, e);
                break;
            }
            default: break;
        }
    }

    /* Each tensor's data is padded up to the alignment before the next one
     * starts, so the offsets are not a plain running byte count. Getting this
     * wrong only shows up for tensors whose size is not already a multiple of
     * the alignment -- which is why the fixtures that happened to be padded
     * still parsed and one did not. */
    const uint64_t kAlign = 32;
    uint64_t offset = 0;
    for (const Tensor& t : tensors) {
        put_str(f, t.name);
        put_u32(f, (uint32_t)t.ne.size());
        int64_t numel = 1;
        for (int64_t d : t.ne) {
            put_i64(f, d);
            numel *= d;
        }
        put_u32(f, kType);
        put_u64(f, offset);
        const uint64_t bytes = (uint64_t)numel * kElem;
        offset += ((bytes + kAlign - 1) / kAlign) * kAlign;
    }

    /* Pad the descriptor table up to the data alignment, then emit the blob the
     * offsets point into. Without the padding the reader fails at open time;
     * without the blob it fails on the first tensor. */
    const long kAlignPos = 32;
    long pos = ftell(f);
    const long pad = (kAlignPos - (pos % kAlignPos)) % kAlignPos;
    for (long i = 0; i < pad; i++) fputc(0, f);
    for (uint64_t i = 0; i < offset; i++) fputc(0, f);

    fclose(f);
    return true;
}

Kv kv_u32(const std::string& k, int64_t v) {
    Kv x; x.key = k; x.type = 4; x.i = v; return x;
}
Kv kv_str(const std::string& k, const std::string& v) {
    Kv x; x.key = k; x.type = 8; x.s = v; return x;
}

Tensor t(const std::string& name, std::vector<int64_t> ne) {
    Tensor x; x.name = name; x.ne = std::move(ne); x.type = 0; return x;
}

/* Shapes. Names and ranks mirror the real files; dimensions are kept small so a
 * fixture stays a few kilobytes. The convention takes the vocabulary from the
 * metadata and reads a shape only for sparse_linear.out_dim, whose value here is
 * the one that decides the formula. */

std::vector<Tensor> encoder_bert() {
    return {
        t("token_embd.weight", {8, 16}),
        t("position_embd.weight", {8, 8}),
        t("enc.0.attn_q.weight", {8, 8}),
        t("enc.0.attn_q.bias", {8}),
    };
}

/* cstr/splade-pp-en-v1-q8_0.gguf and cstr/splade-v3-q8_0.gguf: the MLM head is
 * present. The 30522 x 768 decoder matrix is tied to token_embd, which is why
 * there is no separate mlm_head tensor. */
std::vector<Tensor> splade_head() {
    std::vector<Tensor> v = encoder_bert();
    v.push_back(t("mlm_transform.weight", {8, 8}));
    v.push_back(t("mlm_transform.bias", {8}));
    v.push_back(t("mlm_ln.weight", {8}));
    v.push_back(t("mlm_ln.bias", {8}));
    v.push_back(t("mlm_bias", {16}));
    return v;
}

/* mradermacher/opensearch-neural-sparse-encoding-doc-v2-mini-GGUF: the
 * conversion dropped the sparse projection entirely. No mlm_*, no
 * sparse_linear, nothing. */
std::vector<Tensor> head_dropped() {
    std::vector<Tensor> v = encoder_bert();
    v.push_back(t("token_embd_norm.weight", {8}));
    v.push_back(t("token_types.weight", {8, 2}));
    return v;
}

/* BGE-M3 style: a scalar weight per token, projected from the hidden state.
 * out_dim == 1 is what selects the scalar formula, so numel must be exactly 1. */
std::vector<Tensor> bge_m3_scalar_head() {
    std::vector<Tensor> v = encoder_bert();
    v.push_back(t("sparse_linear.weight", {1, 1}));
    v.push_back(t("sparse_linear.bias", {1}));
    return v;
}

std::vector<Kv> bert_meta(int64_t vocab) {
    return {
        kv_str("general.architecture", "bert"),
        kv_str("general.name", "fixture"),
        kv_u32("bert.vocab_size", vocab),
        kv_u32("bert.embedding_length", 768),
        kv_u32("bert.block_count", 12),
        kv_u32("bert.max_position_embeddings", 512),
    };
}

std::string tmpdir() {
    const char* t = std::getenv("TMPDIR");
    if (!t || !*t) t = std::getenv("TEMP");
    if (!t || !*t) t = std::getenv("TMP");
    if (!t || !*t) t = ".";
    return std::string(t);
}

std::string fixture(const char* name, const std::vector<Kv>& kvs,
                    const std::vector<Tensor>& ts) {
    std::string p = tmpdir() + "/lembed_gguf_" + name + ".gguf";
    if (!write_gguf(p, kvs, ts)) {
        fprintf(stderr, "fixture setup failure: cannot write %s\n", p.c_str());
        g_fail++;
        return std::string();
    }
    return p;
}

bool caps(const lembed_gguf_desc_t& d, lembed_gguf_capabilities_t bit) {
    return (d.capabilities & bit) != 0;
}

} /* namespace */

/* ==========================================================================
 * The exit criterion: both SPLADE files read as sparse, and the conversion that
 * dropped its head does not.
 * ========================================================================== */

static void test_real_splade_files_are_sparse(void) {
    std::string p = fixture("splade_pp_v1", bert_meta(30522), splade_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "splade_pp_v1 opens");
    CHECK(caps(d, LEMBED_GGUF_CAP_DENSE), "splade_pp_v1 can do dense");
    CHECK(caps(d, LEMBED_GGUF_CAP_SPLADE), "splade_pp_v1 has the MLM head");
    CHECK(caps(d, LEMBED_GGUF_CAP_SPLADE_BIAS), "splade_pp_v1 carries mlm_bias");
    CHECK(caps(d, LEMBED_GGUF_CAP_TIED_DECODER),
          "splade_pp_v1 decoder is tied to token_embd");
    CHECK(d.sparse_formula == LEMBED_GGUF_SPARSE_FORMULA_SPLADE,
          "splade_pp_v1 uses the SPLADE formula");
    CHECK(std::string(lembed_gguf_capability_summary(&d)) == "sparse",
          "splade_pp_v1 summarises as sparse");

    CHECK(d.has_vocab_size && d.vocab_size == 30522, "vocabulary is reported");
    CHECK(d.has_embedding_length && d.embedding_length == 768, "width is reported");
    CHECK(std::string(d.architecture) == "bert", "architecture is reported");
    CHECK(std::string(d.missing_hparams).empty(), "nothing is missing");
    CHECK(std::string(d.diagnostic).empty(), "no diagnostic");
    std::remove(p.c_str());
}

static void test_second_splade_file_has_the_same_shape(void) {
    /* Same convention, different model: the reader must not key on a name. */
    std::vector<Kv> meta = bert_meta(30522);
    meta[1] = kv_str("general.name", "a different SPLADE model");
    std::string p = fixture("splade_v3", meta, splade_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "splade_v3 opens");
    CHECK(caps(d, LEMBED_GGUF_CAP_SPLADE),
          "a second SPLADE file is recognised identically");
    CHECK(std::string(lembed_gguf_capability_summary(&d)) == "sparse",
          "splade_v3 summarises as sparse");
    std::remove(p.c_str());
}

static void test_dropped_head_is_not_sparse(void) {
    /* The exit criterion, stated negatively: a conversion that dropped the head
     * must NOT be accepted as sparse. This is the case that would otherwise
     * produce a plausible-looking dense vector where a sparse one was asked for. */
    std::string p = fixture("open_search", bert_meta(30522), head_dropped());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "the file still opens");
    CHECK(!caps(d, LEMBED_GGUF_CAP_SPLADE),
          "a file with no mlm_transform is NOT sparse");
    CHECK(!caps(d, LEMBED_GGUF_CAP_SPARSE_LINEAR), "nor sparse-linear");
    CHECK(d.sparse_formula == LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN,
          "its sparse formula is unknown");
    CHECK(std::string(lembed_gguf_capability_summary(&d)) == "dense",
          "it summarises as dense, not sparse");
    CHECK(caps(d, LEMBED_GGUF_CAP_DENSE), "the encoder is still usable");
    std::remove(p.c_str());
}

/* ==========================================================================
 * Rule 2: a missing key is reported, never defaulted
 * ========================================================================== */

static void test_missing_vocab_size_is_reported_not_defaulted(void) {
    /* The trap: a default of 30522 reads as a valid vector over the wrong
     * vocabulary, and returns a success code. */
    std::vector<Kv> meta = {
        kv_str("general.architecture", "bert"),
        kv_u32("bert.embedding_length", 768),
        kv_u32("bert.block_count", 12),
    };
    std::string p = fixture("no_vocab", meta, splade_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "the file opens");
    CHECK(d.has_vocab_size == 0, "an absent vocabulary is marked absent");
    CHECK(!caps(d, LEMBED_GGUF_CAP_SPLADE),
          "a sparse head without a vocabulary is refused");
    CHECK(d.sparse_formula == LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN,
          "no formula is claimed without a vocabulary");
    CHECK(strstr(d.missing_hparams, "vocab_size") != nullptr,
          "the missing key is named");
    CHECK(std::string(d.diagnostic).find("vocab_size") != std::string::npos,
          "the diagnostic says why");
    std::remove(p.c_str());
}

static void test_missing_architecture_is_reported(void) {
    std::vector<Kv> meta = {
        kv_str("general.name", "no architecture"),
        kv_u32("bert.vocab_size", 30522),
    };
    std::string p = fixture("no_arch", meta, splade_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "the file opens");
    CHECK(strstr(d.missing_hparams, "general.architecture") != nullptr,
          "a missing architecture is named");
    std::remove(p.c_str());
}

/* ==========================================================================
 * Rule 3: the two sparse formulas are different algorithms
 * ========================================================================== */

static void test_bge_m3_scalar_is_not_splade(void) {
    std::string p = fixture("bge_m3", bert_meta(30522), bge_m3_scalar_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "bge_m3 opens");
    CHECK(caps(d, LEMBED_GGUF_CAP_SPARSE_LINEAR), "the sparse_linear head is seen");
    CHECK(!caps(d, LEMBED_GGUF_CAP_SPLADE), "it is not reported as SPLADE");
    CHECK(d.sparse_formula == LEMBED_GGUF_SPARSE_FORMULA_SCALAR,
          "out_dim == 1 selects the scalar formula");
    CHECK(std::string(lembed_gguf_capability_summary(&d)) == "sparse",
          "it is still a sparse model");
    std::remove(p.c_str());
}

/* Both heads present and the shape is ambiguous: refuse rather than pick. */
static void test_ambiguous_formula_is_refused(void) {
    std::vector<Tensor> ts = splade_head();
    ts.push_back(t("sparse_linear.weight", {7, 1}));  /* out_dim 7: neither 1 nor V */
    std::string p = fixture("ambiguous", bert_meta(30522), ts);
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "the file opens");
    CHECK(d.sparse_formula == LEMBED_GGUF_SPARSE_FORMULA_UNKNOWN,
          "an unrecognised sparse_linear shape yields no formula");
    CHECK(std::string(lembed_gguf_capability_summary(&d)) != "sparse",
          "it does not claim to be sparse");
    std::remove(p.c_str());
}

/* ==========================================================================
 * Rule 1: presence, not names or flags
 * ========================================================================== */

static void test_has_mlm_head_flag_is_not_required(void) {
    /* The SPLADE fixtures carry no has_mlm_head key at all, and are still
     * recognised. If the convention ever starts trusting that flag instead of
     * tensor presence, this stops passing -- which is the point. */
    std::string p = fixture("no_flag", bert_meta(30522), splade_head());
    if (p.empty()) return;
    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(caps(d, LEMBED_GGUF_CAP_SPLADE),
          "presence of the head is enough, no flag needed");
    std::remove(p.c_str());
}

static void test_flag_without_head_does_not_create_capability(void) {
    /* The mirror image: advertising the capability without shipping the tensor
     * must not be believed. */
    std::vector<Kv> meta = bert_meta(30522);
    Kv flag; flag.key = "bert.has_mlm_head"; flag.type = 7; flag.i = 1; /* BOOL */
    meta.push_back(flag);
    std::string p = fixture("flag_only", meta, head_dropped());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(!caps(d, LEMBED_GGUF_CAP_SPLADE),
          "a flag without the tensor does not create the capability");
    std::remove(p.c_str());
}

/* ==========================================================================
 * Special tokens come from the metadata
 * ========================================================================== */

static void test_special_tokens_are_read_from_metadata(void) {
    std::vector<Kv> meta = bert_meta(30522);
    meta.push_back(kv_u32("tokenizer.ggml.cls_token_id", 101));
    meta.push_back(kv_u32("tokenizer.ggml.separator_token_id", 102));
    meta.push_back(kv_u32("tokenizer.ggml.pad_token_id", 0));
    std::string p = fixture("specials", meta, splade_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(d.has_cls_token_id && d.cls_token_id == 101, "[CLS] comes from metadata");
    CHECK(d.has_separator_token_id && d.separator_token_id == 102,
          "[SEP] comes from metadata");
    CHECK(d.has_pad_token_id && d.pad_token_id == 0, "[PAD] comes from metadata");
    CHECK(d.has_bos_token_id == 0, "an absent id stays absent");
    CHECK(caps(d, LEMBED_GGUF_CAP_SPECIAL_TOKENS),
          "the presence of special ids is reported");
    std::remove(p.c_str());
}

/* ==========================================================================
 * Refusals and bad input
 * ========================================================================== */

static void test_plain_encoder_has_no_head(void) {
    std::string p = fixture("dense_only", bert_meta(30522), encoder_bert());
    if (p.empty()) return;
    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(caps(d, LEMBED_GGUF_CAP_DENSE), "dense is available");
    CHECK(!caps(d, LEMBED_GGUF_CAP_SPLADE), "no sparse head");
    CHECK(std::string(lembed_gguf_capability_summary(&d)) == "dense", "dense");
    std::remove(p.c_str());
}

static void test_missing_encoder_is_unsupported(void) {
    /* No token embeddings: this is not a text encoder, whatever else it has. */
    std::string p = fixture("no_encoder", bert_meta(30522), splade_head());
    if (p.empty()) return;
    /* Rebuild without the encoder tensor. */
    std::vector<Tensor> only_head;
    only_head.push_back(t("mlm_transform.weight", {768, 768}));
    only_head.push_back(t("mlm_bias", {30522}));
    write_gguf(p, bert_meta(30522), only_head);

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(strstr(d.diagnostic, "token_embd.weight") != nullptr,
          "the missing encoder tensor is named");
    CHECK(std::string(lembed_gguf_capability_summary(&d)) == "unsupported",
          "it is not usable");
    std::remove(p.c_str());
}

static void test_unreadable_file_is_refused_cleanly(void) {
    lembed_gguf_desc_t d;
    std::string missing = tmpdir() + "/lembed_gguf_does_not_exist.gguf";
    std::remove(missing.c_str());

    CHECK(lembed_gguf_inspect(missing.c_str(), &d) != LEMBED_OK,
          "a missing file is an error");
    CHECK(std::string(d.diagnostic).find("cannot read") != std::string::npos,
          "and says so");

    /* A file that exists but is not GGUF. */
    std::string junk = tmpdir() + "/lembed_gguf_junk.gguf";
    FILE* f = fopen(junk.c_str(), "wb");
    if (f) {
        fwrite("NOTGGUF-not-a-header-at-all-really", 1, 33, f);
        fclose(f);
        CHECK(lembed_gguf_inspect(junk.c_str(), &d) != LEMBED_OK,
              "a non-GGUF file is refused");
        std::remove(junk.c_str());
    }

    /* A null path must not crash. */
    CHECK(lembed_gguf_inspect(nullptr, &d) == LEMBED_ERROR_INVALID_ARGUMENT,
          "a null path is rejected");
    CHECK(lembed_gguf_inspect(missing.c_str(), nullptr) ==
              LEMBED_ERROR_INVALID_ARGUMENT,
          "a null destination is rejected");
}

static void test_capability_names(void) {
    CHECK(std::string(lembed_gguf_capability_name(LEMBED_GGUF_CAP_SPLADE)) == "sparse",
          "splade is named");
    CHECK(std::string(lembed_gguf_capability_name(LEMBED_GGUF_CAP_DENSE)) == "dense",
          "dense is named");
    CHECK(std::string(lembed_gguf_capability_name(LEMBED_GGUF_CAP_NONE)) == "unknown",
          "no capability is unknown");
    /* An unassigned bit must not be mistaken for a known one. */
    CHECK(std::string(lembed_gguf_capability_name(
              (lembed_gguf_capabilities_t)1 << 40)) == "unknown",
          "an unknown bit is unknown, not dense");
}

/* Converters spell the same fact differently. llama.cpp's BERT converter writes
 * `bert.context_length`; the SPLADE files carry `bert.max_position_embeddings`.
 * Both have to be accepted, and one of the two spellings must be enough. */
static void test_context_length_accepts_both_spellings(void) {
    std::vector<Kv> llama_style = bert_meta(30522);
    llama_style.push_back(kv_u32("bert.context_length", 512));
    std::string p = fixture("ctx_llama", llama_style, splade_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(d.has_context_length && d.context_length == 512,
          "bert.context_length is read");
    std::remove(p.c_str());

    /* bert_meta() already spells it the SPLADE way, with no context_length key,
     * so the fallback path is what carries this one. */
    p = fixture("ctx_splade", bert_meta(30522), splade_head());
    if (p.empty()) return;

    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(d.has_context_length && d.context_length == 512,
          "bert.max_position_embeddings is read too");
    std::remove(p.c_str());
}

/* llama.cpp's BERT converter writes no vocab_size at all. The file is a fine
 * dense encoder, and the convention has to say so rather than guess a
 * vocabulary -- and must not let it anywhere near a sparse claim. Observed on a
 * real all-MiniLM-L6-v2 Q4_K_M export. */
static void test_dense_file_without_vocab_is_dense_only(void) {
    std::vector<Kv> meta = {
        kv_str("general.architecture", "bert"),
        kv_str("general.name", "all-MiniLM-L6-v2"),
        kv_u32("bert.block_count", 6),
        kv_u32("bert.context_length", 512),
        kv_u32("bert.embedding_length", 384),
        kv_u32("bert.pooling_type", 1),
    };
    std::string p = fixture("dense_no_vocab", meta, encoder_bert());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(d.has_vocab_size == 0, "no vocabulary is invented");
    CHECK(caps(d, LEMBED_GGUF_CAP_DENSE), "dense still works");
    CHECK(strstr(d.missing_hparams, "vocab_size") != nullptr,
          "the absent vocabulary is named");
    std::remove(p.c_str());
}

/* The SPLADE files spell the geometry with llama.cpp's names rather than the
 * llama.cpp-converted dense ones. Without the synonyms those files report no
 * width and no depth, and the runtime that comes next cannot size anything. */
static void test_architecture_field_synonyms(void) {
    std::vector<Kv> llama_style = {
        kv_str("general.architecture", "bert"),
        kv_str("general.name", "splade-like"),
        kv_u32("bert.vocab_size", 30522),
        kv_u32("bert.hidden_size", 768),
        kv_u32("bert.num_hidden_layers", 12),
        kv_u32("bert.max_position_embeddings", 512),
        kv_u32("bert.intermediate_size", 3072),
        kv_u32("bert.num_attention_heads", 12),
    };
    std::string p = fixture("synonyms", llama_style, splade_head());
    if (p.empty()) return;

    lembed_gguf_desc_t d;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(d.has_vocab_size && d.vocab_size == 30522, "vocab_size read");
    CHECK(d.has_embedding_length && d.embedding_length == 768,
          "bert.hidden_size is accepted as the width");
    CHECK(d.has_block_count && d.block_count == 12,
          "bert.num_hidden_layers is accepted as the depth");
    CHECK(d.has_context_length && d.context_length == 512,
          "max_position_embeddings is accepted as the context length");
    CHECK(std::string(d.missing_hparams).empty(), "nothing is missing");
    CHECK(caps(d, LEMBED_GGUF_CAP_SPLADE),
          "and the file is still recognised as sparse");
    std::remove(p.c_str());

    /* bert_meta() uses the other spellings, so they must keep working. */
    p = fixture("no_synonyms", bert_meta(30522), splade_head());
    if (p.empty()) return;
    CHECK(lembed_gguf_inspect(p.c_str(), &d) == LEMBED_OK, "opens");
    CHECK(d.has_embedding_length && d.embedding_length == 768,
          "bert.embedding_length still accepted");
    CHECK(d.has_block_count && d.block_count == 12,
          "bert.block_count still accepted");
    std::remove(p.c_str());
}

int main(void) {
    test_real_splade_files_are_sparse();
    test_second_splade_file_has_the_same_shape();
    test_dropped_head_is_not_sparse();

    test_missing_vocab_size_is_reported_not_defaulted();
    test_missing_architecture_is_reported();

    test_bge_m3_scalar_is_not_splade();
    test_ambiguous_formula_is_refused();

    test_has_mlm_head_flag_is_not_required();
    test_flag_without_head_does_not_create_capability();

    test_special_tokens_are_read_from_metadata();

    test_plain_encoder_has_no_head();
    test_missing_encoder_is_unsupported();
    test_unreadable_file_is_refused_cleanly();
    test_capability_names();
    test_context_length_accepts_both_spellings();
    test_dense_file_without_vocab_is_dense_only();
    test_architecture_field_synonyms();

    printf("%s: %d passed, %d failed\n",
           g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}