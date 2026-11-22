/*
 * test_multivector.cpp - multi-vector (ColBERT) handle
 *
 * Argument validation, layout, error paths: no model, no network.
 * Behaviour on a real model: only when the local answerai-colbert-small-v1
 * directory exists (LEMBED_COLBERT_MODEL_DIR, else
 * ~/.cache/libembedding/models--answerdotai-answerai-colbert-small-v1);
 * otherwise that part is skipped and the test still passes.
 *
 * Auteur: David Orel
 * Version: 1.16.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static int passes = 0, failures = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while (0)

static std::string model_dir() {
    const char* e = getenv("LEMBED_COLBERT_MODEL_DIR");
    if (e && e[0]) return e;
    const char* home = getenv("USERPROFILE");
    if (!home || !home[0]) home = getenv("HOME");
    if (!home || !home[0]) return std::string();
    return std::string(home) +
           "/.cache/libembedding/models--answerdotai-answerai-colbert-small-v1";
}

static float norm_of(const float* v, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; i++) s += (double)v[i] * v[i];
    return (float)std::sqrt(s);
}

static void test_without_model() {
    lembed_multivector_options_t o = lembed_multivector_options_default();
    ASSERT(o.model_dir == nullptr, "default model_dir is NULL");
    ASSERT(o.provider == LEMBED_PROVIDER_CPU, "default provider CPU");
    ASSERT(o.normalize == 1, "default normalize 1");
    ASSERT(o.mask_punctuation == -1, "default mask_punctuation -1");
    ASSERT(o.query_expansion == -1 && o.attend_to_expansion == -1, "default expansion follows the model");
    ASSERT(o.batch_size == 0 && o.query_maxlen == 0 && o.doc_maxlen == 0, "default sizes 0");

    lembed_multivector_t* mv = nullptr;
    ASSERT(lembed_multivector_create(nullptr, &mv) == LEMBED_ERROR_INVALID_ARGUMENT, "create NULL options");
    ASSERT(lembed_multivector_create(&o, nullptr) == LEMBED_ERROR_INVALID_ARGUMENT, "create NULL out");
    ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_INVALID_ARGUMENT, "create NULL model_dir");
    o.model_dir = "";
    ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_INVALID_ARGUMENT, "create empty model_dir");

    o.model_dir = "some/model.gguf";
    ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_UNSUPPORTED, ".gguf is unsupported");
    ASSERT(mv == nullptr, "no handle for .gguf");
    o.model_dir = "SOME/MODEL.GGUF";
    ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_UNSUPPORTED, ".GGUF is unsupported");

    o.model_dir = "this/directory/does/not/exist";
    ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_IO, "missing dir is an IO error");
    ASSERT(mv == nullptr, "no handle for a missing dir");
    ASSERT(lembed_last_error() != nullptr && lembed_last_error()[0] != '\0', "last_error is set");

    /* Invalid configuration: a directory with a graph file, a tokenizer and
     * an onnx_config.json that is not JSON. Rejected before the graph is read. */
    {
        namespace fs = std::filesystem;
        const fs::path d = fs::temp_directory_path() / "lembed_mv_badcfg";
        std::error_code ec;
        fs::remove_all(d, ec);
        fs::create_directories(d, ec);
        std::ofstream(d / "model.onnx", std::ios::binary) << "x";
        std::ofstream(d / "tokenizer.json", std::ios::binary) << "{}";
        std::ofstream(d / "onnx_config.json", std::ios::binary) << "this is not json";
        const std::string ds = d.string();
        o.model_dir = ds.c_str();
        ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_IO, "invalid config is an IO error");
        ASSERT(mv == nullptr, "no handle for an invalid config");
        fs::remove_all(d, ec);
    }

    /* Hostile / inconsistent configuration, rejected before the graph is read
     * (the model.onnx here is a one-byte file). */
    {
        namespace fs = std::filesystem;
        const char* tok = "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"[PAD]\":0,\"[UNK]\":1,"
                          "\"[CLS]\":2,\"[SEP]\":3,\"a\":4,\"[unused0]\":5,\"[unused1]\":6,\"[MASK]\":7}}}";
        struct Case { const char* cfg; const char* expect; } cases[] = {
            { "{\"query_length\":1e12}", "query_length" },
            { "{\"query_length\":3.5}", "query_length" },
            { "{\"document_length\":-4}", "document_length" },
            { "{\"query_length\":\"32\"}", "query_length" },
            { "{\"query_length\":100000}", "query_length" },
            { "{\"query_prefix_id\":99999}", "outside the vocabulary" },
            { "{\"mask_token_id\":99999}", "outside the vocabulary" },
            { "{\"pad_token_id\":99999}", "outside the vocabulary" },
        };
        for (const Case& c : cases) {
            const fs::path d = fs::temp_directory_path() / "lembed_mv_hostile";
            std::error_code ec;
            fs::remove_all(d, ec);
            fs::create_directories(d, ec);
            std::ofstream(d / "model.onnx", std::ios::binary) << "x";
            std::ofstream(d / "tokenizer.json", std::ios::binary) << tok;
            std::ofstream(d / "onnx_config.json", std::ios::binary) << c.cfg;
            const std::string ds = d.string();
            lembed_multivector_options_t oc = lembed_multivector_options_default();
            oc.model_dir = ds.c_str();
            lembed_multivector_t* h = nullptr;
            const lembed_status_t st = lembed_multivector_create(&oc, &h);
            ASSERT(st == LEMBED_ERROR_IO, c.cfg);
            ASSERT(h == nullptr, "no handle for a hostile config");
            const char* err = lembed_last_error();
            ASSERT(err && strstr(err, c.expect) != nullptr, "error names the offending field");
            fs::remove_all(d, ec);
        }
    }

    o.model_dir = "x";
    o.query_maxlen = 100000;
    ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_INVALID_ARGUMENT, "huge query_maxlen");
    o.query_maxlen = 0;
    o.batch_size = -1;
    ASSERT(lembed_multivector_create(&o, &mv) == LEMBED_ERROR_INVALID_ARGUMENT, "negative batch_size");

    /* Accessors on NULL */
    ASSERT(lembed_multivector_dim(nullptr) == 0, "dim(NULL) == 0");
    ASSERT(lembed_multivector_desc(nullptr) == nullptr, "desc(NULL)");
    ASSERT(lembed_multivector_backend_name(nullptr) == nullptr, "backend_name(NULL)");
    lembed_stats_t st;
    memset(&st, 0xFF, sizeof(st));
    lembed_multivector_stats(nullptr, &st);
    ASSERT(st.texts_embedded == 0 && st.batches_run == 0, "stats(NULL) zeroed");
    lembed_multivector_free(nullptr);

    const char* t[] = { "a" };
    lembed_multivectors_t r;
    ASSERT(lembed_multivector_embed(nullptr, t, 1, LEMBED_MV_QUERY, &r) == LEMBED_ERROR_INVALID_ARGUMENT,
           "embed NULL handle");

    /* free: idempotent and zeroing */
    lembed_multivectors_free(nullptr);
    lembed_multivectors_t m;
    memset(&m, 0, sizeof(m));
    lembed_multivectors_free(&m);
    m.data = (float*)malloc(16 * sizeof(float));
    m.offsets = (int64_t*)malloc(3 * sizeof(int64_t));
    m.offsets[0] = 0; m.offsets[1] = 2; m.offsets[2] = 4;
    m.num_docs = 2; m.dim = 4; m.total_vectors = 4;
    lembed_multivectors_free(&m);
    ASSERT(m.data == nullptr && m.offsets == nullptr, "free nulls the buffers");
    ASSERT(m.num_docs == 0 && m.dim == 0 && m.total_vectors == 0, "free zeroes the counts");
    lembed_multivectors_free(&m);
    ASSERT(m.data == nullptr && m.offsets == nullptr, "free is idempotent");
}

static void test_with_model(const std::string& dir) {
    lembed_multivector_options_t o = lembed_multivector_options_default();
    o.model_dir = dir.c_str();
    lembed_multivector_t* mv = nullptr;
    lembed_status_t s = lembed_multivector_create(&o, &mv);
    if (s != LEMBED_OK) {
        fprintf(stderr, "FAIL: create on %s: %d %s\n", dir.c_str(), (int)s, lembed_last_error());
        failures++;
        return;
    }
    ASSERT(lembed_multivector_dim(mv) == 96, "dim 96");
    ASSERT(strcmp(lembed_multivector_backend_name(mv), "onnxruntime") == 0, "backend onnxruntime");
    const lembed_model_desc_t* d = lembed_multivector_desc(mv);
    ASSERT(d && d->dimension == 96 && d->max_length == 300, "desc dimension / max_length");

    /* Invalid embed arguments */
    {
        const char* t[] = { "a" };
        lembed_multivectors_t r;
        ASSERT(lembed_multivector_embed(mv, nullptr, 1, LEMBED_MV_QUERY, &r) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL texts");
        ASSERT(lembed_multivector_embed(mv, t, 0, LEMBED_MV_QUERY, &r) == LEMBED_ERROR_INVALID_ARGUMENT, "n == 0");
        ASSERT(lembed_multivector_embed(mv, t, 1, LEMBED_MV_QUERY, nullptr) == LEMBED_ERROR_INVALID_ARGUMENT, "NULL out");
        ASSERT(lembed_multivector_embed(mv, t, 1, (lembed_mv_mode_t)7, &r) == LEMBED_ERROR_INVALID_ARGUMENT, "bad mode");
    }

    const char* docs[] = {
        "Paris is the capital and most populous city of France.",
        "Bananas are rich in potassium and grow in tropical climates.",
        "Hello, world!",
    };
    const char* query[] = { "what is the capital of France?" };

    lembed_multivectors_t dv, qv;
    memset(&dv, 0, sizeof(dv));
    memset(&qv, 0, sizeof(qv));
    ASSERT(lembed_multivector_embed(mv, docs, 3, LEMBED_MV_DOCUMENT, &dv) == LEMBED_OK, "embed documents");
    ASSERT(lembed_multivector_embed(mv, query, 1, LEMBED_MV_QUERY, &qv) == LEMBED_OK, "embed query");

    /* Layout */
    ASSERT(dv.num_docs == 3 && dv.dim == 96, "doc num_docs / dim");
    ASSERT(dv.offsets[0] == 0, "offsets[0] == 0");
    ASSERT(dv.offsets[3] == dv.total_vectors, "offsets[num_docs] == total_vectors");
    bool mono = true;
    for (int i = 0; i < 3; i++) mono = mono && dv.offsets[i + 1] > dv.offsets[i];
    ASSERT(mono, "every document has at least one vector, offsets increase");

    /* Norms and NaN */
    bool unit = true, finite = true;
    for (int64_t v = 0; v < dv.total_vectors; v++) {
        const float n = norm_of(dv.data + v * dv.dim, dv.dim);
        unit = unit && std::fabs(n - 1.0f) < 1e-4f;
        finite = finite && std::isfinite(n);
    }
    for (int64_t v = 0; v < qv.total_vectors; v++) {
        const float n = norm_of(qv.data + v * qv.dim, qv.dim);
        unit = unit && std::fabs(n - 1.0f) < 1e-4f;
        finite = finite && std::isfinite(n);
    }
    ASSERT(unit, "every vector has norm 1");
    ASSERT(finite, "no NaN / Inf");

    /* Relevant document scores above the unrelated one */
    float s_rel = 0.0f, s_unrel = 0.0f;
    ASSERT(lembed_maxsim(qv.data, (int)qv.total_vectors, dv.data + dv.offsets[0] * dv.dim,
                         (int)(dv.offsets[1] - dv.offsets[0]), dv.dim, &s_rel) == LEMBED_OK, "maxsim rel");
    ASSERT(lembed_maxsim(qv.data, (int)qv.total_vectors, dv.data + dv.offsets[1] * dv.dim,
                         (int)(dv.offsets[2] - dv.offsets[1]), dv.dim, &s_unrel) == LEMBED_OK, "maxsim unrel");
    ASSERT(s_rel > s_unrel, "MaxSim(query, relevant) > MaxSim(query, unrelated)");
    float batch_scores[3];
    ASSERT(lembed_maxsim_batch(qv.data, (int)qv.total_vectors, dv.data, dv.offsets, 3, dv.dim,
                               batch_scores) == LEMBED_OK, "maxsim_batch");
    ASSERT(std::fabs(batch_scores[0] - s_rel) < 1e-5f, "maxsim_batch agrees with maxsim");

    lembed_multivectors_free(&dv);
    lembed_multivectors_free(&qv);

    /* Document punctuation is dropped: "Hello, world!" is [CLS] [D] hello world [SEP]
     * (5 vectors) without punctuation, 7 with it. */
    {
        const char* one[] = { "Hello, world!" };
        lembed_multivectors_t a, b;
        ASSERT(lembed_multivector_embed(mv, one, 1, LEMBED_MV_DOCUMENT, &a) == LEMBED_OK, "punct doc");
        ASSERT(a.total_vectors == 5, "punctuation removed from the document (5 vectors)");

        lembed_multivector_options_t o2 = o;
        o2.mask_punctuation = 0;
        lembed_multivector_t* keep = nullptr;
        ASSERT(lembed_multivector_create(&o2, &keep) == LEMBED_OK, "create mask_punctuation=0");
        if (keep) {
            ASSERT(lembed_multivector_embed(keep, one, 1, LEMBED_MV_DOCUMENT, &b) == LEMBED_OK, "keep punct doc");
            ASSERT(b.total_vectors == 7, "punctuation kept when mask_punctuation = 0 (7 vectors)");
            lembed_multivectors_free(&b);
            lembed_multivector_free(keep);
        }
        lembed_multivectors_free(&a);
    }

    /* Query expansion: padded to query_length with [MASK], all vectors kept. */
    {
        const char* q[] = { "capital of France", "a" };
        lembed_multivector_options_t oe = o;
        oe.query_expansion = 1;
        lembed_multivector_options_t on = o;
        on.query_expansion = 0;
        lembed_multivector_t* mve = nullptr;
        lembed_multivector_t* mvn = nullptr;
        ASSERT(lembed_multivector_create(&oe, &mve) == LEMBED_OK, "create expansion on");
        ASSERT(lembed_multivector_create(&on, &mvn) == LEMBED_OK, "create expansion off");
        if (mve && mvn) {
            lembed_multivectors_t a, b;
            ASSERT(lembed_multivector_embed(mve, q, 2, LEMBED_MV_QUERY, &a) == LEMBED_OK, "expanded query");
            ASSERT(a.offsets[1] == 32 && a.offsets[2] == 64, "expanded queries have 32 vectors each");
            ASSERT(lembed_multivector_embed(mvn, q, 2, LEMBED_MV_QUERY, &b) == LEMBED_OK, "plain query");
            ASSERT(b.offsets[1] == 6 && b.offsets[2] - b.offsets[1] == 4,
                   "unexpanded queries keep real tokens ([CLS] [Q] tokens [SEP])");
            lembed_multivectors_free(&a);
            lembed_multivectors_free(&b);
        }
        lembed_multivector_free(mve);
        lembed_multivector_free(mvn);
    }

    /* A skiplist word absent from the vocabulary must not turn [UNK] into a
     * skipped id: unknown tokens of a document are kept. Needs a copy of the
     * model directory with its own config (hard link of the graph, else copy). */
    {
        namespace fs = std::filesystem;
        const fs::path src(dir);
        const fs::path d = fs::temp_directory_path() / "lembed_mv_unk";
        std::error_code ec;
        fs::remove_all(d, ec);
        fs::create_directories(d, ec);
        fs::copy_file(src / "tokenizer.json", d / "tokenizer.json", ec);
        const fs::path onnx = src / "vespa_colbert.onnx";
        fs::create_hard_link(onnx, d / "vespa_colbert.onnx", ec);
        if (ec) { ec.clear(); fs::copy_file(onnx, d / "vespa_colbert.onnx", ec); }
        std::ofstream(d / "onnx_config.json", std::ios::binary)
            << "{\"query_prefix_id\":1,\"document_prefix_id\":2,\"mask_token_id\":103,"
               "\"skiplist_words\":[\"\xE2\x98\x83\"]}";
        const std::string ds = d.string();
        lembed_multivector_options_t ou = lembed_multivector_options_default();
        ou.model_dir = ds.c_str();
        lembed_multivector_t* h = nullptr;
        ASSERT(lembed_multivector_create(&ou, &h) == LEMBED_OK, "create with an out-of-vocabulary skiplist word");
        if (h) {
            const char* t[] = { "hello \xE2\x98\x83 world" };
            lembed_multivectors_t r;
            ASSERT(lembed_multivector_embed(h, t, 1, LEMBED_MV_DOCUMENT, &r) == LEMBED_OK, "embed unk doc");
            /* [CLS] [D] hello [UNK] world [SEP] */
            ASSERT(r.total_vectors == 6, "unknown tokens are kept, not skipped");
            lembed_multivectors_free(&r);
            lembed_multivector_free(h);
        }
        fs::remove_all(d, ec);
    }

    /* Batch size does not change the vectors of a text. */
    {
        const char* two[] = { docs[0], docs[1] };
        lembed_multivector_options_t o1 = o;
        o1.batch_size = 1;
        lembed_multivector_t* one = nullptr;
        ASSERT(lembed_multivector_create(&o1, &one) == LEMBED_OK, "create batch_size 1");
        lembed_multivectors_t a, b;
        if (one && lembed_multivector_embed(mv, two, 2, LEMBED_MV_DOCUMENT, &a) == LEMBED_OK &&
            lembed_multivector_embed(one, two, 2, LEMBED_MV_DOCUMENT, &b) == LEMBED_OK) {
            ASSERT(a.total_vectors == b.total_vectors, "same vector count whatever the batching");
            float maxdiff = 0.0f;
            for (int64_t i = 0; i < a.total_vectors * a.dim && i < b.total_vectors * b.dim; i++)
                maxdiff = std::fmax(maxdiff, std::fabs(a.data[i] - b.data[i]));
            ASSERT(maxdiff < 1e-4f, "same vectors whatever the batching");
            lembed_multivectors_free(&a);
            lembed_multivectors_free(&b);
        } else {
            ASSERT(false, "batched / unbatched embed");
        }
        lembed_multivector_free(one);
    }

    lembed_stats_t st;
    lembed_multivector_stats(mv, &st);
    ASSERT(st.texts_embedded >= 6 && st.batches_run >= 1, "stats counted the calls");

    lembed_multivector_free(mv);
}

int main(void) {
    test_without_model();

    const std::string dir = model_dir();
    if (!dir.empty() && std::filesystem::exists(std::filesystem::path(dir) / "tokenizer.json")) {
        test_with_model(dir);
    } else {
        printf("SKIP: model directory not found (%s)\n", dir.c_str());
    }

    printf("%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
