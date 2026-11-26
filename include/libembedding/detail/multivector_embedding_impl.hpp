/*
 * libembedding - detail/multivector_embedding_impl.hpp
 * ONNX Runtime implementation of the multi-vector (ColBERT) handle.
 *
 * Same shape as the sparse ONNX context (OnnxSession + TokenizerWrapper,
 * call_mutex + stats_mutex, try/catch and set_error at the C boundary); what
 * differs is the text preparation (marker token, [MASK] query expansion,
 * skiplist) and the output, a ragged [token, dim] block per text.
 *
 * Model conventions come from onnx_config.json (or colbert_config.json) next to
 * the graph. Every field is optional; the defaults are those of the
 * answerdotai/answerai-colbert-small-v1 family and are listed in
 * docs/c_api/multivector_embedding.md.
 *
 * Auteur: David Orel
 * Version: 1.16.6
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_MULTIVECTOR_EMBEDDING_IMPL_HPP
#define LIBEMBEDDING_DETAIL_MULTIVECTOR_EMBEDDING_IMPL_HPP

#include "../error.h"
#include "../types.h"
#include "../model_registry.h"
#include "download_manager.hpp"
#include "model_loader_impl.hpp"
#include "onnx_session_impl.hpp"
#include "tokenizer_impl.hpp"
#include "sparse_batching.hpp"
#include "cJSON.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace lembed {
namespace detail {

/* Invalid or unusable model configuration: LEMBED_ERROR_IO (decision of the
 * ColBERT RFC, no dedicated status code). */
class mv_config_error : public std::runtime_error {
public:
    explicit mv_config_error(const std::string& what) : std::runtime_error(what) {}
};

/* A registry model that cannot be fetched: LEMBED_ERROR_DOWNLOAD. */
class mv_download_error : public std::runtime_error {
public:
    explicit mv_download_error(const std::string& what) : std::runtime_error(what) {}
};

/* The graph cannot serve this API (no projection): LEMBED_ERROR_UNSUPPORTED. */
class mv_unsupported_error : public std::runtime_error {
public:
    explicit mv_unsupported_error(const std::string& what) : std::runtime_error(what) {}
};

/* What the model directory declares. -1 / 0 mean "not declared". */
struct MvModelConfig {
    int query_prefix_id = -1;
    int document_prefix_id = -1;
    int mask_token_id = -1;
    int pad_token_id = -1;
    std::string query_prefix;       /* marker as a string, e.g. "[unused0]" or "[Q] " */
    std::string document_prefix;
    int query_length = 0;
    int document_length = 0;
    int do_query_expansion = -1;    /* -1 = undeclared */
    int attend_to_expansion = -1;
    bool has_skiplist = false;
    std::vector<std::string> skiplist_words;
    int embedding_dim = 0;
};

/* Documented defaults when onnx_config.json is absent or silent. */
constexpr int kMvDefaultQueryLength = 32;
constexpr int kMvDefaultDocLength = 300;
constexpr int kMvDefaultBatchSize = 32;

inline std::string mv_trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) b++;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) e--;
    return s.substr(b, e - b);
}

/* Upper bound for any length or id read from a configuration or an option: it
 * keeps a hostile file from asking for a multi-GB buffer. */
constexpr int kMvMaxLength = 8192;

/* Absent key: -1. Present but not a finite integer in [0, 1e9]: mv_config_error. */
inline int mv_json_int(const cJSON* root, const char* key) {
    const cJSON* it = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!it || cJSON_IsNull(it)) return -1;
    if (!cJSON_IsNumber(it))
        throw mv_config_error(std::string("multivector: \"") + key + "\" must be an integer");
    const double v = it->valuedouble;
    if (!(v >= 0.0 && v <= 1e9) || v != std::floor(v))
        throw mv_config_error(std::string("multivector: \"") + key +
                              "\" is not an integer in the supported range");
    return (int)v;
}

struct MvJsonGuard {
    cJSON* p;
    ~MvJsonGuard() { if (p) cJSON_Delete(p); }
};

inline int mv_json_bool(const cJSON* root, const char* key) {
    const cJSON* it = cJSON_GetObjectItemCaseSensitive(root, key);
    if (it && cJSON_IsBool(it)) return cJSON_IsTrue(it) ? 1 : 0;
    return -1;
}

inline std::string mv_json_string(const cJSON* root, const char* key) {
    const cJSON* it = cJSON_GetObjectItemCaseSensitive(root, key);
    return (it && cJSON_IsString(it) && it->valuestring) ? std::string(it->valuestring)
                                                         : std::string();
}

/* Throws mv_config_error on malformed JSON. */
inline MvModelConfig mv_parse_config(const std::string& json) {
    cJSON* root = cJSON_Parse(json.c_str());
    MvJsonGuard guard{root};
    if (!root || !cJSON_IsObject(root))
        throw mv_config_error("multivector: the model configuration is not a valid JSON object");
    MvModelConfig c;
    c.pad_token_id = mv_json_int(root, "pad_token_id");
    c.query_prefix_id = mv_json_int(root, "query_prefix_id");
    c.document_prefix_id = mv_json_int(root, "document_prefix_id");
    c.mask_token_id = mv_json_int(root, "mask_token_id");
    c.query_prefix = mv_json_string(root, "query_prefix");
    c.document_prefix = mv_json_string(root, "document_prefix");
    c.query_length = mv_json_int(root, "query_length");
    c.document_length = mv_json_int(root, "document_length");
    c.do_query_expansion = mv_json_bool(root, "do_query_expansion");
    c.attend_to_expansion = mv_json_bool(root, "attend_to_expansion_tokens");
    c.embedding_dim = mv_json_int(root, "embedding_dim");
    const cJSON* sk = cJSON_GetObjectItemCaseSensitive(root, "skiplist_words");
    if (sk && cJSON_IsArray(sk)) {
        c.has_skiplist = true;
        const cJSON* w = nullptr;
        cJSON_ArrayForEach(w, sk) {
            if (cJSON_IsString(w) && w->valuestring) c.skiplist_words.emplace_back(w->valuestring);
        }
    }
    if (c.query_length > kMvMaxLength || c.document_length > kMvMaxLength)
        throw mv_config_error("multivector: query_length / document_length above " +
                              std::to_string(kMvMaxLength));
    if (c.embedding_dim > 65536)
        throw mv_config_error("multivector: embedding_dim above 65536");
    return c;
}

/* The runtime: everything a handle holds. */
struct MvRuntime {
    OnnxSession session;
    TokenizerWrapper tokenizer;
    int dim = 0;
    int query_marker = -1;
    int doc_marker = -1;
    int mask_id = -1;
    int query_maxlen = kMvDefaultQueryLength;
    int doc_maxlen = kMvDefaultDocLength;
    bool query_expansion = true;
    bool attend_expansion = false;
    bool mask_punctuation = true;
    bool normalize = true;
    std::unordered_set<int> skip_ids;
};

} /* namespace detail */
} /* namespace lembed */

struct lembed_multivector {
    lembed::detail::MvRuntime rt;

    std::string model_name_str;
    int num_threads = 0;
    int batch_size = lembed::detail::kMvDefaultBatchSize;
    int max_tokens_per_batch = 0;
    lembed_execution_provider_t provider = LEMBED_PROVIDER_CPU;
    int device_id = 0;
    lembed_model_desc_t desc;

    uint64_t texts_embedded = 0;
    uint64_t batches_run = 0;
    double   total_latency_ms = 0.0;
    int      stats_calls = 0;

    /* The tokenizer keeps scratch strings, so calls on one handle are
     * serialised; stats_mutex guards the counters alone so the stats getter
     * never waits for an inference. Lock order: call_mutex, then stats_mutex. */
    std::mutex call_mutex;
    mutable std::mutex stats_mutex;
};

namespace lembed {
namespace detail {

inline bool mv_ends_with_gguf(const char* p) {
    const size_t n = strlen(p);
    if (n < 5) return false;
    const char* s = p + n - 5;
    return s[0] == '.' && (s[1] == 'g' || s[1] == 'G') && (s[2] == 'g' || s[2] == 'G') &&
           (s[3] == 'u' || s[3] == 'U') && (s[4] == 'f' || s[4] == 'F');
}

/* Resolves a marker: declared id first, then the declared string looked up in
 * the vocabulary, then `fallback_surface`. Returns -1 when nothing resolves. */
inline int mv_resolve_marker(const TokenizerWrapper& tok, int declared_id,
                             const std::string& declared_str,
                             const char* fallback_surface) {
    if (declared_id >= 0) return declared_id;
    if (!declared_str.empty()) {
        const int id = tok.token_to_id(mv_trim(declared_str));
        if (id >= 0) return id;
        /* A string marker the vocabulary cannot resolve is a configuration
         * error, not a reason to guess another marker. */
        return -1;
    }
    return tok.token_to_id(fallback_surface);
}

/* string.punctuation, the skiplist of the ColBERT family. */
inline std::vector<std::string> mv_default_skiplist() {
    static const char kPunct[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    std::vector<std::string> v;
    for (const char* p = kPunct; *p; p++) v.emplace_back(1, *p);
    return v;
}

inline int mv_word_to_id(const TokenizerWrapper& tok, const std::string& w) {
    int id = tok.token_to_id(w);
    if (id >= 0) return id;
    /* Not a vocabulary entry as such: encode it and take the single body
     * token, if the encoding has exactly one ([CLS] x [SEP]). */
    const std::vector<int> ids = tok.encode(w);
    /* [UNK] means "not in the vocabulary", not "this word": skipping it would
     * drop every unknown token of every document. */
    if (ids.size() == 3 && ids[1] != tok.unk_token_id()) return ids[1];
    return -1;
}

inline std::string mv_find_model_file(const std::string& dir) {
    static const char* const kCandidates[] = {
        "/model.onnx", "/vespa_colbert.onnx", "/onnx/model.onnx"
    };
    for (const char* c : kCandidates) {
        const std::string p = dir + c;
        if (file_exists(p)) return p;
    }
    throw std::runtime_error("multivector: no ONNX file (model.onnx, vespa_colbert.onnx, "
                             "onnx/model.onnx) in " + dir);
}

/* Builds the runtime from a directory. Throws; the caller maps the exception to
 * a status. */
inline void mv_load(MvRuntime& rt, const std::string& dir, const lembed_multivector_options_t& o) {
    const std::string onnx_path = mv_find_model_file(dir);
    const std::string tok_path = dir + "/tokenizer.json";
    if (!file_exists(tok_path))
        throw std::runtime_error("multivector: tokenizer.json not found in " + dir);

    MvModelConfig cfg;
    bool have_cfg = false;
    for (const char* name : {"/onnx_config.json", "/colbert_config.json"}) {
        const std::string p = dir + name;
        if (file_exists(p)) {
            cfg = mv_parse_config(read_file_to_string(p));
            have_cfg = true;
            break;
        }
    }
    (void)have_cfg;

    rt.query_maxlen = o.query_maxlen > 0 ? o.query_maxlen
                    : (cfg.query_length > 0 ? cfg.query_length : kMvDefaultQueryLength);
    rt.doc_maxlen = o.doc_maxlen > 0 ? o.doc_maxlen
                  : (cfg.document_length > 0 ? cfg.document_length : kMvDefaultDocLength);
    /* [CLS] marker [SEP] is the shortest sequence the pipeline can build. */
    if (rt.query_maxlen < 3 || rt.doc_maxlen < 3)
        throw mv_config_error("multivector: query_maxlen / doc_maxlen must be at least 3");

    /* Tokenizer and ids first: a bad configuration is reported before the
     * (heavy) graph is loaded. */
    rt.tokenizer.load_from_file(tok_path, std::max(rt.query_maxlen, rt.doc_maxlen));

    rt.query_marker = mv_resolve_marker(rt.tokenizer, cfg.query_prefix_id, cfg.query_prefix, "[unused0]");
    rt.doc_marker = mv_resolve_marker(rt.tokenizer, cfg.document_prefix_id, cfg.document_prefix, "[unused1]");
    if (rt.query_marker < 0 || rt.doc_marker < 0)
        throw mv_config_error("multivector: cannot resolve the query / document marker token "
                              "(declare query_prefix_id / document_prefix_id in onnx_config.json)");

    rt.mask_id = cfg.mask_token_id >= 0 ? cfg.mask_token_id : rt.tokenizer.token_to_id("[MASK]");

    /* Declared ids index the embedding table: they must be inside the vocabulary. */
    {
        const int vs = rt.tokenizer.vocab_size();
        const struct { const char* name; int id; } ids_to_check[] = {
            { "query_prefix_id", rt.query_marker }, { "document_prefix_id", rt.doc_marker },
            { "mask_token_id", rt.mask_id }, { "pad_token_id", cfg.pad_token_id },
        };
        for (const auto& c : ids_to_check) {
            if (c.id >= 0 && vs > 0 && c.id >= vs)
                throw mv_config_error(std::string("multivector: ") + c.name + " " +
                                      std::to_string(c.id) + " is outside the vocabulary (" +
                                      std::to_string(vs) + " entries)");
        }
    }

    rt.session.load_from_file(onnx_path.c_str(), o.num_threads, (int)o.provider);

    /* Expansion: an explicit option wins, else ON (the ColBERT convention).
     * onnx_config.json's do_query_expansion is deliberately not consulted: the
     * official answerai-colbert-small-v1 file declares false, yet pylate, the
     * reference, pads its queries with [MASK] up to query_length and keeps those
     * vectors (measured: per-token cosine 0.999995 with expansion, no match
     * without, tests/test_multivector_fidelity.cpp). */
    rt.query_expansion = o.query_expansion >= 0 ? o.query_expansion != 0 : true;
    rt.attend_expansion = o.attend_to_expansion >= 0 ? o.attend_to_expansion != 0
                        : (cfg.attend_to_expansion >= 0 ? cfg.attend_to_expansion != 0
                                                        : false);
    if (rt.query_expansion && rt.mask_id < 0)
        throw mv_config_error("multivector: query expansion needs a [MASK] token id "
                              "(declare mask_token_id in onnx_config.json)");

    rt.mask_punctuation = o.mask_punctuation >= 0 ? o.mask_punctuation != 0 : true;
    if (rt.mask_punctuation) {
        const std::vector<std::string> words =
            cfg.has_skiplist ? cfg.skiplist_words : mv_default_skiplist();
        for (const std::string& w : words) {
            const int id = mv_word_to_id(rt.tokenizer, w);
            if (id >= 0) rt.skip_ids.insert(id);
        }
    }
    rt.normalize = o.normalize != 0;

    /* Probe: [CLS] marker [SEP] gives the output dimension, and fails here
     * rather than on the first user call if the graph is not token-level. */
    const int64_t ids[3] = { rt.tokenizer.cls_token_id(), rt.doc_marker, rt.tokenizer.sep_token_id() };
    const int64_t mask[3] = { 1, 1, 1 };
    const int64_t types[3] = { 0, 0, 0 };
    auto out = rt.session.run(ids, mask, types, 1, 3, 0);
    if (out.empty() || out[0].shape.size() != 3 || out[0].shape[2] <= 0)
        throw mv_config_error("multivector: the graph output is not [batch, tokens, dim]");
    rt.dim = (int)out[0].shape[2];
    if (cfg.embedding_dim > 0 && cfg.embedding_dim != rt.dim)
        throw mv_unsupported_error(
            "multivector: the graph outputs dimension " + std::to_string(rt.dim) +
            " but the configuration declares " + std::to_string(cfg.embedding_dim) +
            " (projection layer missing from the ONNX graph)");
}

inline void mv_l2_normalize(float* v, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; i++) s += (double)v[i] * (double)v[i];
    if (s <= 0.0) return;
    const float inv = (float)(1.0 / std::sqrt(s));
    for (int i = 0; i < dim; i++) v[i] *= inv;
}

/* One prepared text: the ids fed to the graph, how many are real tokens, and
 * how many of the leading positions attend. */
struct MvPrepared {
    std::vector<int> ids;
    int attend = 0;      /* positions [0, attend) have attention 1 */
    int real = 0;        /* positions [0, real) are text tokens; the rest are [MASK] pads */
};

inline MvPrepared mv_prepare(const MvRuntime& rt, const char* text, bool query) {
    const int maxlen = query ? rt.query_maxlen : rt.doc_maxlen;
    MvPrepared p;
    p.ids = rt.tokenizer.encode(text ? text : "");
    /* Truncate to maxlen - 1 so that the marker brings it back to maxlen; the
     * last token stays [SEP]. */
    if ((int)p.ids.size() > maxlen - 1) {
        p.ids.resize((size_t)maxlen - 1);
        p.ids.back() = rt.tokenizer.sep_token_id();
    }
    p.ids.insert(p.ids.begin() + (p.ids.empty() ? 0 : 1), query ? rt.query_marker : rt.doc_marker);
    p.real = (int)p.ids.size();
    p.attend = p.real;
    if (query && rt.query_expansion && (int)p.ids.size() < maxlen) {
        p.ids.resize((size_t)maxlen, rt.mask_id);
        if (rt.attend_expansion) p.attend = maxlen;
    }
    return p;
}

} /* namespace detail */
} /* namespace lembed */

#ifdef __cplusplus
extern "C" {
#endif

lembed_multivector_options_t lembed_multivector_options_default(void) {
    lembed_multivector_options_t o;
    memset(&o, 0, sizeof(o));
    o.provider = LEMBED_PROVIDER_CPU;
    o.normalize = 1;
    o.mask_punctuation = -1;
    o.query_expansion = -1;
    o.attend_to_expansion = -1;
    return o;
}

lembed_status_t lembed_multivector_create(const lembed_multivector_options_t* options,
                                          lembed_multivector_t** out) {
    if (!options || !out) {
        lembed::detail::set_error("multivector: create needs options and an output pointer");
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (!options->model_dir || !options->model_dir[0]) {
        lembed::detail::set_error("multivector: model_dir is empty");
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }
    if (options->batch_size < 0 || options->max_tokens_per_batch < 0 ||
        options->query_maxlen < 0 || options->doc_maxlen < 0 || options->num_threads < 0) {
        lembed::detail::set_error("multivector: negative option");
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }
    if (options->query_maxlen > lembed::detail::kMvMaxLength ||
        options->doc_maxlen > lembed::detail::kMvMaxLength) {
        lembed::detail::set_error("multivector: query_maxlen / doc_maxlen above " +
                                  std::to_string(lembed::detail::kMvMaxLength));
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }
    if (lembed::detail::mv_ends_with_gguf(options->model_dir)) {
        lembed::detail::set_error("multivector: GGUF is not supported, only ONNX model directories");
        return LEMBED_ERROR_UNSUPPORTED;
    }
    {
        /* Validate on the raw value first (no enum-typed read of a bad value). */
        lembed_status_t ps = lembed::detail::check_provider(options->provider);
        if (ps != LEMBED_OK) return ps;
        ps = lembed::detail::onnx_provider_precheck((int)options->provider);
        if (ps != LEMBED_OK) return ps;
    }

    try {
        std::string dir(options->model_dir);
        std::string display_name;
        {
            /* A string that is not an existing directory but names a registry
             * model is fetched into the cache. An existing directory always
             * wins, so a local copy can shadow the registry. */
            std::error_code ec;
            if (!std::filesystem::is_directory(std::filesystem::path(dir), ec)) {
                const int idx = lembed_find_multivector_model_by_code(dir.c_str());
                int model_idx = idx;
                if (model_idx < 0) {
                    const lembed_model_info_t* models = nullptr;
                    int count = 0;
                    lembed_list_multivector_models(&models, &count);
                    for (int i = 0; i < count; i++) {
                        if (dir == models[i].model_name) { model_idx = i; break; }
                    }
                }
                if (model_idx >= 0) {
                    lembed_model_info_t info;
                    lembed_get_multivector_model_info((lembed_multivector_model_t)model_idx, &info);
                    display_name = info.model_code;
                    try {
                        dir = lembed::detail::ensure_model_by_kind(
                            model_idx, LEMBED__MODEL_TYPE_MULTIVECTOR, options->cache_dir,
                            options->show_download_progress != 0, options->offline != 0);
                    } catch (const lembed::detail::model_refused_error&) {
                        throw;
                    } catch (const std::exception& e) {
                        throw lembed::detail::mv_download_error(e.what());
                    }
                }
            }
        }
        if (!lembed::detail::file_exists(dir + "/tokenizer.json") &&
            !lembed::detail::file_exists(dir + "/model.onnx") &&
            !lembed::detail::file_exists(dir + "/vespa_colbert.onnx") &&
            !lembed::detail::file_exists(dir + "/onnx/model.onnx")) {
            lembed::detail::set_error("multivector: model directory not found or empty: " + dir);
            return LEMBED_ERROR_IO;
        }

        struct Deleter { void operator()(lembed_multivector_t* p) const { delete p; } };
        std::unique_ptr<lembed_multivector_t, Deleter> mv(new lembed_multivector());
        lembed::detail::mv_load(mv->rt, dir, *options);
        mv->model_name_str = display_name.empty() ? dir : display_name;
        mv->num_threads = options->num_threads;
        mv->batch_size = options->batch_size > 0 ? options->batch_size
                                                 : lembed::detail::kMvDefaultBatchSize;
        mv->max_tokens_per_batch = options->max_tokens_per_batch;
        mv->provider = options->provider;
        mv->device_id = options->device_id;

        memset(&mv->desc, 0, sizeof(mv->desc));
        mv->desc.name = mv->model_name_str.c_str();
        mv->desc.dimension = mv->rt.dim;
        mv->desc.max_length = mv->rt.doc_maxlen;
        mv->desc.pooling = 0;
        mv->desc.num_threads = mv->num_threads;
        mv->desc.batch_size = mv->batch_size;
        mv->desc.provider = mv->provider;
        mv->desc.device_id = mv->device_id;

        *out = mv.release();
        return LEMBED_OK;
    } catch (const lembed::detail::mv_unsupported_error& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_UNSUPPORTED;
    } catch (const lembed::detail::model_refused_error& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_UNSUPPORTED;
    } catch (const lembed::detail::mv_download_error& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return lembed::detail::onnx_create_status(e, LEMBED_ERROR_IO);
    } catch (...) {
        lembed::detail::set_error("unknown error while creating the multivector embedding");
        return LEMBED_ERROR_IO;
    }
}

void lembed_multivectors_free(lembed_multivectors_t* result) {
    if (!result) return;
    free(result->data);
    free(result->offsets);
    result->data = nullptr;
    result->offsets = nullptr;
    result->num_docs = 0;
    result->dim = 0;
    result->total_vectors = 0;
}

lembed_status_t lembed_multivector_embed(lembed_multivector_t* mv,
                                         const char* const* texts, int n,
                                         lembed_mv_mode_t mode,
                                         lembed_multivectors_t* out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!mv || !texts || n <= 0 || !out) {
        lembed::detail::set_error("multivector: embed needs a handle, texts, n > 0 and an output");
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }
    /* Raw read: `mode` comes from a C caller and may be out of range. */
    if (lembed::detail::check_enum(mode, LEMBED_MV_DOCUMENT, LEMBED_MV_QUERY,
                                   "multivector mode") != LEMBED_OK) {
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> call_lock(mv->call_mutex);
    try {
        using namespace lembed::detail;
        const MvRuntime& rt = mv->rt;
        const bool query = (mode == LEMBED_MV_QUERY);
        const int dim = rt.dim;
        const auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<MvPrepared> prep((size_t)n);
        std::vector<std::vector<int>> id_lists((size_t)n);
        for (int i = 0; i < n; i++) {
            prep[(size_t)i] = mv_prepare(rt, texts[i], query);
            id_lists[(size_t)i] = prep[(size_t)i].ids;
        }

        const int batch_size = mv->batch_size;
        const int token_budget = mv->max_tokens_per_batch > 0 ? mv->max_tokens_per_batch
                                                              : kDenseMaxBatchTokens;
        const std::vector<size_t> order = sparse_batch_order(id_lists, batch_size);

        /* The number of vectors of every text is known before inference (it
         * depends on the ids only), so the result is allocated once, exactly,
         * and written in place: no per-text staging copy. */
        std::vector<int64_t> counts((size_t)n, 0);
        int64_t total = 0;
        for (int i = 0; i < n; i++) {
            const MvPrepared& p = prep[(size_t)i];
            const int limit = (query && rt.query_expansion) ? (int)p.ids.size() : p.real;
            int64_t c = 0;
            for (int j = 0; j < limit; j++) {
                if (!query && rt.mask_punctuation && rt.skip_ids.count(p.ids[(size_t)j])) continue;
                c++;
            }
            counts[(size_t)i] = c;
            total += c;
        }
        {
            const uint64_t row_bytes = (uint64_t)dim * sizeof(float);
            const uint64_t cap = (uint64_t)SIZE_MAX / row_bytes;
            if ((uint64_t)total > cap || (uint64_t)n > (uint64_t)SIZE_MAX / sizeof(int64_t) - 1) {
                set_error("multivector: result too large");
                return LEMBED_ERROR_OUT_OF_MEMORY;
            }
        }
        out->offsets = (int64_t*)malloc(((size_t)n + 1) * sizeof(int64_t));
        out->data = (float*)malloc((size_t)(total > 0 ? total : 1) * (size_t)dim * sizeof(float));
        if (!out->offsets || !out->data) {
            lembed_multivectors_free(out);
            set_error("multivector: out of memory");
            return LEMBED_ERROR_OUT_OF_MEMORY;
        }
        out->offsets[0] = 0;
        for (int i = 0; i < n; i++) out->offsets[i + 1] = out->offsets[i] + counts[(size_t)i];
        out->num_docs = n;
        out->dim = dim;
        out->total_vectors = total;
        int num_batches = 0;

        for (size_t start = 0; start < order.size();) {
            const size_t end = sparse_batch_end(start, order.size(),
                                                id_lists[order[start]].size(),
                                                batch_size, token_budget);
            const int bsz = (int)(end - start);
            const int seq = (int)id_lists[order[start]].size();   /* longest first */
            num_batches++;

            std::vector<int64_t> ids_flat((size_t)bsz * seq, (int64_t)rt.tokenizer.pad_token_id());
            std::vector<int64_t> mask_flat((size_t)bsz * seq, 0);
            std::vector<int64_t> type_flat((size_t)bsz * seq, 0);
            for (int b = 0; b < bsz; b++) {
                const MvPrepared& p = prep[order[start + (size_t)b]];
                const size_t base = (size_t)b * seq;
                for (int j = 0; j < (int)p.ids.size(); j++) {
                    ids_flat[base + j] = p.ids[(size_t)j];
                    mask_flat[base + j] = j < p.attend ? 1 : 0;
                }
            }

            auto outputs = mv->rt.session.run(ids_flat.data(), mask_flat.data(), type_flat.data(),
                                              bsz, seq, 0);
            /* The graph's own output extent is the stride, and the buffer is
             * checked against it before any row is read. */
            if (outputs.empty() || outputs[0].shape.size() != 3 ||
                outputs[0].shape[0] != bsz || outputs[0].shape[2] != dim ||
                outputs[0].shape[1] < seq ||
                outputs[0].data.size() < (size_t)bsz * (size_t)outputs[0].shape[1] * (size_t)dim)
                throw std::runtime_error("multivector: unexpected output shape from the graph");
            const float* o = outputs[0].data.data();
            const size_t stride = (size_t)outputs[0].shape[1];

            for (int b = 0; b < bsz; b++) {
                const size_t slot = order[start + (size_t)b];
                const MvPrepared& p = prep[slot];
                /* Which positions survive: a query keeps every position when
                 * expanded (the [MASK] pads included), else its real tokens; a
                 * document keeps its real tokens minus the skiplist. */
                const int limit = (query && rt.query_expansion) ? (int)p.ids.size() : p.real;
                float* dst = out->data + (size_t)out->offsets[slot] * (size_t)dim;
                for (int j = 0; j < limit; j++) {
                    if (!query && rt.mask_punctuation && rt.skip_ids.count(p.ids[(size_t)j])) continue;
                    const float* row = o + ((size_t)b * stride + (size_t)j) * dim;
                    memcpy(dst, row, (size_t)dim * sizeof(float));
                    dst += dim;
                }
            }
            start = end;
        }

        if (rt.normalize) {
            for (int64_t v = 0; v < total; v++) mv_l2_normalize(out->data + (size_t)v * dim, dim);
        }

        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        std::lock_guard<std::mutex> stats_lock(mv->stats_mutex);
        mv->texts_embedded += (uint64_t)n;
        mv->batches_run += (uint64_t)num_batches;
        mv->total_latency_ms += elapsed_ms;
        mv->stats_calls++;
        return LEMBED_OK;
    } catch (const std::bad_alloc&) {
        lembed_multivectors_free(out);
        lembed::detail::set_error("multivector: out of memory");
        return LEMBED_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        lembed_multivectors_free(out);
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    } catch (...) {
        lembed_multivectors_free(out);
        lembed::detail::set_error("unknown error while embedding");
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

int lembed_multivector_dim(const lembed_multivector_t* mv) {
    return mv ? mv->rt.dim : 0;
}

const lembed_model_desc_t* lembed_multivector_desc(const lembed_multivector_t* mv) {
    return mv ? &mv->desc : nullptr;
}

const char* lembed_multivector_backend_name(const lembed_multivector_t* mv) {
    return mv ? "onnxruntime" : nullptr;
}

void lembed_multivector_stats(const lembed_multivector_t* mv, lembed_stats_t* out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!mv) return;
    std::lock_guard<std::mutex> stats_lock(mv->stats_mutex);
    out->texts_embedded = mv->texts_embedded;
    out->batches_run = mv->batches_run;
    out->avg_latency_ms = mv->stats_calls > 0 ? mv->total_latency_ms / (double)mv->stats_calls : 0.0;
}

void lembed_multivector_free(lembed_multivector_t* mv) {
    delete mv;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_DETAIL_MULTIVECTOR_EMBEDDING_IMPL_HPP */
