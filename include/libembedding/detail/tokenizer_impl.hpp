/*
 * libembedding - detail/tokenizer_impl.hpp
 * Built-in HuggingFace tokenizer.json parser (WordPiece + BPE)
 * No external Rust/tokenizers-cpp dependency required.
 *
 * Supports the tokenizer types used by embedding models in the registry:
 * - WordPiece (BERT-style: bge, MiniLM, mpnet, etc.)
 * - BPE (GPT/Sentencepiece-style: nomic, jina, CLIP, etc.)
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP

#include "cJSON.h"

#include <climits>

#if defined(_WIN32) || defined(WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lembed { namespace detail {

struct EncodingBatch {
    std::vector<std::vector<int64_t>> input_ids;
    std::vector<std::vector<int64_t>> attention_mask;
    std::vector<std::vector<int64_t>> token_type_ids;
    int seq_length;
};

/* Simple Unicode-aware lowercasing for ASCII range */
static inline std::string to_lower_ascii(const std::string& s) {
    std::string r = s;
    for (auto& c : r) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
    return r;
}

/* =========================================================================
 * HF-faithful normalisation for BertNormalizer
 *
 * Why this exists: the reference SPLADE model is a HuggingFace
 * BertTokenizer, and its `tokenize()` does, per whitespace-delimited token,
 *
 *     token = token.lower()                  # Python str.lower() semantics
 *     token = _run_strip_accents(token)      # NFD, then drop the Mn category
 *
 * Without the accent strip this tokeniser disagreed with HF on accented text
 * while llama.cpp's own WPM tokenizer stripped them -- so the two backends were
 * not comparable, and any "GGUF vs ONNX" figure measured the tokeniser gap
 * rather than the sparse implementation.
 *
 * Two details of the reference implementation are easy to get wrong, and both
 * were wrong in a first draft of this code:
 *
 *  - Lowercase happens **before** the accent strip, not after.
 *  - The strip is a real NFD followed by dropping combining marks. Letters with
 *    no canonical decomposition -- AE, OE, Eth, Thorn, D-stroke, L-stroke --
 *    are therefore NOT folded, and end up as [UNK] exactly as HF leaves them.
 *    Only "ss"-style output for sharp-s is real, and it comes from the
 *    decomposition U+00DF -> U+0073 U+0303, not from a ligature rule.
 *
 * Scope, stated plainly: this covers Latin-1 Supplement (U+00C0-U+00FF) and
 * Latin Extended-A (U+0100-U+017F), which is where French, Spanish, Portuguese,
 * German, Italian, Polish, Czech and Turkish live. Codepoints outside those
 * ranges pass through unchanged, as they did before, so Greek and Cyrillic are
 * still not lowercased here. That is a documented gap rather than a silent one:
 * closing it means adding tables, not changing the shape.
 * ========================================================================= */

/* Base ASCII letter of a Latin-1 Supplement codepoint, or -1 when the codepoint
 * has no canonical decomposition (AE, OE, Eth, Thorn, D-stroke, O-stroke,
 * sharp-s). The sentinel has to be negative: 0 would be indistinguishable from a
 * successful fold to NUL, and the first draft of this code had exactly that
 * bug -- it silently turned non-decomposable letters into NUL bytes. */
static inline int latin1_base(uint32_t cp) {
    /* Rows are the uppercase half U+00C0-U+00DE and the lowercase half
     * U+00E0-U+00FF; both fold to the same base. -1 means "leave it alone",
     * which is what HF does for a letter with no decomposition. */
    static const int kBases[64] = {
        /* C0 A-grave   */ 'a', /* C1 A-acute   */ 'a',
        /* C2 A-circum  */ 'a', /* C3 A-tilde   */ 'a',
        /* C4 A-diaer   */ 'a', /* C5 A-ring    */ 'a',
        /* C6 AE        */  -1, /* C7 C-cedilla */ 'c',
        /* C8 E-grave   */ 'e', /* C9 E-acute   */ 'e',
        /* CA E-circum  */ 'e', /* CB E-diaer   */ 'e',
        /* CC I-grave   */ 'i', /* CD I-acute   */ 'i',
        /* CE I-circum  */ 'i', /* CF I-diaer   */ 'i',
        /* D0 Eth       */  -1, /* D1 N-tilde   */ 'n',
        /* D2 O-grave   */ 'o', /* D3 O-acute   */ 'o',
        /* D4 O-circum  */ 'o', /* D5 O-tilde   */ 'o',
        /* D6 O-diaer   */ 'o', /* D7 mult sign */  -1,
        /* D8 O-stroke  */  -1, /* D9 U-grave   */ 'u',
        /* DA U-acute   */ 'u', /* DB U-circum  */ 'u',
        /* DC U-diaer   */ 'u', /* DD Y-acute   */ 'y',
        /* DE Thorn     */  -1, /* DF sharp-s   */  -1,
        /* E0 a-grave   */ 'a', /* E1 a-acute   */ 'a',
        /* E2 a-circum  */ 'a', /* E3 a-tilde   */ 'a',
        /* E4 a-diaer   */ 'a', /* E5 a-ring    */ 'a',
        /* E6 ae        */  -1, /* E7 c-cedilla */ 'c',
        /* E8 e-grave   */ 'e', /* E9 e-acute   */ 'e',
        /* EA e-circum  */ 'e', /* EB e-diaer   */ 'e',
        /* EC i-grave   */ 'i', /* ED i-acute   */ 'i',
        /* EE i-circum  */ 'i', /* EF i-diaer   */ 'i',
        /* F0 eth       */  -1, /* F1 n-tilde   */ 'n',
        /* F2 o-grave   */ 'o', /* F3 o-acute   */ 'o',
        /* F4 o-circum  */ 'o', /* F5 o-tilde   */ 'o',
        /* F6 o-diaer   */ 'o', /* F7 division  */  -1,
        /* F8 o-stroke  */  -1, /* F9 u-grave   */ 'u',
        /* FA u-acute   */ 'u', /* FB u-circum  */ 'u',
        /* FC u-diaer   */ 'u', /* FD y-acute   */ 'y',
        /* FE thorn     */  -1, /* FF y-diaer   */ 'y',
    };
    if (cp >= 0x00C0 && cp <= 0x00FF) return kBases[cp - 0x00C0];
    return -1;
}

/* Latin Extended-A is a regular block: even codepoint is the uppercase base,
 * odd is that base plus a diacritic. Reducing to the base and reusing the
 * Latin-1 table is therefore equivalent to NFD for the folded subset. The
 * exceptions are the letters with no canonical decomposition, which HF leaves
 * untouched. */
static inline int extended_a_base(uint32_t cp) {
    switch (cp) {
        /* No canonical decomposition: HF keeps these verbatim. */
        case 0x0110: case 0x0111:   /* D with stroke   */
        case 0x0126: case 0x0127:   /* H with stroke   */
        case 0x0132: case 0x0133:   /* IJ ligature     */
        case 0x013F: case 0x0140:   /* L with middle dot */
        case 0x014A: case 0x014B:   /* ENG             */
        case 0x0152: case 0x0153:   /* OE ligature     */
        case 0x0166: case 0x0167:   /* T with stroke   */
            return -1;
        default:
            break;
    }
    /* Turkish dotted/dotless I and long-s are not part of the even/odd
     * pattern but do decompose, and both fold to plain "i"/"s". */
    if (cp == 0x0130 || cp == 0x0131) return 'i';
    if (cp == 0x017F) return 's';   /* long s -> s with stroke above -> s */
    const uint32_t base = (cp % 2 == 0) ? cp : (cp - 1);
    if (base >= 0x00C0 && base <= 0x00FF) return latin1_base(base);
    return -1;
}

/* Decode one UTF-8 codepoint. Returns bytes consumed and stores the codepoint;
 * malformed bytes are passed through so nothing is silently dropped. */
static inline size_t utf8_next(const std::string& s, size_t i, uint32_t& cp) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        cp = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        cp = ((uint32_t)(c & 0x0F) << 12) |
             (((unsigned char)s[i + 1] & 0x3F) << 6) |
             ((unsigned char)s[i + 2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        cp = ((uint32_t)(c & 0x07) << 18) |
             (((unsigned char)s[i + 1] & 0x3F) << 12) |
             (((unsigned char)s[i + 2] & 0x3F) << 6) |
             ((unsigned char)s[i + 3] & 0x3F);
        return 4;
    }
    cp = c;
    return 1;
}

/* BertNormalizer-equivalent, in HF's order: lowercase, then strip accents. */
static inline std::string bert_normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        const size_t n = utf8_next(s, i, cp);
        i += n;

        /* Step 1: lowercase. ASCII plus the Latin ranges we fold below; other
         * scripts keep their case, which is the documented limit. */
        if (cp >= 'A' && cp <= 'Z') {
            out += (char)(cp - 'A' + 'a');
            continue;
        }
        if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7) {
            const int b = latin1_base(cp);
            if (b >= 0) { out += (char)b; continue; }
            out.append(s, i - n, n);
            continue;
        }

        /* Step 2: strip accents, i.e. decompose and drop the Mn category. */
        int folded = -1;
        if (cp >= 0x00C0 && cp <= 0x00FF) folded = latin1_base(cp);
        else if (cp >= 0x0100 && cp <= 0x017F) folded = extended_a_base(cp);
        if (folded >= 0) {
            out += (char)folded;
            continue;
        }

        out.append(s, i - n, n);
    }
    return out;
}

/* Basic whitespace + punctuation pre-tokenization (BERT-style) */
static inline std::vector<std::string> basic_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        if (c <= 0x20) {
            /* whitespace */
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            i++;
        } else if ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
                   (c >= '[' && c <= '`') || (c >= '{' && c <= '~')) {
            /* punctuation — separate token */
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            current += (char)c;
            tokens.push_back(current);
            current.clear();
            i++;
        } else if (c >= 0x80) {
            /* multi-byte UTF-8 — keep as single token */
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            int bytes = 1;
            if ((c & 0xE0) == 0xC0) bytes = 2;
            else if ((c & 0xF0) == 0xE0) bytes = 3;
            else if ((c & 0xF8) == 0xF0) bytes = 4;
            for (int b = 0; b < bytes && i < text.size(); b++, i++)
                current += text[i];
            tokens.push_back(current);
            current.clear();
        } else {
            current += (char)c;
            i++;
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

class TokenizerWrapper {
public:
    enum Type { WORDPIECE, BPE };

    TokenizerWrapper()
        : max_length_(512), pad_token_id_(0), cls_token_id_(101),
          sep_token_id_(102), unk_token_id_(100), type_(WORDPIECE),
          do_lower_case_(true), add_special_tokens_(true) {}

    void load_from_file(const std::string& path, int max_length = 512) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open tokenizer file: " + path);
        std::string blob((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        load_from_blob(blob, max_length);
    }

    void load_from_blob(const std::string& json_blob, int max_length = 512) {
        max_length_ = max_length;

        cJSON* root = cJSON_Parse(json_blob.c_str());
        if (!root)
            throw std::runtime_error("Failed to parse tokenizer.json");

        /* Detect model type */
        cJSON* model = cJSON_GetObjectItem(root, "model");
        if (model) {
            cJSON* mtype = cJSON_GetObjectItem(model, "type");
            if (mtype && mtype->valuestring) {
                std::string t = mtype->valuestring;
                if (t == "BPE" || t == "bpe") type_ = BPE;
                else type_ = WORDPIECE;
            }

            /* Load vocabulary */
            cJSON* vocab = cJSON_GetObjectItem(model, "vocab");
            if (vocab) {
                cJSON* item = vocab->child;
                while (item) {
                    if (item->string && cJSON_IsNumber(item)) {
                        vocab_[item->string] = item->valueint;
                        if ((int)id_to_token_.size() <= item->valueint)
                            id_to_token_.resize(item->valueint + 1);
                        id_to_token_[item->valueint] = item->string;
                    }
                    item = item->next;
                }
            }

            /* BPE merges */
            if (type_ == BPE) {
                cJSON* merges = cJSON_GetObjectItem(model, "merges");
                if (merges && cJSON_IsArray(merges)) {
                    int n = cJSON_GetArraySize(merges);
                    for (int i = 0; i < n; i++) {
                        cJSON* m = cJSON_GetArrayItem(merges, i);
                        if (m && m->valuestring) {
                            bpe_merges_.push_back(m->valuestring);
                            bpe_ranks_[m->valuestring] = i;
                        }
                    }
                }
            }

            /* WordPiece continuing_subword_prefix */
            cJSON* prefix = cJSON_GetObjectItem(model, "continuing_subword_prefix");
            if (prefix && prefix->valuestring)
                wp_prefix_ = prefix->valuestring;
            else
                wp_prefix_ = "##"; /* BERT default */

            /* unk_token */
            cJSON* unk = cJSON_GetObjectItem(model, "unk_token");
            if (unk && unk->valuestring) {
                auto it = vocab_.find(unk->valuestring);
                if (it != vocab_.end()) unk_token_id_ = it->second;
            }
        }

        /* Normalizer: detect do_lower_case */
        cJSON* normalizer = cJSON_GetObjectItem(root, "normalizer");
        if (normalizer) {
            cJSON* ntype = cJSON_GetObjectItem(normalizer, "type");
            if (ntype && ntype->valuestring) {
                std::string nt = ntype->valuestring;
                if (nt == "BertNormalizer" || nt == "Lowercase") {
                    cJSON* lc = cJSON_GetObjectItem(normalizer, "lowercase");
                    do_lower_case_ = (!lc || cJSON_IsTrue(lc));
                } else if (nt == "Sequence") {
                    cJSON* normalizers = cJSON_GetObjectItem(normalizer, "normalizers");
                    if (normalizers && cJSON_IsArray(normalizers)) {
                        int n = cJSON_GetArraySize(normalizers);
                        for (int i = 0; i < n; i++) {
                            cJSON* sub = cJSON_GetArrayItem(normalizers, i);
                            cJSON* st = cJSON_GetObjectItem(sub, "type");
                            if (st && st->valuestring && strcmp(st->valuestring, "Lowercase") == 0)
                                do_lower_case_ = true;
                        }
                    }
                } else {
                    do_lower_case_ = false;
                }
            }
        } else {
            do_lower_case_ = false;
        }

        /* Post-processor: detect special tokens to add */
        cJSON* post = cJSON_GetObjectItem(root, "post_processor");
        if (post) {
            cJSON* ptype = cJSON_GetObjectItem(post, "type");
            if (ptype && ptype->valuestring &&
                strcmp(ptype->valuestring, "TemplateProcessing") == 0) {
                /* Check for [CLS] and [SEP] in template */
                add_special_tokens_ = true;
            }
        }

        /* Added tokens (for [CLS], [SEP], [PAD], [UNK], etc.) */
        cJSON* added = cJSON_GetObjectItem(root, "added_tokens");
        if (added && cJSON_IsArray(added)) {
            int n = cJSON_GetArraySize(added);
            for (int i = 0; i < n; i++) {
                cJSON* at = cJSON_GetArrayItem(added, i);
                cJSON* content = cJSON_GetObjectItem(at, "content");
                cJSON* id = cJSON_GetObjectItem(at, "id");
                if (content && content->valuestring && id && cJSON_IsNumber(id)) {
                    std::string tok = content->valuestring;
                    int tid = id->valueint;
                    vocab_[tok] = tid;
                    if ((int)id_to_token_.size() <= tid)
                        id_to_token_.resize(tid + 1);
                    id_to_token_[tid] = tok;

                    if (tok == "[CLS]" || tok == "<s>") cls_token_id_ = tid;
                    if (tok == "[SEP]" || tok == "</s>") sep_token_id_ = tid;
                    if (tok == "[PAD]" || tok == "<pad>") pad_token_id_ = tid;
                    if (tok == "[UNK]" || tok == "<unk>") unk_token_id_ = tid;
                }
            }
        }

        /* Padding config */
        cJSON* padding = cJSON_GetObjectItem(root, "padding");
        if (padding) {
            cJSON* pid = cJSON_GetObjectItem(padding, "pad_id");
            if (pid && cJSON_IsNumber(pid)) pad_token_id_ = pid->valueint;
        }

        cJSON_Delete(root);
    }

    EncodingBatch encode_batch(const std::vector<std::string>& texts) const {
        EncodingBatch result;
        result.seq_length = 0;

        std::vector<std::vector<int>> all_ids;
        for (const auto& text : texts) {
            auto ids = encode_single(text);
            if ((int)ids.size() > result.seq_length)
                result.seq_length = (int)ids.size();
            all_ids.push_back(std::move(ids));
        }

        /* Cap at max_length */
        if (result.seq_length > max_length_)
            result.seq_length = max_length_;

        int batch_size = (int)texts.size();
        result.input_ids.resize(batch_size);
        result.attention_mask.resize(batch_size);
        result.token_type_ids.resize(batch_size);

        for (int i = 0; i < batch_size; i++) {
            int len = std::min((int)all_ids[i].size(), result.seq_length);
            result.input_ids[i].resize(result.seq_length, (int64_t)pad_token_id_);
            result.attention_mask[i].resize(result.seq_length, 0);
            result.token_type_ids[i].resize(result.seq_length, 0);

            for (int j = 0; j < len; j++) {
                result.input_ids[i][j] = (int64_t)all_ids[i][j];
                result.attention_mask[i][j] = 1;
            }
        }

        return result;
    }

    /* Loads the vocabulary from a plain id-ordered list instead of a
     * tokenizer.json.
     *
     * A GGUF file stores its vocabulary as an array of strings under
     * `tokenizer.ggml.tokens`, so there is no JSON document to parse -- but the
     * tokenisation itself is unchanged, and deliberately so: the sparse GGUF
     * backend and the sparse ONNX backend must produce the same ids for the same
     * text, otherwise a fidelity comparison between them measures the
     * tokeniser instead of the runtime. Sharing this class is what makes that
     * true by construction rather than by convention.
     *
     * The cost of sharing is that the known approximations are shared too:
     * lowercasing is ASCII-only, punctuation is split per byte, and accents are
     * not stripped. Those are the same limits the ONNX backend has always had.
     *
     * `tokens[i]` is the surface form of token id i.
     *
     * The four special ids are passed as ids, not as surface forms, because they
     * are vocabulary specific: [CLS] is 101 on BERT and 0 on XLM-R. Pass -1 for
     * one the caller has no id for; the surface form is then looked up in the
     * vocabulary, and if that also fails the id stays unusable (never guessed).
     *
     * Returns false with a reason in `err` when the vocabulary is empty. */
    bool load_vocab(const std::vector<std::string>& tokens,
                    int pad_id, int unk_id, int cls_id, int sep_id,
                    int max_length, bool add_special_tokens,
                    std::string& err) {
        if (tokens.empty()) {
            err = "vocabulary is empty";
            return false;
        }
        /* First id wins on a duplicated surface form. cJSON's path iterates the
         * document's key order, so the two can differ on a vocabulary that
         * contains the same string twice -- which no real vocabulary does. */
        vocab_.clear();
        vocab_.reserve(tokens.size() * 2);
        id_to_token_.assign(tokens.size(), std::string());
        for (size_t i = 0; i < tokens.size(); i++) {
            vocab_.emplace(tokens[i], (int)i);
            id_to_token_[i] = tokens[i];
        }
        type_ = WORDPIECE;
        wp_prefix_ = "##";
        do_lower_case_ = true;   /* every SPLADE export is uncased */
        add_special_tokens_ = add_special_tokens;
        max_length_ = max_length > 0 ? max_length : 512;

        pad_token_id_ = resolve_special(pad_id, "[PAD]", 0);
        unk_token_id_ = resolve_special(unk_id, "[UNK]", 100);
        cls_token_id_ = resolve_special(cls_id, "[CLS]", 101);
        sep_token_id_ = resolve_special(sep_id, "[SEP]", 102);
        return true;
    }

    int pad_token_id() const { return pad_token_id_; }

    /* Public: encode a single text to token IDs (for length bucketing) */
    std::vector<int> encode(const std::string& text) const {
        return encode_single(text);
    }

    int max_length() const { return max_length_; }
    void set_pad_token_id(int id) { pad_token_id_ = id; }

private:
    /* An explicit id wins over the surface form: the metadata is what the model
     * was exported with. `fallback_id` is the conventional BERT value, tried
     * only when the caller passed nothing and the vocabulary carries no such
     * token.
     *
     * The last resort is 0 rather than -1, and not arbitrarily: an unusable
     * special id is emitted verbatim into the input ids, which the embedding
     * graph then uses as an index. 0 exists in every non-empty vocabulary, so
     * this degrades "the delimiter token is wrong" instead of "an index past
     * the end of the embedding table". The caller's declared ids are validated
     * by the loader, so this only fires for a vocabulary too small to hold the
     * conventional ids. */
    int resolve_special(int explicit_id, const char* surface, int fallback_id) const {
        const int vocab_size = (int)id_to_token_.size();
        if (explicit_id >= 0 && explicit_id < vocab_size) return explicit_id;
        auto it = vocab_.find(surface);
        if (it != vocab_.end()) return it->second;
        if (fallback_id >= 0 && fallback_id < vocab_size) return fallback_id;
        return 0;
    }

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::vector<std::string> bpe_merges_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::string wp_prefix_;
    int max_length_;
    int pad_token_id_;
    int cls_token_id_;
    int sep_token_id_;
    int unk_token_id_;
    Type type_;
    bool do_lower_case_;
    bool add_special_tokens_;
    mutable std::string wp_scratch_;
    mutable std::string bpe_scratch_;

    /* Encode a single text to token IDs */
    std::vector<int> encode_single(const std::string& text) const {
        /* HuggingFace's basic_tokenizer does strip_accents *then* lowercase when
         * lowercase is on, and BertTokenizer leaves strip_accents following
         * do_lower_case. Gating the whole thing on do_lower_case_ therefore
         * reproduces both settings rather than only the lowercase one. */
        std::string processed =
            do_lower_case_ ? bert_normalize(text) : text;

        std::vector<int> ids;

        /* Add [CLS] for BERT-style models */
        if (add_special_tokens_) {
            ids.push_back(cls_token_id_);
        }

        /* Pre-tokenize into words */
        auto words = basic_tokenize(processed);

        for (const auto& word : words) {
            if (type_ == WORDPIECE) {
                wordpiece_tokenize(word, ids);
            } else {
                bpe_tokenize(word, ids);
            }
        }

        /* Add [SEP] */
        if (add_special_tokens_) {
            ids.push_back(sep_token_id_);
        }

        /* Truncate */
        if ((int)ids.size() > max_length_) {
            ids.resize(max_length_);
            /* Ensure SEP at end */
            if (add_special_tokens_)
                ids.back() = sep_token_id_;
        }

        return ids;
    }

    /* WordPiece tokenization (BERT-style) */
    void wordpiece_tokenize(const std::string& word, std::vector<int>& ids) const {
        if (word.empty()) return;

        /* Try full word first */
        auto it = vocab_.find(word);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
            return;
        }

        /* Greedy longest-match-first from left */
        size_t start = 0;
        bool found_any = false;
        while (start < word.size()) {
            size_t end = word.size();
            bool found = false;
            while (end > start) {
                wp_scratch_.clear();
                if (start > 0) wp_scratch_.append(wp_prefix_);
                wp_scratch_.append(word, start, end - start);

                auto vit = vocab_.find(wp_scratch_);
                if (vit != vocab_.end()) {
                    ids.push_back(vit->second);
                    found = true;
                    found_any = true;
                    start = end;
                    break;
                }
                end--;
            }
            if (!found) {
                /* Character not in vocab — use [UNK] for entire word */
                if (!found_any) ids.push_back(unk_token_id_);
                return;
            }
        }
    }

    /* BPE tokenization */
    void bpe_tokenize(const std::string& word, std::vector<int>& ids) const {
        if (word.empty()) return;

        /* Start with individual characters (UTF-8 aware) */
        std::vector<std::string> symbols;
        for (size_t i = 0; i < word.size(); ) {
            unsigned char c = (unsigned char)word[i];
            int bytes = 1;
            if ((c & 0xE0) == 0xC0) bytes = 2;
            else if ((c & 0xF0) == 0xE0) bytes = 3;
            else if ((c & 0xF8) == 0xF0) bytes = 4;
            symbols.push_back(word.substr(i, bytes));
            i += bytes;
        }

        /* Iteratively merge the highest-priority pair */
        while (symbols.size() > 1) {
            int best_rank = INT_MAX;
            int best_pos = -1;

            for (int i = 0; i < (int)symbols.size() - 1; i++) {
                bpe_scratch_.clear();
                bpe_scratch_.append(symbols[i]);
                bpe_scratch_.push_back(' ');
                bpe_scratch_.append(symbols[i + 1]);
                auto it = bpe_ranks_.find(bpe_scratch_);
                if (it != bpe_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                }
            }

            if (best_pos < 0) break; /* no more merges */

            /* Merge the pair */
            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);
        }

        /* Look up each symbol in vocab */
        for (const auto& sym : symbols) {
            auto it = vocab_.find(sym);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(unk_token_id_);
            }
        }
    }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP */




