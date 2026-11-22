/*
 * libembedding - detail/tokenizer_impl.hpp
 * Built-in HuggingFace tokenizer.json parser (WordPiece + BPE)
 * No external Rust/tokenizers-cpp dependency required.
 *
 * Supports the tokenizer types used by embedding models in the registry:
 * - WordPiece (BERT-style: bge, MiniLM, mpnet, etc.)
 * - BPE (GPT/Sentencepiece-style: nomic, jina, CLIP, etc.)
 * - Unigram (XLM-R family: precompiled charsmap + Metaspace, see tokenizer_unigram.hpp)
 *
 * Auteur: David Orel
 * Version: 1.16.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP

#include "cJSON.h"
#include "tokenizer_bytelevel.hpp"
#include "tokenizer_nfc.hpp"
#include "tokenizer_unicode.hpp"
#include "tokenizer_unigram.hpp"

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
 * Scope: the per-codepoint tables live in tokenizer_unicode.hpp and were
 * checked codepoint by codepoint against HF `tokenizers` (see the fidelity
 * test). They cover Latin (all of Latin-1, Extended-A/B, Additional), Greek,
 * Cyrillic, Armenian, Arabic, Devanagari, kana, fullwidth Latin and Hangul;
 * every other codepoint passes through unchanged, and the fidelity test
 * reports the remaining mismatch rate per block.
 *
 * Two facts the first draft of this code got wrong, both visible in HF's
 * output: letters without a canonical decomposition (AE, OE, Thorn, Eth) are
 * still LOWERCASED (they just keep their shape), and sharp-s / dotless i are
 * left alone because they are already lowercase.
 * ========================================================================= */

/* Sentinel codepoint for a byte that is not part of well-formed UTF-8. It is
 * outside every table, so callers keep the byte verbatim instead of
 * reinterpreting it as a Latin-1 letter. */
static constexpr uint32_t kInvalidCp = 0xFFFFFFFFu;

/* Decode one UTF-8 codepoint. Returns bytes consumed and stores the codepoint.
 * Strict: continuation bytes, overlong forms, surrogates and values above
 * U+10FFFF are rejected; a malformed byte is consumed alone and reported as
 * kInvalidCp. */
static inline size_t utf8_next(const std::string& s, size_t i, uint32_t& cp) {
    const size_t size = s.size();
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { cp = c; return 1; }
    auto cont = [&](size_t k) { return k < size && ((unsigned char)s[k] & 0xC0) == 0x80; };
    if (c >= 0xC2 && c <= 0xDF && cont(i + 1)) {
        cp = ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
        return 2;
    }
    if (c >= 0xE0 && c <= 0xEF && cont(i + 1) && cont(i + 2)) {
        const unsigned char b1 = (unsigned char)s[i + 1];
        if ((c == 0xE0 && b1 < 0xA0) || (c == 0xED && b1 > 0x9F)) { cp = kInvalidCp; return 1; }
        cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(b1 & 0x3F) << 6) |
             ((unsigned char)s[i + 2] & 0x3F);
        return 3;
    }
    if (c >= 0xF0 && c <= 0xF4 && cont(i + 1) && cont(i + 2) && cont(i + 3)) {
        const unsigned char b1 = (unsigned char)s[i + 1];
        if ((c == 0xF0 && b1 < 0x90) || (c == 0xF4 && b1 > 0x8F)) { cp = kInvalidCp; return 1; }
        cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(b1 & 0x3F) << 12) |
             ((uint32_t)((unsigned char)s[i + 2] & 0x3F) << 6) |
             ((unsigned char)s[i + 3] & 0x3F);
        return 4;
    }
    cp = kInvalidCp;
    return 1;
}

/* BertNormalizer's case and accent steps, without clean_text and without CJK
 * padding (those live in basic_tokenize), per codepoint. The two switches are
 * independent, as in HF:
 *   strip  NFD, then drop the nonspacing marks (Mn)
 *   lower  lowercase
 * Both: the default uncased BERT. Only strip: the base letter keeps its case.
 * Only lower: accented letters stay composed. See tokenizer_unicode.hpp. */
static inline std::string bert_normalize(const std::string& s, bool lower = true,
                                         bool strip = true) {
    if (!lower && !strip) return s;
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        const size_t n = utf8_next(s, i, cp);
        i += n;

        if (cp < 0x80) {
            out += (lower && cp >= 'A' && cp <= 'Z') ? (char)(cp - 'A' + 'a') : (char)cp;
            continue;
        }
        uint32_t f = cp;
        if (strip) {
            if (unicode::bert_is_stripped(cp)) continue;
            if (unicode::append_hangul_nfd(out, cp)) continue;
            const uint32_t both = unicode::bert_fold(cp);   /* strip, then lowercase */
            if (lower) {
                f = both;
            } else {
                /* Strip alone: a letter that merely lowercases is unchanged; a
                 * decomposed one yields its base letter, upper-cased again when
                 * the source was upper case. */
                const uint32_t lc = unicode::bert_lower(cp);
                if (cp == 0x0130) f = 'I';          /* I-dot -> I + dot above */
                else if (cp == 0x212A) f = 'K';     /* Kelvin sign -> K */
                else if (cp == 0x2126) f = 0x03A9;  /* Ohm sign -> capital omega */
                else if (both != lc) f = (lc != cp) ? unicode::bert_upper(both) : both;
            }
        } else if (cp == 0x0130) {
            out += "i\xCC\x87";   /* HF lowercases I-dot to i + combining dot */
            continue;
        } else {
            f = unicode::bert_lower(cp);
        }
        if (f == cp) out.append(s, i - n, n);   /* keeps malformed bytes verbatim */
        else unicode::append_utf8(out, f);
    }
    return out;
}

/* BertNormalizer(clean_text, handle_chinese_chars) + BertPreTokenizer: drop
 * control/format characters (clean_text), split on whitespace, make every CJK
 * ideograph its own word (handle_chinese_chars), and split on punctuation (each
 * mark is a word). Everything else, including accented letters, kana, Hangul
 * and emoji, stays glued into words. */
static inline std::vector<std::string> basic_tokenize(const std::string& text,
                                                      bool clean_text = true,
                                                      bool handle_chinese = true) {
    std::vector<std::string> tokens;
    tokens.reserve(text.size() / 4 + 1);
    std::string current;
    auto flush = [&]() {
        if (!current.empty()) { tokens.push_back(current); current.clear(); }
    };
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char)text[i];
        if (c < 0x80) {
            /* ASCII fast path, no table lookup. */
            i++;
            if (c == ' ' || (c >= 0x09 && c <= 0x0D)) {
                /* \v and \f are deleted by clean_text, whitespace otherwise */
                if (clean_text && (c == 0x0B || c == 0x0C)) continue;
                flush();
            } else if (c < 0x20 || c == 0x7F) {
                if (!clean_text) current += (char)c;
            } else if ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
                       (c >= '[' && c <= '`') || (c >= '{' && c <= '~')) {
                flush();
                tokens.emplace_back(1, (char)c);
            } else {
                current += (char)c;
            }
            continue;
        }
        uint32_t cp = 0;
        const size_t n = utf8_next(text, i, cp);
        i += n;
        if (clean_text && unicode::bert_is_removed(cp)) continue;
        if (unicode::bert_is_whitespace(cp)) { flush(); continue; }
        if ((handle_chinese && unicode::bert_is_chinese(cp)) || unicode::bert_is_punct(cp)) {
            flush();
            tokens.push_back(text.substr(i - n, n));
            continue;
        }
        current.append(text, i - n, n);
    }
    flush();
    return tokens;
}

/* =========================================================================
 * Legacy pre-tokenisation, kept ONLY for byte-level BPE models.
 *
 * Byte-level BPE (CLIP, ModernBERT) is not implemented faithfully here and is
 * due a dedicated rewrite; until then its behaviour is deliberately left as it
 * was before the WordPiece fidelity work (T26): Latin-1 / Extended-A accent
 * folding and lowercasing, and every non-ASCII character as its own word.
 * Pinned by test_tokenizer so that the later change is explicit.
 * ========================================================================= */

/* Base ASCII letter of a Latin-1 Supplement codepoint, or -1 when the codepoint
 * has no canonical decomposition. */
static inline int legacy_latin1_base(uint32_t cp) {
    static const int kBases[64] = {
        'a', 'a', 'a', 'a', 'a', 'a', -1, 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',
        -1, 'n', 'o', 'o', 'o', 'o', 'o', -1, -1, 'u', 'u', 'u', 'u', 'y', -1, -1,
        'a', 'a', 'a', 'a', 'a', 'a', -1, 'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',
        -1, 'n', 'o', 'o', 'o', 'o', 'o', -1, -1, 'u', 'u', 'u', 'u', 'y', -1, 'y',
    };
    if (cp >= 0x00C0 && cp <= 0x00FF) return kBases[cp - 0x00C0];
    return -1;
}

static inline int legacy_extended_a_base(uint32_t cp) {
    switch (cp) {
        case 0x0110: case 0x0111: case 0x0126: case 0x0127: case 0x0132: case 0x0133:
        case 0x013F: case 0x0140: case 0x014A: case 0x014B: case 0x0152: case 0x0153:
        case 0x0166: case 0x0167:
            return -1;
        default:
            break;
    }
    if (cp == 0x0130 || cp == 0x0131) return 'i';
    if (cp == 0x017F) return 's';
    const uint32_t base = (cp % 2 == 0) ? cp : (cp - 1);
    if (base >= 0x00C0 && base <= 0x00FF) return legacy_latin1_base(base);
    return -1;
}

static inline std::string legacy_bert_normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        const size_t n = utf8_next(s, i, cp);
        i += n;
        if (cp >= 'A' && cp <= 'Z') { out += (char)(cp - 'A' + 'a'); continue; }
        if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7) {
            const int b = legacy_latin1_base(cp);
            if (b >= 0) { out += (char)b; continue; }
            out.append(s, i - n, n);
            continue;
        }
        int folded = -1;
        if (cp >= 0x00C0 && cp <= 0x00FF) folded = legacy_latin1_base(cp);
        else if (cp >= 0x0100 && cp <= 0x017F) folded = legacy_extended_a_base(cp);
        if (folded >= 0) { out += (char)folded; continue; }
        out.append(s, i - n, n);
    }
    return out;
}

static inline std::vector<std::string> legacy_basic_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        if (c <= 0x20) {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            i++;
        } else if ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
                   (c >= '[' && c <= '`') || (c >= '{' && c <= '~')) {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            current += (char)c;
            tokens.push_back(current);
            current.clear();
            i++;
        } else if (c >= 0x80) {
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
    /* BPE is the legacy approximation; BPE_BYTELEVEL is the HF-exact byte-level
     * path (CLIP, GPT-2 style), chosen only when tokenizer.json is fully
     * understood (see parse_bytelevel). */
    enum Type { WORDPIECE, BPE, UNIGRAM, BPE_BYTELEVEL };

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
                else if (t == "Unigram") type_ = UNIGRAM;
                else type_ = WORDPIECE;
            }

            if (type_ == UNIGRAM) {
                try {
                    unigram_.load(model);
                } catch (...) {
                    cJSON_Delete(root);
                    throw;
                }
                if (unigram_.unk_id() >= 0) unk_token_id_ = unigram_.unk_id();
            }

            /* Load vocabulary */
            cJSON* vocab = cJSON_GetObjectItem(model, "vocab");
            if (vocab && type_ != UNIGRAM) {
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

            /* Words longer than this many characters become a single [UNK]
             * (HF default 100). */
            cJSON* maxc = cJSON_GetObjectItem(model, "max_input_chars_per_word");
            if (maxc && cJSON_IsNumber(maxc) && maxc->valueint > 0)
                wp_max_chars_ = (size_t)std::min(maxc->valueint, 10000);

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

        /* WordPiece normalisation switches, independent of one another as in
         * HF (do_lower_case_ above only drives the legacy BPE path). */
        parse_wordpiece_normalizer(normalizer);

        if (type_ == UNIGRAM) {
            try {
                parse_unigram_normalizer(normalizer);
                parse_metaspace(cJSON_GetObjectItem(root, "pre_tokenizer"));
            } catch (...) {
                cJSON_Delete(root);
                throw;
            }
        }

        /* Byte-level BPE (CLIP, GPT-2): only when the JSON is fully understood,
         * otherwise the legacy BPE path stays in charge. */
        if (type_ == BPE && parse_bytelevel(root, model)) type_ = BPE_BYTELEVEL;
        if (type_ == BPE) parse_whitespace_bpe(root, model);

        /* Post-processor: detect special tokens to add */
        cJSON* post = cJSON_GetObjectItem(root, "post_processor");
        if (type_ == BPE_BYTELEVEL) add_special_tokens_ = false;
        if (post) {
            cJSON* ptype = cJSON_GetObjectItem(post, "type");
            if (ptype && ptype->valuestring &&
                strcmp(ptype->valuestring, "TemplateProcessing") == 0) {
                /* Check for [CLS] and [SEP] in template */
                add_special_tokens_ = true;
                if (type_ == UNIGRAM || type_ == BPE_BYTELEVEL) parse_single_template(post);
            } else if (type_ == BPE_BYTELEVEL && ptype && ptype->valuestring &&
                       (strcmp(ptype->valuestring, "RobertaProcessing") == 0 ||
                        strcmp(ptype->valuestring, "BertProcessing") == 0)) {
                /* [cls] $A [sep], ids given as ["token", id] */
                cJSON* cls = cJSON_GetObjectItem(post, "cls");
                cJSON* sep = cJSON_GetObjectItem(post, "sep");
                cJSON* cid = cls ? cJSON_GetArrayItem(cls, 1) : nullptr;
                cJSON* sid = sep ? cJSON_GetArrayItem(sep, 1) : nullptr;
                if (cid && sid && cJSON_IsNumber(cid) && cJSON_IsNumber(sid)) {
                    cls_token_id_ = cid->valueint;
                    sep_token_id_ = sid->valueint;
                    add_special_tokens_ = true;
                }
            }
            pair_tmpl_.clear();
            parse_pair_processor(post);
        }

        /* Truncation strategy used when a pair does not fit */
        cJSON* trunc = cJSON_GetObjectItem(root, "truncation");
        if (trunc && cJSON_IsObject(trunc)) {
            cJSON* st = cJSON_GetObjectItem(trunc, "strategy");
            if (st && st->valuestring) {
                if (strcmp(st->valuestring, "OnlyFirst") == 0) trunc_strategy_ = 1;
                else if (strcmp(st->valuestring, "OnlySecond") == 0) trunc_strategy_ = 2;
                else trunc_strategy_ = 0;
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
        std::vector<std::vector<int>> all_ids;
        all_ids.reserve(texts.size());
        for (const auto& text : texts) {
            all_ids.push_back(encode_single(text));
        }
        return encode_batch(all_ids);
    }

    /* Encodes (a, b) as a real sequence pair, the way HF tokenizers does with the
     * tokenizer.json post_processor: special tokens and segment (type) ids come
     * from the pair template (TemplateProcessing / BertProcessing /
     * RobertaProcessing), BERT style `[CLS] a [SEP] b [SEP]` when none is
     * declared. When the pair exceeds max_length it is truncated with the
     * tokenizer's strategy (default longest_first: trim the longer segment). */
    void encode_pair(const std::string& a, const std::string& b,
                     std::vector<int>& ids, std::vector<int>& types) const {
        std::vector<int> ta, tb;
        tokenize_body(a, ta);
        tokenize_body(b, tb);

        std::vector<PairItem> tmpl;
        if (!pair_tmpl_.empty()) {
            tmpl = pair_tmpl_;
        } else {
            tmpl = {{0, cls_token_id_, 0}, {1, 0, 0}, {0, sep_token_id_, 0},
                    {2, 0, 1}, {0, sep_token_id_, 1}};
        }
        int nspecial = 0;
        for (const auto& it : tmpl) if (it.kind == 0) nspecial++;
        const int budget = std::max(0, max_length_ - nspecial);

        size_t na = ta.size(), nb = tb.size();
        while ((int)(na + nb) > budget) {
            if (trunc_strategy_ == 1) {
                if (na == 0) break;
                na--;
            } else if (trunc_strategy_ == 2) {
                if (nb == 0) break;
                nb--;
            } else if (na > nb) {
                na--;
            } else if (nb > 0) {
                nb--;
            } else {
                break;
            }
        }
        ta.resize(na);
        tb.resize(nb);

        ids.clear();
        types.clear();
        for (const auto& it : tmpl) {
            if (it.kind == 0) {
                ids.push_back(it.id);
                types.push_back(it.type);
            } else {
                const std::vector<int>& seg = (it.kind == 1) ? ta : tb;
                for (int t : seg) {
                    ids.push_back(t);
                    types.push_back(it.type);
                }
            }
        }
    }

    EncodingBatch encode_batch(
        const std::vector<std::vector<int>>& all_ids,
        const std::vector<std::vector<int>>* all_types = nullptr) const {
        EncodingBatch result;
        result.seq_length = 0;
        for (const auto& ids : all_ids) {
            if ((int)ids.size() > result.seq_length)
                result.seq_length = (int)ids.size();
        }

        /* Cap at max_length */
        if (result.seq_length > max_length_)
            result.seq_length = max_length_;

        const int batch_size = (int)all_ids.size();
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
                if (all_types && j < (int)(*all_types)[i].size())
                    result.token_type_ids[i][j] = (*all_types)[i][j];
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
        lower_ = strip_ = clean_text_ = handle_chinese_ = true;
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
    int cls_token_id() const { return cls_token_id_; }
    int sep_token_id() const { return sep_token_id_; }
    int unk_token_id() const { return unk_token_id_; }
    /* Number of vocabulary entries known by id (0 when the tokenizer type
     * keeps no id table). */
    int vocab_size() const { return (int)id_to_token_.size(); }

    /* Id of the vocabulary entry whose surface form is exactly `token`
     * (e.g. "[MASK]", "[unused0]", "!"), or -1 when the vocabulary has none.
     * Added tokens that are not part of model.vocab are not seen. Used by the
     * multivector runtime to resolve the markers and the skiplist by string. */
    int token_to_id(const std::string& token) const {
        auto it = vocab_.find(token);
        return it == vocab_.end() ? -1 : it->second;
    }

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
    size_t wp_max_chars_ = 100;
    int max_length_;
    int pad_token_id_;
    int cls_token_id_;
    int sep_token_id_;
    int unk_token_id_;
    Type type_;
    bool do_lower_case_;
    /* WordPiece normalisation, as BertNormalizer's four switches. */
    bool lower_ = true;
    bool strip_ = true;
    bool clean_text_ = true;
    bool handle_chinese_ = true;
    bool add_special_tokens_;

    /* Read the normalizer section of tokenizer.json into the four switches.
     * BertNormalizer: `lowercase` (default true), `strip_accents` (null or
     * absent follows `lowercase`), `clean_text` and `handle_chinese_chars`
     * (default true). Lowercase alone lowercases only. Sequence: Lowercase,
     * StripAccents and NFD/NFKD are honoured, accents being stripped only when
     * both StripAccents and a decomposition are present, as in HF. Anything
     * else (or no normalizer) leaves all four off. */
    void parse_wordpiece_normalizer(const cJSON* normalizer) {
        lower_ = strip_ = clean_text_ = handle_chinese_ = false;
        if (!normalizer) return;
        auto flag = [](const cJSON* n, const char* key, bool dflt) {
            const cJSON* v = cJSON_GetObjectItem(const_cast<cJSON*>(n), key);
            if (cJSON_IsTrue(v)) return true;
            if (cJSON_IsFalse(v)) return false;
            return dflt;
        };
        auto bert = [&](const cJSON* n) {
            lower_ = flag(n, "lowercase", true);
            const cJSON* sa = cJSON_GetObjectItem(const_cast<cJSON*>(n), "strip_accents");
            strip_ = (cJSON_IsTrue(sa) || cJSON_IsFalse(sa)) ? cJSON_IsTrue(sa) : lower_;
            clean_text_ = flag(n, "clean_text", true);
            handle_chinese_ = flag(n, "handle_chinese_chars", true);
        };
        const cJSON* type = cJSON_GetObjectItem(const_cast<cJSON*>(normalizer), "type");
        if (!type || !type->valuestring) return;
        const std::string t = type->valuestring;
        if (t == "BertNormalizer") {
            bert(normalizer);
        } else if (t == "Lowercase") {
            lower_ = true;
        } else if (t == "Sequence") {
            bool strip_accents = false, decomposed = false;
            const cJSON* list = cJSON_GetObjectItem(const_cast<cJSON*>(normalizer), "normalizers");
            for (const cJSON* sub = cJSON_IsArray(list) ? list->child : nullptr; sub; sub = sub->next) {
                const cJSON* st = cJSON_GetObjectItem(const_cast<cJSON*>(sub), "type");
                if (!st || !st->valuestring) continue;
                const std::string s = st->valuestring;
                if (s == "Lowercase") lower_ = true;
                else if (s == "StripAccents") strip_accents = true;
                else if (s == "NFD" || s == "NFKD") decomposed = true;
                else if (s == "BertNormalizer") bert(sub);
            }
            if (strip_accents && decomposed) strip_ = true;
        }
    }
    mutable std::string wp_scratch_;
    mutable std::string bpe_scratch_;

    /* One element of a pair template: kind 0 = special token `id`,
     * 1 = first sequence, 2 = second sequence; `type` is the segment id. */
    struct PairItem { int kind; int id; int type; };
    std::vector<PairItem> pair_tmpl_;
    int trunc_strategy_ = 0; /* 0 longest_first, 1 only_first, 2 only_second */

    /* Reads the pair layout of a post_processor node (recursing into
     * `Sequence`). Leaves pair_tmpl_ empty when nothing usable is found. */
    void parse_pair_processor(cJSON* post) {
        cJSON* ptype = cJSON_GetObjectItem(post, "type");
        if (!ptype || !ptype->valuestring) return;
        const std::string t = ptype->valuestring;
        if (t == "Sequence") {
            cJSON* procs = cJSON_GetObjectItem(post, "processors");
            if (procs && cJSON_IsArray(procs)) {
                int n = cJSON_GetArraySize(procs);
                for (int i = 0; i < n && pair_tmpl_.empty(); i++)
                    parse_pair_processor(cJSON_GetArrayItem(procs, i));
            }
        } else if (t == "BertProcessing" || t == "RobertaProcessing") {
            cJSON* sep = cJSON_GetObjectItem(post, "sep");
            cJSON* cls = cJSON_GetObjectItem(post, "cls");
            if (!sep || !cls || !cJSON_IsArray(sep) || !cJSON_IsArray(cls)) return;
            cJSON* sid = cJSON_GetArrayItem(sep, 1);
            cJSON* cid = cJSON_GetArrayItem(cls, 1);
            if (!sid || !cid || !cJSON_IsNumber(sid) || !cJSON_IsNumber(cid)) return;
            const int s = sid->valueint, c = cid->valueint;
            if (t == "BertProcessing") {
                pair_tmpl_ = {{0, c, 0}, {1, 0, 0}, {0, s, 0}, {2, 0, 1}, {0, s, 1}};
            } else {
                pair_tmpl_ = {{0, c, 0}, {1, 0, 0}, {0, s, 0}, {0, s, 0},
                              {2, 0, 0}, {0, s, 0}};
            }
        } else if (t == "TemplateProcessing") {
            cJSON* pair = cJSON_GetObjectItem(post, "pair");
            cJSON* specials = cJSON_GetObjectItem(post, "special_tokens");
            if (!pair || !cJSON_IsArray(pair)) return;
            std::vector<PairItem> out;
            int n = cJSON_GetArraySize(pair);
            for (int i = 0; i < n; i++) {
                cJSON* el = cJSON_GetArrayItem(pair, i);
                cJSON* seq = cJSON_GetObjectItem(el, "Sequence");
                cJSON* sp = cJSON_GetObjectItem(el, "SpecialToken");
                if (seq) {
                    cJSON* id = cJSON_GetObjectItem(seq, "id");
                    cJSON* ty = cJSON_GetObjectItem(seq, "type_id");
                    if (!id || !id->valuestring) return;
                    out.push_back({strcmp(id->valuestring, "B") == 0 ? 2 : 1, 0,
                                   ty && cJSON_IsNumber(ty) ? ty->valueint : 0});
                } else if (sp) {
                    cJSON* id = cJSON_GetObjectItem(sp, "id");
                    cJSON* ty = cJSON_GetObjectItem(sp, "type_id");
                    if (!id || !id->valuestring) return;
                    const int type = ty && cJSON_IsNumber(ty) ? ty->valueint : 0;
                    cJSON* def = specials ? cJSON_GetObjectItem(specials, id->valuestring) : nullptr;
                    cJSON* ids = def ? cJSON_GetObjectItem(def, "ids") : nullptr;
                    if (ids && cJSON_IsArray(ids) && cJSON_GetArraySize(ids) > 0) {
                        int m = cJSON_GetArraySize(ids);
                        for (int k = 0; k < m; k++) {
                            cJSON* v = cJSON_GetArrayItem(ids, k);
                            if (!v || !cJSON_IsNumber(v)) return;
                            out.push_back({0, v->valueint, type});
                        }
                    } else {
                        auto vit = vocab_.find(id->valuestring);
                        if (vit == vocab_.end()) return;
                        out.push_back({0, vit->second, type});
                    }
                } else {
                    return;
                }
            }
            pair_tmpl_ = out;
        }
    }

    /* Unigram (SentencePiece) state. */
    UnigramModel unigram_;
    PrecompiledCharsmap charsmap_;
    struct NormOp {
        enum Kind { CHARSMAP, REPLACE_LITERAL, COLLAPSE_SPACES, PREPEND, STRIP } kind;
        std::string a, b;           /* pattern/content or prepend text */
        bool left = true, right = true;
    };
    std::vector<NormOp> unigram_norm_;
    MetaspaceConfig metaspace_;
    /* Single-sequence TemplateProcessing: ids before and after $A. The pair
     * template is not interpreted here. */
    std::vector<int> single_prefix_, single_suffix_;
    bool single_template_ = false;

    /* Normalizer (Sequence or single) -> ordered ops. Supported: Precompiled,
     * Replace (literal String, or Regex " {2,}" / " +" collapsing spaces),
     * Prepend, Strip. Other types are ignored. */
    void parse_unigram_normalizer(const cJSON* normalizer) {
        unigram_norm_.clear();
        if (!normalizer) return;
        auto one = [&](const cJSON* n) {
            auto get = [&](const char* k) { return cJSON_GetObjectItem(const_cast<cJSON*>(n), k); };
            const cJSON* t = get("type");
            if (!t || !t->valuestring) return;
            const std::string ty = t->valuestring;
            if (ty == "Precompiled") {
                const cJSON* c = get("precompiled_charsmap");
                if (!c || !c->valuestring) throw std::runtime_error("Precompiled normalizer without charsmap");
                charsmap_.load_base64(c->valuestring);
                unigram_norm_.push_back({NormOp::CHARSMAP, "", "", true, true});
            } else if (ty == "Replace") {
                const cJSON* p = get("pattern");
                const cJSON* c = get("content");
                const std::string content = (c && c->valuestring) ? c->valuestring : "";
                const cJSON* s = p ? cJSON_GetObjectItem(const_cast<cJSON*>(p), "String") : nullptr;
                const cJSON* r = p ? cJSON_GetObjectItem(const_cast<cJSON*>(p), "Regex") : nullptr;
                if (s && s->valuestring) {
                    unigram_norm_.push_back({NormOp::REPLACE_LITERAL, s->valuestring, content, true, true});
                } else if (r && r->valuestring) {
                    const std::string re = r->valuestring;
                    if (re == " {2,}" || re == " +" || re == " {1,}")
                        unigram_norm_.push_back({NormOp::COLLAPSE_SPACES, re == " {2,}" ? "2" : "1", content, true, true});
                }
            } else if (ty == "Prepend") {
                const cJSON* pr = get("prepend");
                if (pr && pr->valuestring) unigram_norm_.push_back({NormOp::PREPEND, pr->valuestring, "", true, true});
            } else if (ty == "Strip") {
                const cJSON* l = get("strip_left");
                const cJSON* r = get("strip_right");
                unigram_norm_.push_back({NormOp::STRIP, "", "", !cJSON_IsFalse(l), !cJSON_IsFalse(r)});
            }
        };
        const cJSON* ty = cJSON_GetObjectItem(const_cast<cJSON*>(normalizer), "type");
        if (ty && ty->valuestring && strcmp(ty->valuestring, "Sequence") == 0) {
            const cJSON* list = cJSON_GetObjectItem(const_cast<cJSON*>(normalizer), "normalizers");
            for (const cJSON* sub = cJSON_IsArray(list) ? list->child : nullptr; sub; sub = sub->next) one(sub);
        } else {
            one(normalizer);
        }
    }

    /* Metaspace: `replacement`, `prepend_scheme` ("always" default, "first",
     * "never") or the older boolean `add_prefix_space`, `split` (default true).
     * Looks inside a Sequence pre-tokenizer too. */
    void parse_metaspace(const cJSON* pre) {
        metaspace_ = MetaspaceConfig();
        if (!pre) return;
        const cJSON* ty = cJSON_GetObjectItem(const_cast<cJSON*>(pre), "type");
        if (!ty || !ty->valuestring) return;
        if (strcmp(ty->valuestring, "Sequence") == 0) {
            const cJSON* list = cJSON_GetObjectItem(const_cast<cJSON*>(pre), "pretokenizers");
            bool ws_split = false;
            for (const cJSON* sub = cJSON_IsArray(list) ? list->child : nullptr; sub; sub = sub->next) {
                const cJSON* st = cJSON_GetObjectItem(const_cast<cJSON*>(sub), "type");
                if (!st || !st->valuestring) continue;
                if (strcmp(st->valuestring, "WhitespaceSplit") == 0) ws_split = true;
                if (strcmp(st->valuestring, "Metaspace") == 0) {
                    parse_metaspace(sub);   /* resets metaspace_ */
                    metaspace_.whitespace_split = ws_split;
                    return;
                }
            }
            return;
        }
        if (strcmp(ty->valuestring, "Metaspace") != 0) return;
        metaspace_.enabled = true;
        const cJSON* rep = cJSON_GetObjectItem(const_cast<cJSON*>(pre), "replacement");
        if (rep && rep->valuestring && rep->valuestring[0]) metaspace_.replacement = rep->valuestring;
        const cJSON* scheme = cJSON_GetObjectItem(const_cast<cJSON*>(pre), "prepend_scheme");
        const cJSON* aps = cJSON_GetObjectItem(const_cast<cJSON*>(pre), "add_prefix_space");
        if (scheme && scheme->valuestring) metaspace_.prepend = strcmp(scheme->valuestring, "never") != 0;
        else if (cJSON_IsFalse(aps)) metaspace_.prepend = false;
        const cJSON* sp = cJSON_GetObjectItem(const_cast<cJSON*>(pre), "split");
        if (cJSON_IsFalse(sp)) metaspace_.split = false;
    }

    /* TemplateProcessing "single": SpecialToken ids around the Sequence A. */
    void parse_single_template(const cJSON* post) {
        const cJSON* single = cJSON_GetObjectItem(const_cast<cJSON*>(post), "single");
        const cJSON* specials = cJSON_GetObjectItem(const_cast<cJSON*>(post), "special_tokens");
        if (!cJSON_IsArray(single)) return;
        bool seen_a = false;
        std::vector<int> pre, suf;
        for (const cJSON* it = single->child; it; it = it->next) {
            if (cJSON_GetObjectItem(const_cast<cJSON*>(it), "Sequence")) { seen_a = true; continue; }
            const cJSON* st = cJSON_GetObjectItem(const_cast<cJSON*>(it), "SpecialToken");
            const cJSON* name = st ? cJSON_GetObjectItem(const_cast<cJSON*>(st), "id") : nullptr;
            const cJSON* def = (name && name->valuestring && specials)
                                   ? cJSON_GetObjectItem(const_cast<cJSON*>(specials), name->valuestring) : nullptr;
            const cJSON* ids = def ? cJSON_GetObjectItem(const_cast<cJSON*>(def), "ids") : nullptr;
            if (!cJSON_IsArray(ids)) return;
            for (const cJSON* x = ids->child; x; x = x->next)
                (seen_a ? suf : pre).push_back(x->valueint);
        }
        if (!seen_a) return;
        single_prefix_ = pre;
        single_suffix_ = suf;
        single_template_ = true;
    }

    /* Normalizer ops, then Metaspace, then one Viterbi per pre-token. */
    void unigram_encode(const std::string& text, std::vector<int>& ids) const {
        std::string s = text;
        for (const NormOp& op : unigram_norm_) {
            switch (op.kind) {
            case NormOp::CHARSMAP: s = charsmap_.normalize(s); break;
            case NormOp::PREPEND: s = op.a + s; break;
            case NormOp::STRIP: {
                size_t b = 0, e = s.size();
                if (op.left) while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) b++;
                if (op.right) while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) e--;
                s = s.substr(b, e - b);
                break;
            }
            case NormOp::REPLACE_LITERAL: {
                if (op.a.empty()) break;
                std::string o;
                size_t i = 0, f;
                while ((f = s.find(op.a, i)) != std::string::npos) {
                    o.append(s, i, f - i);
                    o += op.b;
                    i = f + op.a.size();
                }
                o.append(s, i, std::string::npos);
                s.swap(o);
                break;
            }
            case NormOp::COLLAPSE_SPACES: {
                const size_t minrun = op.a == "2" ? 2 : 1;
                std::string o;
                o.reserve(s.size());
                for (size_t i = 0; i < s.size();) {
                    if (s[i] != ' ') { o.push_back(s[i++]); continue; }
                    size_t j = i;
                    while (j < s.size() && s[j] == ' ') j++;
                    if (j - i >= minrun) o += op.b; else o.append(s, i, j - i);
                    i = j;
                }
                s.swap(o);
                break;
            }
            }
        }
        if (s.empty()) return;

        if (!metaspace_.enabled) {
            unigram_.encode_word(s, ids);
            return;
        }
        if (!metaspace_.whitespace_split) {
            metaspace_encode(s, ids);
            return;
        }
        /* WhitespaceSplit, then Metaspace on each word: whitespace runs are
         * dropped instead of becoming lone replacement pieces. */
        std::string word;
        for (size_t i = 0; i < s.size();) {
            size_t len = 1;
            if (utf8_is_whitespace(s, i, len)) {
                if (!word.empty()) { metaspace_encode(word, ids); word.clear(); }
            } else {
                word.append(s, i, len);
            }
            i += len;
        }
        if (!word.empty()) metaspace_encode(word, ids);
    }

    void metaspace_encode(const std::string& s, std::vector<int>& ids) const {
        const std::string& rep = metaspace_.replacement;
        std::string m;
        m.reserve(s.size() + rep.size() * 2);
        for (char c : s) {
            if (c == ' ') m += rep; else m.push_back(c);
        }
        if (metaspace_.prepend && m.compare(0, rep.size(), rep) != 0) m.insert(0, rep);
        if (!metaspace_.split) {
            unigram_.encode_word(m, ids);
            return;
        }
        /* split, merged with next: each replacement starts a new word */
        size_t start = 0;
        std::string word;
        for (size_t i = 1; i <= m.size();) {
            if (i == m.size() || m.compare(i, rep.size(), rep) == 0) {
                word.assign(m, start, i - start);
                unigram_.encode_word(word, ids);
                start = i;
                i += rep.size();
                if (i > m.size()) break;
            } else {
                i++;
            }
        }
    }

    /* Normalise, pre-tokenize and sub-tokenize `text`; no special tokens. */
    void tokenize_body(const std::string& text, std::vector<int>& ids) const {
        if (type_ == UNIGRAM) {
            unigram_encode(text, ids);
            return;
        }
        if (type_ == BPE_BYTELEVEL) {
            tokenize_bytelevel(text, ids);
            return;
        }
        if (type_ == BPE && ws_bpe_) {
            ws_bpe_tokenize(text, ids);
            return;
        }
        /* Normalise and pre-tokenize into words. WordPiece follows HF's
         * BertNormalizer switches; byte-level BPE keeps its legacy path. */
        std::vector<std::string> words;
        if (type_ == BPE) {
            words = legacy_basic_tokenize(
                do_lower_case_ ? legacy_bert_normalize(text) : text);
        } else {
            words = basic_tokenize(bert_normalize(text, lower_, strip_),
                                   clean_text_, handle_chinese_);
        }

        for (const auto& word : words) {
            if (type_ == WORDPIECE) {
                wordpiece_tokenize(word, ids);
            } else {
                bpe_tokenize(word, ids);
            }
        }
    }

    /* Encode a single text to token IDs */
    std::vector<int> encode_single(const std::string& text) const {
        std::vector<int> ids;

        if (type_ == UNIGRAM || type_ == BPE_BYTELEVEL) {
            /* <s> $A </s> (or whatever the single template says) */
            if (add_special_tokens_) {
                if (single_template_) ids = single_prefix_;
                else ids.push_back(cls_token_id_);
            }
            tokenize_body(text, ids);
            if (add_special_tokens_) {
                if (single_template_) ids.insert(ids.end(), single_suffix_.begin(), single_suffix_.end());
                else ids.push_back(sep_token_id_);
            }
            if ((int)ids.size() > max_length_) {
                ids.resize(max_length_);
                if (add_special_tokens_) {
                    if (single_template_ && !single_suffix_.empty()) {
                        const size_t k = std::min(single_suffix_.size(), ids.size());
                        std::copy(single_suffix_.end() - k, single_suffix_.end(), ids.end() - k);
                    } else if (!single_template_) {
                        ids.back() = sep_token_id_;
                    }
                }
            }
            return ids;
        }

        /* Add [CLS] for BERT-style models */
        if (add_special_tokens_) {
            ids.push_back(cls_token_id_);
        }

        tokenize_body(text, ids);

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

        /* HF: a word of more than max_input_chars_per_word characters is one
         * [UNK], whatever the vocabulary holds. */
        size_t nchars = 0;
        for (unsigned char c : word) if ((c & 0xC0) != 0x80) nchars++;
        if (nchars > wp_max_chars_) {
            ids.push_back(unk_token_id_);
            return;
        }

        /* Try full word first */
        auto it = vocab_.find(word);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
            return;
        }

        /* Greedy longest-match-first from left */
        size_t start = 0;
        const size_t ids_before = ids.size();
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
                    start = end;
                    break;
                }
                end--;
            }
            if (!found) {
                /* A piece is not in the vocabulary: HF discards the pieces
                 * already matched and emits one [UNK] for the whole word. */
                ids.resize(ids_before);
                ids.push_back(unk_token_id_);
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
            /* A truncated trailing sequence is kept as the bytes that are there. */
            const size_t len = std::min((size_t)bytes, word.size() - i);
            symbols.push_back(word.substr(i, len));
            i += len;
        }
        bpe_merge_symbols(symbols, ids);
    }

    /* Merge `symbols` by rank, then look each result up in the vocabulary
     * (unknown symbols become the unk id, unfused, as HF with fuse_unk=false). */
    void bpe_merge_symbols(std::vector<std::string>& symbols, std::vector<int>& ids) const {
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

    /* ---- Byte-level BPE (CLIP, GPT-2 style), HF `tokenizers` semantics ---- */

    /* One alternative of the Split regex: a literal, `\p{L}+`, `\p{N}` (one
     * character) or `[^\s\p{L}\p{N}]+`. */
    enum BlKind { BL_LITERAL, BL_LETTERS, BL_NUMBER, BL_OTHER };
    struct BlAlt { BlKind kind; std::string literal; };

    std::vector<BlAlt> bl_alts_;       /* Split alternatives, in regex order */
    bool bl_split_ = false;            /* a Split pre-tokenizer precedes ByteLevel */
    bool bl_gpt2_regex_ = true;        /* ByteLevel use_regex */
    bool bl_nfc_ = false;              /* normalizer NFC (Latin scope, see tokenizer_nfc.hpp) */
    bool bl_lower_ = false;            /* normalizer Lowercase */
    bool bl_collapse_ws_ = false;      /* normalizer Replace(\s+ -> " ") */
    std::string bl_suffix_;            /* model end_of_word_suffix */
    std::vector<std::pair<std::string, int>> bl_added_; /* special added tokens, longest first */

    /* Split a top-level regex on `|`, honouring `\` escapes and `[...]` classes. */
    static std::vector<std::string> bl_split_alternatives(const std::string& re) {
        std::vector<std::string> out;
        std::string cur;
        bool in_class = false;
        for (size_t i = 0; i < re.size(); i++) {
            const char c = re[i];
            if (c == '\\' && i + 1 < re.size()) { cur += c; cur += re[++i]; continue; }
            if (c == '[') in_class = true;
            else if (c == ']') in_class = false;
            if (c == '|' && !in_class) { out.push_back(cur); cur.clear(); continue; }
            cur += c;
        }
        out.push_back(cur);
        return out;
    }

    /* Read one Split alternative; false when it is not one we implement. */
    static bool bl_parse_alternative(const std::string& alt, BlAlt& out) {
        if (alt == "[\\p{L}]+" || alt == "\\p{L}+") { out = {BL_LETTERS, ""}; return true; }
        if (alt == "[\\p{N}]" || alt == "\\p{N}") { out = {BL_NUMBER, ""}; return true; }
        if (alt == "[^\\s\\p{L}\\p{N}]+") { out = {BL_OTHER, ""}; return true; }
        std::string lit;
        for (size_t i = 0; i < alt.size(); i++) {
            const char c = alt[i];
            if (c == '\\') {
                if (i + 1 >= alt.size() || !strchr("|.()[]{}*+?^$\\/-", alt[i + 1])) return false;
                lit += alt[++i];
            } else if (strchr(".()[]{}*+?^$", c)) {
                return false;
            } else {
                lit += c;
            }
        }
        if (lit.empty()) return false;
        out = {BL_LITERAL, lit};
        return true;
    }

    /* Recognise a CLIP / GPT-2 style tokenizer.json. NFC is accepted but not
     * applied (no composition tables): text already in NFC, the usual case,
     * is unaffected, and the fidelity corpus measures the rest. Anything else
     * unknown returns false and leaves the legacy BPE path in place. */
    bool parse_bytelevel(cJSON* root, cJSON* model) {
        if (!model) return false;
        cJSON* pre = cJSON_GetObjectItem(root, "pre_tokenizer");
        if (!pre) return false;

        std::vector<cJSON*> steps;
        cJSON* ptype = cJSON_GetObjectItem(pre, "type");
        if (ptype && ptype->valuestring && strcmp(ptype->valuestring, "Sequence") == 0) {
            cJSON* arr = cJSON_GetObjectItem(pre, "pretokenizers");
            if (!arr || !cJSON_IsArray(arr)) return false;
            for (int i = 0; i < cJSON_GetArraySize(arr); i++) steps.push_back(cJSON_GetArrayItem(arr, i));
        } else {
            steps.push_back(pre);
        }

        std::vector<BlAlt> alts;
        bool has_split = false, has_bytelevel = false, use_regex = true;
        for (cJSON* st : steps) {
            cJSON* t = cJSON_GetObjectItem(st, "type");
            if (!t || !t->valuestring) return false;
            const std::string name = t->valuestring;
            if (name == "Split" && !has_split && !has_bytelevel) {
                cJSON* pat = cJSON_GetObjectItem(st, "pattern");
                cJSON* rx = pat ? cJSON_GetObjectItem(pat, "Regex") : nullptr;
                cJSON* beh = cJSON_GetObjectItem(st, "behavior");
                cJSON* inv = cJSON_GetObjectItem(st, "invert");
                /* Keep the matches, drop what lies between them. */
                if (!rx || !rx->valuestring || !beh || !beh->valuestring ||
                    strcmp(beh->valuestring, "Removed") != 0 || !cJSON_IsTrue(inv))
                    return false;
                for (const std::string& a : bl_split_alternatives(rx->valuestring)) {
                    BlAlt alt;
                    if (!bl_parse_alternative(a, alt)) return false;
                    alts.push_back(alt);
                }
                has_split = true;
            } else if (name == "ByteLevel" && !has_bytelevel) {
                cJSON* aps = cJSON_GetObjectItem(st, "add_prefix_space");
                if (aps && cJSON_IsTrue(aps)) return false;
                cJSON* ur = cJSON_GetObjectItem(st, "use_regex");
                if (ur && cJSON_IsFalse(ur)) use_regex = false;
                has_bytelevel = true;
            } else {
                return false;
            }
        }
        if (!has_bytelevel) return false;

        /* Normalizer: NFC (ignored), Lowercase, Replace(`\s+` -> " ") only. */
        bool lower = false, collapse = false, nfc = false;
        cJSON* norm = cJSON_GetObjectItem(root, "normalizer");
        if (norm && !cJSON_IsNull(norm)) {
            std::vector<cJSON*> ns;
            cJSON* nt = cJSON_GetObjectItem(norm, "type");
            if (nt && nt->valuestring && strcmp(nt->valuestring, "Sequence") == 0) {
                cJSON* arr = cJSON_GetObjectItem(norm, "normalizers");
                if (!arr || !cJSON_IsArray(arr)) return false;
                for (int i = 0; i < cJSON_GetArraySize(arr); i++) ns.push_back(cJSON_GetArrayItem(arr, i));
            } else {
                ns.push_back(norm);
            }
            for (cJSON* n : ns) {
                cJSON* t = cJSON_GetObjectItem(n, "type");
                if (!t || !t->valuestring) return false;
                const std::string name = t->valuestring;
                if (name == "NFC") { nfc = true; continue; }
                if (name == "Lowercase") { lower = true; continue; }
                if (name == "Replace") {
                    cJSON* pat = cJSON_GetObjectItem(n, "pattern");
                    cJSON* rx = pat ? cJSON_GetObjectItem(pat, "Regex") : nullptr;
                    cJSON* content = cJSON_GetObjectItem(n, "content");
                    if (rx && rx->valuestring && strcmp(rx->valuestring, "\\s+") == 0 &&
                        content && content->valuestring && strcmp(content->valuestring, " ") == 0) {
                        collapse = true;
                        continue;
                    }
                }
                return false;
            }
        }

        /* Model: a continuing-subword prefix is not implemented. */
        cJSON* csp = cJSON_GetObjectItem(model, "continuing_subword_prefix");
        if (csp && csp->valuestring && csp->valuestring[0]) return false;
        cJSON* eow = cJSON_GetObjectItem(model, "end_of_word_suffix");

        bl_alts_ = alts;
        bl_split_ = has_split;
        bl_gpt2_regex_ = use_regex;
        bl_nfc_ = nfc;
        bl_lower_ = lower;
        bl_collapse_ws_ = collapse;
        bl_suffix_ = (eow && eow->valuestring) ? eow->valuestring : "";

        /* Special added tokens are cut out of the text before anything else. */
        bl_added_.clear();
        cJSON* added = cJSON_GetObjectItem(root, "added_tokens");
        if (added && cJSON_IsArray(added)) {
            for (int i = 0; i < cJSON_GetArraySize(added); i++) {
                cJSON* at = cJSON_GetArrayItem(added, i);
                cJSON* content = cJSON_GetObjectItem(at, "content");
                cJSON* id = cJSON_GetObjectItem(at, "id");
                cJSON* special = cJSON_GetObjectItem(at, "special");
                if (content && content->valuestring && content->valuestring[0] &&
                    id && cJSON_IsNumber(id) && special && cJSON_IsTrue(special))
                    bl_added_.emplace_back(content->valuestring, id->valueint);
            }
            std::stable_sort(bl_added_.begin(), bl_added_.end(),
                             [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                                 return a.first.size() > b.first.size();
                             });
        }
        return true;
    }

    /* NFC over the whole text; malformed UTF-8 is left untouched. */
    static std::string bl_apply_nfc(const std::string& text) {
        std::vector<uint32_t> cps;
        cps.reserve(text.size());
        for (size_t i = 0; i < text.size(); ) {
            uint32_t cp;
            i += utf8_next(text, i, cp);
            if (cp == kInvalidCp) return text;
            cps.push_back(cp);
        }
        unicode::nfc_compose(cps);
        std::string out;
        out.reserve(text.size());
        for (uint32_t cp : cps) unicode::append_utf8(out, cp);
        return out;
    }

    /* Normalizer: NFC, collapse whitespace runs to one space, lowercase. */
    std::string bl_normalize(const std::string& raw) const {
        const std::string text = bl_nfc_ ? bl_apply_nfc(raw) : raw;
        std::string out;
        out.reserve(text.size());
        bool prev_space = false;
        for (size_t i = 0; i < text.size(); ) {
            uint32_t cp;
            const size_t n = utf8_next(text, i, cp);
            if (cp == kInvalidCp) {
                out.append(text, i, n);
                prev_space = false;
            } else if (bl_collapse_ws_ && unicode::is_space(cp)) {
                if (!prev_space) out += ' ';
                prev_space = true;
            } else if (bl_lower_ && cp == 0x130) {
                /* The only multi-codepoint lowercase: i + combining dot above. */
                out += 'i';
                unicode::append_utf8(out, 0x307);
                prev_space = false;
            } else {
                if (bl_lower_) cp = (cp >= 'A' && cp <= 'Z') ? cp + 0x20 : unicode::bert_lower(cp);
                unicode::append_utf8(out, cp);
                prev_space = false;
            }
            i += n;
        }
        return out;
    }

    /* ---- Non byte-level BPE with the `Whitespace` pre-tokenizer ----
     * (jina-reranker-v1-turbo-en style: normalizer NFC + Lowercase, characters
     * are vocabulary symbols). HF splits on `\w+|[^\w\s]+` and lowercases with
     * full Unicode rules; the legacy path made every non-ASCII character a word
     * of its own, which diverged on Cyrillic, CJK and other non-Latin text. */
    bool ws_bpe_ = false;
    bool ws_nfc_ = false;

    /* Enable the path only when pre_tokenizer and normalizer are fully understood. */
    void parse_whitespace_bpe(cJSON* root, cJSON* model) {
        ws_bpe_ = false;
        cJSON* pre = cJSON_GetObjectItem(root, "pre_tokenizer");
        cJSON* pt = pre ? cJSON_GetObjectItem(pre, "type") : nullptr;
        if (!pt || !pt->valuestring || strcmp(pt->valuestring, "Whitespace") != 0) return;
        cJSON* csp = model ? cJSON_GetObjectItem(model, "continuing_subword_prefix") : nullptr;
        cJSON* eow = model ? cJSON_GetObjectItem(model, "end_of_word_suffix") : nullptr;
        if ((csp && csp->valuestring && csp->valuestring[0]) ||
            (eow && eow->valuestring && eow->valuestring[0])) return;

        bool lower = false, nfc = false;
        cJSON* norm = cJSON_GetObjectItem(root, "normalizer");
        if (norm && !cJSON_IsNull(norm)) {
            std::vector<cJSON*> ns;
            cJSON* nt = cJSON_GetObjectItem(norm, "type");
            if (nt && nt->valuestring && strcmp(nt->valuestring, "Sequence") == 0) {
                cJSON* arr = cJSON_GetObjectItem(norm, "normalizers");
                if (!arr || !cJSON_IsArray(arr)) return;
                for (int i = 0; i < cJSON_GetArraySize(arr); i++) ns.push_back(cJSON_GetArrayItem(arr, i));
            } else {
                ns.push_back(norm);
            }
            for (cJSON* n : ns) {
                cJSON* t = cJSON_GetObjectItem(n, "type");
                if (!t || !t->valuestring) return;
                if (strcmp(t->valuestring, "NFC") == 0) nfc = true;
                else if (strcmp(t->valuestring, "Lowercase") == 0) lower = true;
                else return;
            }
        }
        ws_bpe_ = true;
        ws_nfc_ = nfc;
        bl_lower_ = lower;   /* reused by bl_normalize */
    }

    /* Regex `\w`: letter, number, underscore, combining mark (Mn approximation). */
    static bool ws_is_word(uint32_t cp) {
        return cp == '_' || unicode::is_letter(cp) || unicode::is_number(cp) ||
               unicode::bert_is_stripped(cp);
    }

    /* NFC + Lowercase, then `\w+|[^\w\s]+` words. */
    void ws_bpe_tokenize(const std::string& raw, std::vector<int>& ids) const {
        const std::string s = (ws_nfc_ ? bl_apply_nfc(raw) : raw);
        std::string text;
        text.reserve(s.size());
        for (size_t i = 0; i < s.size(); ) {
            uint32_t cp;
            const size_t n = utf8_next(s, i, cp);
            if (cp == kInvalidCp) { text.append(s, i, n); }
            else if (bl_lower_ && cp == 0x130) { text += 'i'; unicode::append_utf8(text, 0x307); }
            else {
                if (bl_lower_) cp = (cp >= 'A' && cp <= 'Z') ? cp + 0x20 : unicode::bert_lower(cp);
                unicode::append_utf8(text, cp);
            }
            i += n;
        }
        size_t i = 0;
        while (i < text.size()) {
            uint32_t cp;
            size_t n = utf8_next(text, i, cp);
            if (cp != kInvalidCp && unicode::is_space(cp)) { i += n; continue; }
            const bool word = cp != kInvalidCp && ws_is_word(cp);
            size_t end = i + n;
            while (end < text.size()) {
                uint32_t c2;
                const size_t m = utf8_next(text, end, c2);
                if (c2 == kInvalidCp || unicode::is_space(c2) || ws_is_word(c2) != word) break;
                end += m;
            }
            bpe_tokenize(text.substr(i, end - i), ids);
            i = end;
        }
    }

    /* Length in bytes of the codepoint at `i` and its class. */
    enum BlClass { BLC_LETTER, BLC_NUMBER, BLC_SPACE, BLC_OTHER };
    static BlClass bl_class(const std::string& s, size_t i, size_t& n) {
        uint32_t cp;
        n = utf8_next(s, i, cp);
        if (cp == kInvalidCp) return BLC_OTHER;
        if (unicode::is_letter(cp)) return BLC_LETTER;
        if (unicode::is_number(cp)) return BLC_NUMBER;
        if (unicode::is_space(cp)) return BLC_SPACE;
        return BLC_OTHER;
    }

    /* Split regex of `pre_tokenizer`: byte spans of the matches in `s`. */
    void bl_split_matches(const std::string& s, std::vector<std::pair<size_t, size_t>>& spans) const {
        size_t i = 0;
        while (i < s.size()) {
            size_t n;
            const BlClass c = bl_class(s, i, n);
            size_t end = 0;
            for (const BlAlt& a : bl_alts_) {
                if (a.kind == BL_LITERAL) {
                    if (s.compare(i, a.literal.size(), a.literal) == 0) end = i + a.literal.size();
                } else if (a.kind == BL_NUMBER) {
                    if (c == BLC_NUMBER) end = i + n;
                } else {
                    const BlClass want = a.kind == BL_LETTERS ? BLC_LETTER : BLC_OTHER;
                    if (c == want) {
                        end = i + n;
                        size_t m;
                        while (end < s.size() && bl_class(s, end, m) == want) end += m;
                    }
                }
                if (end) break;
            }
            if (end) { spans.emplace_back(i, end); i = end; }
            else i += n;     /* between matches: removed */
        }
    }

    /* GPT-2 pre-tokenizer regex (ByteLevel use_regex) over s[a, b):
     * 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ */
    static void bl_gpt2_split(const std::string& s, size_t a, size_t b,
                              std::vector<std::pair<size_t, size_t>>& spans) {
        static const char* const kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        size_t i = a;
        while (i < b) {
            size_t end = 0;
            for (const char* c : kContractions) {
                const size_t len = strlen(c);
                if (i + len <= b && s.compare(i, len, c) == 0) { end = i + len; break; }
            }
            if (!end) {
                size_t j = i;
                if (s[j] == ' ') j++;
                size_t n = 0;
                if (j < b) {
                    const BlClass c = bl_class(s, j, n);
                    if (c != BLC_SPACE) {
                        end = j + n;
                        size_t m;
                        while (end < b && bl_class(s, end, m) == c) end += m;
                    }
                }
            }
            if (!end) {
                /* \s+(?!\S): leave the last whitespace for the next token when
                 * a non-space follows; otherwise \s+ takes the whole run. */
                size_t k = i, last = i, n;
                while (k < b && bl_class(s, k, n) == BLC_SPACE) { last = k; k += n; }
                end = (k == b || last == i) ? k : last;
            }
            spans.emplace_back(i, end);
            i = end;
        }
    }

    /* BPE of one pre-token: bytes -> GPT-2 characters, suffix on the last one. */
    void bl_encode_piece(const std::string& s, size_t a, size_t b, std::vector<int>& ids) const {
        std::vector<std::string> symbols;
        symbols.reserve(b - a);
        for (size_t i = a; i < b; i++) {
            std::string sym;
            unicode::append_utf8(sym, unicode::bytelevel_cp((unsigned char)s[i]));
            symbols.push_back(sym);
        }
        if (symbols.empty()) return;
        symbols.back() += bl_suffix_;
        bpe_merge_symbols(symbols, ids);
    }

    void bl_encode_segment(const std::string& raw, std::vector<int>& ids) const {
        const std::string s = bl_normalize(raw);
        std::vector<std::pair<size_t, size_t>> pieces;
        if (bl_split_) bl_split_matches(s, pieces);
        else if (!s.empty()) pieces.emplace_back(0, s.size());
        for (const auto& p : pieces) {
            if (bl_gpt2_regex_) {
                std::vector<std::pair<size_t, size_t>> sub;
                bl_gpt2_split(s, p.first, p.second, sub);
                for (const auto& q : sub) bl_encode_piece(s, q.first, q.second, ids);
            } else {
                bl_encode_piece(s, p.first, p.second, ids);
            }
        }
    }

    void tokenize_bytelevel(const std::string& text, std::vector<int>& ids) const {
        size_t seg = 0, i = 0;
        while (i < text.size() && !bl_added_.empty()) {
            const std::pair<std::string, int>* hit = nullptr;
            for (const auto& t : bl_added_)
                if (text.compare(i, t.first.size(), t.first) == 0) { hit = &t; break; }
            if (!hit) { i++; continue; }
            if (i > seg) bl_encode_segment(text.substr(seg, i - seg), ids);
            ids.push_back(hit->second);
            i += hit->first.size();
            seg = i;
        }
        if (seg < text.size()) bl_encode_segment(seg ? text.substr(seg) : text, ids);
    }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP */

