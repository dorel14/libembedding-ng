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
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_TOKENIZER_IMPL_HPP

#include "cJSON.h"

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
            /* punctuation â€” separate token */
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            current += (char)c;
            tokens.push_back(current);
            current.clear();
            i++;
        } else if (c >= 0x80) {
            /* multi-byte UTF-8 â€” keep as single token */
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

    /* Public: encode a single text to token IDs (for length bucketing) */
    std::vector<int> encode(const std::string& text) const {
        return encode_single(text);
    }

    int max_length() const { return max_length_; }
    void set_pad_token_id(int id) { pad_token_id_ = id; }

private:
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
        std::string processed = text;
        if (do_lower_case_) processed = to_lower_ascii(processed);

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
                /* Character not in vocab â€” use [UNK] for entire word */
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




