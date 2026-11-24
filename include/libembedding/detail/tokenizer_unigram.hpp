/*
 * libembedding - detail/tokenizer_unigram.hpp
 * Unigram (SentencePiece) tokenizer pieces for tokenizer.json:
 *   - Precompiled normalizer (SentencePiece charsmap: base64 -> double-array
 *     trie + replacement pool, longest-match replacement);
 *   - Metaspace pre-tokenizer;
 *   - Unigram model (Viterbi over a byte trie, unk fusion, byte fallback).
 *
 * Auteur: David Orel
 * Version: 1.16.3
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_TOKENIZER_UNIGRAM_HPP
#define LIBEMBEDDING_DETAIL_TOKENIZER_UNIGRAM_HPP

#include "cJSON.h"
#include "tokenizer_unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* ---------------------------------------------------------------------------
 * SentencePiece precompiled charsmap
 * Blob layout: [u32 trie_bytes][trie_bytes of u32 double-array units][pool].
 * A trie value is the offset of a NUL-terminated replacement in the pool.
 * ------------------------------------------------------------------------ */
class PrecompiledCharsmap {
public:
    bool empty() const { return units_.empty(); }

    void load_base64(const char* b64) {
        std::string raw = base64_decode(b64);
        if (raw.size() < 4) throw std::runtime_error("Precompiled charsmap: blob too short");
        uint32_t trie_bytes = rd32(raw.data());
        if ((size_t)trie_bytes + 4 > raw.size() || trie_bytes % 4 != 0)
            throw std::runtime_error("Precompiled charsmap: bad trie size");
        units_.resize(trie_bytes / 4);
        for (size_t i = 0; i < units_.size(); i++) units_[i] = rd32(raw.data() + 4 + i * 4);
        pool_.assign(raw, 4 + trie_bytes, std::string::npos);
    }

    /* Same walk as HF's spm_precompiled: split into grapheme-like clusters
     * (a base character plus following combining marks / ZWJ); a cluster
     * shorter than 6 bytes is replaced as a whole if the map has exactly that
     * key, otherwise (and for longer clusters) each character is looked up on
     * its own. Unmatched characters are copied through. Clusters are
     * approximated with the Mn table of tokenizer_unicode.hpp. */
    std::string normalize(const std::string& s) const {
        if (units_.empty()) return s;
        std::string out;
        out.reserve(s.size() + 8);
        size_t i = 0;
        while (i < s.size()) {
            const size_t first = char_len(s, i);
            size_t end = i + first;
            while (end < s.size() && is_extend(s, end)) end += char_len(s, end);
            if (end - i < 6 && exact(s, i, end - i, out)) {
                i = end;
                continue;
            }
            while (i < end) {
                const size_t n = char_len(s, i);
                if (!exact(s, i, n, out)) out.append(s, i, n);
                i += n;
            }
        }
        return out;
    }

private:
    static size_t char_len(const std::string& s, size_t i) {
        size_t n = 1;
        const unsigned char c = (unsigned char)s[i];
        if (c >= 0xF0) n = 4; else if (c >= 0xE0) n = 3; else if (c >= 0xC0) n = 2;
        return std::min(n, s.size() - i);
    }
    static bool is_extend(const std::string& s, size_t i) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0xC0) return false;   /* ASCII is never a combining mark */
        const size_t n = char_len(s, i);
        uint32_t cp = c;
        if (n == 2) cp = ((c & 0x1Fu) << 6) | ((unsigned char)s[i + 1] & 0x3Fu);
        else if (n == 3) cp = ((c & 0x0Fu) << 12) | (((unsigned char)s[i + 1] & 0x3Fu) << 6) | ((unsigned char)s[i + 2] & 0x3Fu);
        else if (n == 4) cp = ((c & 0x07u) << 18) | (((unsigned char)s[i + 1] & 0x3Fu) << 12) |
                              (((unsigned char)s[i + 2] & 0x3Fu) << 6) | ((unsigned char)s[i + 3] & 0x3Fu);
        return cp == 0x200D || unicode::bert_is_stripped(cp);
    }
    /* Append the replacement when [i, i+len) is exactly a key of the map. */
    bool exact(const std::string& s, size_t i, size_t len, std::string& out) const {
        size_t mlen = 0;
        uint32_t val = 0;
        longest_prefix(s.data() + i, len, mlen, val);
        if (mlen != len || val >= pool_.size()) return false;
        out.append(pool_.c_str() + val);
        return true;
    }

    std::vector<uint32_t> units_;
    std::string pool_;

    static uint32_t rd32(const char* p) {
        const unsigned char* u = (const unsigned char*)p;
        return (uint32_t)u[0] | ((uint32_t)u[1] << 8) | ((uint32_t)u[2] << 16) | ((uint32_t)u[3] << 24);
    }

    struct B64Table {
        signed char v[256];
        B64Table() {
            for (int i = 0; i < 256; i++) v[i] = -1;
            const char* al = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            for (int i = 0; i < 64; i++) v[(unsigned char)al[i]] = (signed char)i;
        }
    };

    static std::string base64_decode(const char* in) {
        /* Function-local static: initialised once, thread-safe (C++11), so two
         * tokenizers loaded concurrently do not race on the table. */
        static const B64Table table;
        const signed char* tab = table.v;
        std::string out;
        out.reserve(strlen(in) / 4 * 3);
        uint32_t acc = 0;
        int bits = 0;
        for (; *in; in++) {
            const signed char v = tab[(unsigned char)*in];
            if (v < 0) continue;   /* padding, whitespace */
            acc = (acc << 6) | (uint32_t)v;
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                out.push_back((char)((acc >> bits) & 0xFF));
            }
        }
        return out;
    }

    /* darts-clone traversal, as in SentencePiece. */
    void longest_prefix(const char* key, size_t len, size_t& best_len, uint32_t& best_val) const {
        best_len = 0;
        best_val = 0;
        const size_t n = units_.size();
        size_t pos = 0;
        uint32_t unit = units_[0];
        pos ^= offset_of(unit);
        for (size_t i = 0; i < len; i++) {
            const uint32_t k = (unsigned char)key[i];
            if (k == 0) return;
            pos ^= k;
            if (pos >= n) return;
            unit = units_[pos];
            if ((unit & 0x800000FFu) != k) return;
            pos ^= offset_of(unit);
            if ((unit >> 8) & 1u) {
                if (pos >= n) return;
                best_len = i + 1;
                best_val = units_[pos] & 0x7FFFFFFFu;
            }
        }
    }
    static size_t offset_of(uint32_t u) { return (size_t)((u >> 10) << ((u & 0x200u) >> 6)); }
};

/* ---------------------------------------------------------------------------
 * Metaspace pre-tokenizer settings
 * ------------------------------------------------------------------------ */
struct MetaspaceConfig {
    bool enabled = false;
    std::string replacement = "\xE2\x96\x81";   /* U+2581 */
    bool prepend = true;                        /* "always"/"first" or add_prefix_space */
    bool split = true;
    /* A WhitespaceSplit before Metaspace in a Sequence pre-tokenizer
     * (paraphrase-multilingual): split on Unicode whitespace, drop it, and
     * apply Metaspace to each word on its own. */
    bool whitespace_split = false;
};

/* Rust char::is_whitespace on the UTF-8 sequence at s[i]; `len` receives the
 * byte length of that character (whitespace or not). */
inline bool utf8_is_whitespace(const std::string& s, size_t i, size_t& len) {
    const unsigned char c = (unsigned char)s[i];
    len = 1;
    if (c < 0x80) return c == ' ' || (c >= 0x09 && c <= 0x0D);
    uint32_t cp = 0;
    if (c >= 0xF0) { len = 4; cp = c & 0x07u; }
    else if (c >= 0xE0) { len = 3; cp = c & 0x0Fu; }
    else if (c >= 0xC0) { len = 2; cp = c & 0x1Fu; }
    else return false;   /* stray continuation byte */
    if (i + len > s.size()) { len = s.size() - i; return false; }
    for (size_t k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3Fu);
    return cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

/* ---------------------------------------------------------------------------
 * Unigram model
 * ------------------------------------------------------------------------ */
class UnigramModel {
public:
    bool loaded() const { return !scores_.empty(); }
    int unk_id() const { return unk_id_; }

    /* `model` is the tokenizer.json "model" object (type Unigram). */
    void load(const cJSON* model) {
        const cJSON* vocab = cJSON_GetObjectItem(const_cast<cJSON*>(model), "vocab");
        if (!cJSON_IsArray(vocab)) throw std::runtime_error("Unigram: missing vocab array");
        const int n = cJSON_GetArraySize(vocab);
        pieces_.reserve((size_t)n);
        scores_.reserve((size_t)n);
        double min_score = std::numeric_limits<double>::infinity();
        for (const cJSON* e = vocab->child; e; e = e->next) {
            const cJSON* p = e->child;
            const cJSON* s = p ? p->next : nullptr;
            if (!p || !p->valuestring || !cJSON_IsNumber(s))
                throw std::runtime_error("Unigram: malformed vocab entry");
            pieces_.emplace_back(p->valuestring);
            scores_.push_back(s->valuedouble);
            min_score = std::min(min_score, s->valuedouble);
        }
        const cJSON* unk = cJSON_GetObjectItem(const_cast<cJSON*>(model), "unk_id");
        unk_id_ = cJSON_IsNumber(unk) ? unk->valueint : -1;
        byte_fallback_ = cJSON_IsTrue(cJSON_GetObjectItem(const_cast<cJSON*>(model), "byte_fallback"));
        unk_score_ = (scores_.empty() ? 0.0 : min_score) - 10.0;   /* HF K_UNK_PENALTY */
        build_trie();
        if (byte_fallback_) {
            for (int b = 0; b < 256; b++) {
                char name[8];
                snprintf(name, sizeof name, "<0x%02X>", b);
                byte_ids_[b] = find_exact(name, strlen(name));
            }
        }
    }

    /* Segment one pre-token (already normalised, Metaspace applied) and append
     * the ids. */
    void encode_word(const std::string& w, std::vector<int>& ids) const {
        const size_t n = w.size();
        if (n == 0) return;
        /* Per-call buffers (not members): encode stays const and thread-safe. */
        std::vector<BestNode> best(n + 1, BestNode{0.0, -1, -1});
        std::vector<Seg> path;
        best[0].start = 0;
        const unsigned char* d = (const unsigned char*)w.data();
        size_t i = 0;
        while (i < n) {
            size_t cl = 1;
            if (d[i] >= 0xF0) cl = 4; else if (d[i] >= 0xE0) cl = 3; else if (d[i] >= 0xC0) cl = 2;
            cl = std::min(cl, n - i);
            const double base = best[i].score;
            bool single = false;
            uint32_t node = 0;
            for (size_t j = i; j < n; j++) {
                node = child(node, d[j]);
                if (node == kNone) break;
                const int32_t id = node_id_[node];
                if (id < 0) continue;
                const size_t end = j + 1;
                if (end - i == cl) single = true;
                const double cand = scores_[(size_t)id] + base;
                BestNode& t = best[end];
                if (t.start < 0 || cand > t.score) t = BestNode{cand, (int32_t)i, id};
            }
            if (!single) {
                const double cand = unk_score_ + base;
                BestNode& t = best[i + cl];
                if (t.start < 0 || cand > t.score) t = BestNode{cand, (int32_t)i, unk_id_};
            }
            i += cl;
        }
        /* Backtrack, then emit forwards: consecutive unks fuse into one. */
        for (size_t e = n; e > 0;) {
            const BestNode& nd = best[e];
            path.push_back({(size_t)nd.start, e, nd.id});
            e = (size_t)nd.start;
        }
        for (size_t k = path.size(); k-- > 0;) {
            const Seg& sg = path[k];
            if (sg.id == unk_id_) {
                size_t last = sg.end;
                while (k > 0 && path[k - 1].id == unk_id_) {
                    k--;
                    last = path[k].end;
                }
                emit_unk(d + sg.start, last - sg.start, ids);
            } else {
                ids.push_back(sg.id);
            }
        }
    }

private:
    struct Seg { size_t start, end; int32_t id; };
    static constexpr uint32_t kNone = 0xFFFFFFFFu;

    std::vector<std::string> pieces_;
    std::vector<double> scores_;
    int unk_id_ = -1;
    double unk_score_ = 0.0;
    bool byte_fallback_ = false;
    int byte_ids_[256] = {};

    /* Flat byte trie: node n owns edges [edge_begin_[n], edge_end_[n]),
     * sorted by byte. */
    std::vector<uint32_t> edge_begin_;
    std::vector<unsigned char> edge_byte_;
    std::vector<uint32_t> edge_child_;
    std::vector<int32_t> node_id_;

    std::vector<uint32_t> edge_end_;
    struct BestNode { double score; int32_t start; int32_t id; };

    void emit_unk(const unsigned char* p, size_t len, std::vector<int>& ids) const {
        if (byte_fallback_) {
            bool ok = true;
            for (size_t i = 0; i < len; i++)
                if (byte_ids_[p[i]] < 0) { ok = false; break; }
            if (ok) {
                for (size_t i = 0; i < len; i++) ids.push_back(byte_ids_[p[i]]);
                return;
            }
        }
        /* HF raises "Encountered an unknown token but unk_id is missing";
         * pushing -1 would hand an out-of-range id to the model's gather. */
        if (unk_id_ < 0)
            throw std::runtime_error("Unigram: unknown token and the model has no unk_id");
        ids.push_back(unk_id_);
    }

    uint32_t child(uint32_t node, unsigned char b) const {
        uint32_t lo = edge_begin_[node], hi = edge_end_[node];
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (edge_byte_[mid] < b) lo = mid + 1;
            else if (edge_byte_[mid] > b) hi = mid;
            else return edge_child_[mid];
        }
        return kNone;
    }

    int find_exact(const char* s, size_t len) const {
        uint32_t node = 0;
        for (size_t i = 0; i < len; i++) {
            node = child(node, (unsigned char)s[i]);
            if (node == kNone) return -1;
        }
        return node_id_[node];
    }

    void build_trie() {
        std::vector<uint32_t> order;
        order.reserve(pieces_.size());
        for (uint32_t i = 0; i < pieces_.size(); i++)
            if (!pieces_[i].empty()) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return pieces_[a] < pieces_[b];
        });
        /* Work list of (node, lo, hi, depth); children of a node are laid out
         * contiguously when the node is expanded. */
        struct Job { uint32_t node, lo, hi, depth; };
        std::vector<Job> stack;
        std::vector<uint32_t> first_edge(1, 0), edge_count(1, 0);
        node_id_.assign(1, -1);
        stack.push_back({0, 0, (uint32_t)order.size(), 0});
        while (!stack.empty()) {
            const Job j = stack.back();
            stack.pop_back();
            uint32_t lo = j.lo;
            /* a piece that ends exactly at this depth is this node's piece */
            while (lo < j.hi && pieces_[order[lo]].size() == j.depth) {
                if (node_id_[j.node] < 0) node_id_[j.node] = (int32_t)order[lo];
                lo++;
            }
            first_edge[j.node] = (uint32_t)edge_byte_.size();
            std::vector<Job> kids;
            while (lo < j.hi) {
                const unsigned char b = (unsigned char)pieces_[order[lo]][j.depth];
                uint32_t hi = lo + 1;
                while (hi < j.hi && (unsigned char)pieces_[order[hi]][j.depth] == b) hi++;
                const uint32_t c = (uint32_t)node_id_.size();
                node_id_.push_back(-1);
                first_edge.push_back(0);
                edge_count.push_back(0);
                edge_byte_.push_back(b);
                edge_child_.push_back(c);
                kids.push_back({c, lo, hi, j.depth + 1});
                lo = hi;
            }
            edge_count[j.node] = (uint32_t)kids.size();
            for (const Job& k : kids) stack.push_back(k);
        }
        /* edges of a node are contiguous but nodes were expanded out of id
         * order, so store explicit begin/end instead of a prefix array. */
        const size_t nn = node_id_.size();
        edge_begin_.assign(nn, 0);
        edge_end_.assign(nn, 0);
        for (size_t k = 0; k < nn; k++) {
            edge_begin_[k] = first_edge[k];
            edge_end_[k] = first_edge[k] + edge_count[k];
        }
    }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_TOKENIZER_UNIGRAM_HPP */
