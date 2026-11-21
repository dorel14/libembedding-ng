/*
 * test_tokenizer_fidelity.cpp - libembedding's tokenizer against HF `tokenizers`
 * on a multilingual corpus (roadmap T26).
 *
 * The reference outputs live in tests/data/tokenizer/ and are produced by
 * scripts/gen_tokenizer_fixtures.py with the HF `tokenizers` package; this test
 * only reads JSON, so neither Python nor the HF package is needed to run it.
 *
 * Three layers, from hermetic to model-dependent:
 *
 *  1. Normaliser, per codepoint. bert_normalize() is compared with HF's
 *     BertNormalizer(strip_accents + lowercase) on every codepoint from U+0080
 *     to U+2FFFF. Blocks listed as covered in kBlocks must match exactly; the
 *     others are counted and reported, so the remaining gap stays visible.
 *  2. Pre-tokenisation, per corpus text. The words basic_tokenize() produces
 *     after normalisation are compared with BertNormalizer + BertPreTokenizer,
 *     uncased and cased. Hermetic, must match exactly.
 *  3. Token ids, per tokenizer family of the registry. Needs that family's
 *     tokenizer.json in the model cache ($LIBEMBEDDING_CACHE_DIR, else
 *     ~/.cache/libembedding) with the SHA-256 the fixture was generated from;
 *     a missing or different file skips the family. Families libembedding
 *     implements (WordPiece, Unigram) must match exactly; the others (byte-level BPE) are
 *     only measured when LEMBED_TOKENIZER_REPORT_ALL=1, because their 17 MB
 *     vocabularies make a Debug run take minutes.
 *
 * Usage: test_tokenizer_fidelity [fixtures_dir]
 * The directory is argv[1], else $LEMBED_TOKENIZER_FIXTURES, else the source
 * tree path compiled in.
 *
 * Auteur: David Orel
 * Version: 1.14.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/detail/tokenizer_impl.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef LEMBED_TOKENIZER_FIXTURES_DEFAULT
#define LEMBED_TOKENIZER_FIXTURES_DEFAULT "tests/data/tokenizer"
#endif

static int g_pass = 0;
static int g_fail = 0;

static bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static std::string utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

/* Printable form of a UTF-8 string for failure messages: non-ASCII as U+XXXX,
 * so a console in any code page shows something readable. */
static std::string show(const std::string& s) {
    std::string r;
    char buf[16];
    for (size_t i = 0; i < s.size();) {
        uint32_t cp = 0;
        i += lembed::detail::utf8_next(s, i, cp);
        if (cp >= 0x20 && cp < 0x7F) {
            r += (char)cp;
        } else {
            snprintf(buf, sizeof(buf), "<U+%04X>", (unsigned)cp);
            r += buf;
        }
    }
    return r;
}

/* ==========================================================================
 * Layer 1 - normaliser, per codepoint
 * ========================================================================== */

struct Block {
    uint32_t first, last;
    const char* name;
    bool covered;   /* must match HF exactly */
};

/* Coarse Unicode blocks, for reporting. `covered` is the documented scope of
 * bert_normalize(); flipping one to true is how coverage is extended. */
static const Block kBlocks[] = {
    {0x0080, 0x00FF, "Latin-1 Supplement", true},
    {0x0100, 0x017F, "Latin Extended-A", true},
    {0x0180, 0x024F, "Latin Extended-B", true},
    {0x0250, 0x02FF, "IPA / spacing modifiers", true},
    {0x0300, 0x036F, "Combining diacritics", true},
    {0x0370, 0x03FF, "Greek and Coptic", true},
    {0x0400, 0x052F, "Cyrillic (+ Supplement)", true},
    {0x0530, 0x058F, "Armenian", true},
    {0x0590, 0x05FF, "Hebrew", true},
    {0x0600, 0x06FF, "Arabic", true},
    {0x0700, 0x08FF, "Syriac .. Arabic Ext-A", false},
    {0x0900, 0x097F, "Devanagari", true},
    {0x0980, 0x0DFF, "Other Indic", false},
    {0x0E00, 0x0E7F, "Thai", true},
    {0x0E80, 0x10FF, "Lao .. Georgian", false},
    {0x1100, 0x11FF, "Hangul Jamo", true},
    {0x1200, 0x1DFF, "Ethiopic .. Combining Suppl.", false},
    {0x1E00, 0x1EFF, "Latin Extended Additional", true},
    {0x1F00, 0x1FFF, "Greek Extended", true},
    {0x2000, 0x2BFF, "Punctuation, symbols", true},
    {0x2C00, 0x2FFF, "Glagolitic .. Kangxi", false},
    {0x3000, 0x30FF, "CJK punct., kana", true},
    {0x3100, 0x9FFF, "CJK ideographs ..", true},
    {0xA000, 0xABFF, "Yi .. Meetei", false},
    {0xAC00, 0xD7A3, "Hangul syllables", true},
    {0xD7A4, 0xF8FF, "Jamo Ext-B, private use", false},
    {0xF900, 0xFAFF, "CJK compat. ideographs", false},
    {0xFB00, 0xFEFF, "Presentation forms", false},
    {0xFF00, 0xFFEF, "Halfwidth / fullwidth", true},
    {0xFFF0, 0xFFFF, "Specials", false},
    {0x10000, 0x1F0FF, "SMP (scripts, symbols)", false},
    {0x1F100, 0x1FAFF, "SMP (emoji)", true},
    {0x1FB00, 0x2FFFF, "SMP/SIP (rest)", false},
};

/* `lower` / `strip` select the BertNormalizer mode the fixture was generated
 * with: both (hf_normalizer.json), lowercase only, or strip_accents only.
 * `all_covered` makes every block binding (the lowercase table is complete). */
static void test_normalizer(const std::string& dir, const char* file, const char* label,
                            bool lower, bool strip, bool all_covered) {
    std::string blob;
    if (!read_file(dir + "/" + file, blob)) {
        fprintf(stderr, "FAIL: cannot read %s/%s\n", dir.c_str(), file);
        g_fail++;
        return;
    }
    cJSON* root = cJSON_Parse(blob.c_str());
    cJSON* changed = root ? cJSON_GetObjectItem(root, "changed") : nullptr;
    cJSON* step_item = root ? cJSON_GetObjectItem(root, "hangul_step") : nullptr;
    if (!changed || !cJSON_IsNumber(step_item)) {
        fprintf(stderr, "FAIL: %s is malformed\n", file);
        g_fail++;
        cJSON_Delete(root);
        return;
    }
    const uint32_t hangul_step = (uint32_t)step_item->valueint;
    std::map<uint32_t, std::string> expected;
    for (cJSON* it = changed->child; it; it = it->next) {
        if (it->string && cJSON_IsString(it))
            expected[(uint32_t)strtoul(it->string, nullptr, 16)] = it->valuestring;
    }
    cJSON_Delete(root);

    printf("\n[1] Normaliser vs BertNormalizer (%s), per codepoint\n", label);
    printf("    %-30s %8s %8s %9s  %s\n", "block", "checked", "differ", "rate", "scope");
    long total_checked = 0, total_differ = 0;
    for (const Block& b : kBlocks) {
        long checked = 0, differ = 0;
        int shown = 0;
        for (uint32_t cp = b.first; cp <= b.last; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) continue;
            if (cp >= 0xAC00 && cp <= 0xD7A3 && (cp - 0xAC00) % hangul_step) continue;
            const std::string in = utf8(cp);
            auto e = expected.find(cp);
            const std::string& want = e != expected.end() ? e->second : in;
            const std::string got = lembed::detail::bert_normalize(in, lower, strip);
            const bool covered = b.covered || all_covered;
            checked++;
            if (got != want) {
                differ++;
                if (covered && shown < 5) {
                    fprintf(stderr, "FAIL: normalise U+%04X: got \"%s\", want \"%s\"\n",
                            (unsigned)cp, show(got).c_str(), show(want).c_str());
                    shown++;
                }
            }
        }
        total_checked += checked;
        total_differ += differ;
        printf("    %-30s %8ld %8ld %8.2f%%  %s\n", b.name, checked, differ,
               checked ? 100.0 * differ / checked : 0.0,
               (b.covered || all_covered) ? "covered" : "not covered");
        if (b.covered || all_covered) {
            if (differ == 0) g_pass++; else g_fail++;
        }
    }
    printf("    %-30s %8ld %8ld %8.2f%%\n", "total", total_checked, total_differ,
           100.0 * total_differ / total_checked);
}

/* ==========================================================================
 * Corpus
 * ========================================================================== */

struct CorpusText {
    std::string id, script, text;
};

static bool load_corpus(const std::string& dir, std::vector<CorpusText>& out) {
    std::string blob;
    if (!read_file(dir + "/corpus.json", blob)) return false;
    cJSON* root = cJSON_Parse(blob.c_str());
    cJSON* texts = root ? cJSON_GetObjectItem(root, "texts") : nullptr;
    if (!cJSON_IsArray(texts)) { cJSON_Delete(root); return false; }
    for (cJSON* t = texts->child; t; t = t->next) {
        cJSON* id = cJSON_GetObjectItem(t, "id");
        cJSON* sc = cJSON_GetObjectItem(t, "script");
        cJSON* tx = cJSON_GetObjectItem(t, "text");
        if (cJSON_IsString(id) && cJSON_IsString(sc) && cJSON_IsString(tx))
            out.push_back({id->valuestring, sc->valuestring, tx->valuestring});
    }
    cJSON_Delete(root);
    return !out.empty();
}

/* ==========================================================================
 * Layer 2 - pre-tokenisation, per corpus text
 * ========================================================================== */

static std::vector<std::string> our_words(const std::string& text, bool lowercase) {
    /* Same composition as TokenizerWrapper::encode_single(). */
    return lembed::detail::basic_tokenize(
        lowercase ? lembed::detail::bert_normalize(text) : text);
}

static void test_pretokenizer(const std::string& dir, const std::vector<CorpusText>& corpus) {
    std::string blob;
    if (!read_file(dir + "/hf_pretokenized.json", blob)) {
        fprintf(stderr, "FAIL: cannot read %s/hf_pretokenized.json\n", dir.c_str());
        g_fail++;
        return;
    }
    cJSON* root = cJSON_Parse(blob.c_str());
    printf("\n[2] Pre-tokenisation vs BertNormalizer + BertPreTokenizer, per text\n");
    for (const char* variant : {"uncased", "cased"}) {
        cJSON* v = root ? cJSON_GetObjectItem(root, variant) : nullptr;
        if (!v) {
            fprintf(stderr, "FAIL: hf_pretokenized.json has no \"%s\"\n", variant);
            g_fail++;
            continue;
        }
        const bool lower = strcmp(variant, "uncased") == 0;
        int differ = 0;
        std::map<std::string, std::pair<int, int>> by_script;
        for (const CorpusText& t : corpus) {
            cJSON* words = cJSON_GetObjectItem(v, t.id.c_str());
            std::vector<std::string> want;
            for (cJSON* w = words ? words->child : nullptr; w; w = w->next)
                if (cJSON_IsString(w)) want.push_back(w->valuestring);
            const std::vector<std::string> got = our_words(t.text, lower);
            auto& s = by_script[t.script];
            s.first++;
            if (got == want) {
                g_pass++;
                continue;
            }
            differ++;
            s.second++;
            g_fail++;
            std::string g, w;
            for (const auto& x : got) g += "[" + show(x) + "]";
            for (const auto& x : want) w += "[" + show(x) + "]";
            fprintf(stderr, "FAIL: pre-tokenise %s/%s:\n  got  %s\n  want %s\n",
                    variant, t.id.c_str(), g.c_str(), w.c_str());
        }
        printf("    %-8s %d/%zu texts differ\n", variant, differ, corpus.size());
        for (const auto& s : by_script)
            if (s.second.second)
                printf("             %-18s %d/%d\n", s.first.c_str(), s.second.second,
                       s.second.first);
    }
    cJSON_Delete(root);
}

/* ==========================================================================
 * Layer 3 - token ids, per tokenizer family
 * ========================================================================== */

/* Families whose model type, normaliser and pre-tokeniser libembedding
 * implements, and which must therefore match HF id for id. The others are
 * measured and reported so the gap has a number attached. */
static bool family_supported(const std::string& family) {
    return family.rfind("wordpiece-", 0) == 0 || family.rfind("unigram-", 0) == 0;
}

static std::string cache_root() {
    const char* env = getenv("LIBEMBEDDING_CACHE_DIR");
    if (env && env[0]) return env;
#ifdef _WIN32
    const char* home = getenv("USERPROFILE");
#else
    const char* home = getenv("HOME");
#endif
    return home && home[0] ? std::string(home) + "/.cache/libembedding" : std::string();
}

/* Minimal SHA-256, only to make sure the cached tokenizer.json is the one the
 * fixture was generated from: a different revision legitimately yields
 * different ids and must skip, not fail. */
static std::string sha256_hex(const std::string& data) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string m = data;
    const uint64_t bits = (uint64_t)data.size() * 8;
    m += (char)0x80;
    while (m.size() % 64 != 56) m += (char)0;
    for (int i = 7; i >= 0; i--) m += (char)((bits >> (i * 8)) & 0xFF);
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            const unsigned char* p = (const unsigned char*)m.data() + off + i * 4;
            w[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) +
                                ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
                                ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char out[65];
    for (int i = 0; i < 8; i++) snprintf(out + i * 8, 9, "%08x", (unsigned)h[i]);
    return std::string(out, 64);
}

static void test_ids(const std::string& dir, const std::vector<CorpusText>& corpus) {
    std::string blob;
    if (!read_file(dir + "/hf_ids.json", blob)) {
        fprintf(stderr, "FAIL: cannot read %s/hf_ids.json\n", dir.c_str());
        g_fail++;
        return;
    }
    cJSON* root = cJSON_Parse(blob.c_str());
    cJSON* families = root ? cJSON_GetObjectItem(root, "families") : nullptr;
    if (!cJSON_IsArray(families)) {
        fprintf(stderr, "FAIL: hf_ids.json is malformed\n");
        g_fail++;
        cJSON_Delete(root);
        return;
    }
    const std::string cache = cache_root();
    /* The unsupported families have 17 MB vocabularies and no decoder here, so
     * measuring them is slow and only informative: opt in. */
    const char* ra = getenv("LEMBED_TOKENIZER_REPORT_ALL");
    const bool report_all = ra && ra[0] && ra[0] != '0';
    printf("\n[3] Token ids vs HF tokenizers, per family (cache: %s)\n", cache.c_str());
    for (cJSON* fam = families->child; fam; fam = fam->next) {
        const std::string name = cJSON_GetObjectItem(fam, "family")->valuestring;
        const std::string cdir = cJSON_GetObjectItem(fam, "cache_dir")->valuestring;
        const std::string sha = cJSON_GetObjectItem(fam, "tokenizer_sha256")->valuestring;
        cJSON* ids = cJSON_GetObjectItem(fam, "ids");
        const bool supported = family_supported(name);
        if (!supported && !report_all) {
            printf("    %-26s not implemented (set LEMBED_TOKENIZER_REPORT_ALL=1 to measure the gap)\n",
                   name.c_str());
            continue;
        }

        const std::string path = cache + "/" + cdir + "/tokenizer.json";
        std::string tok_blob;
        if (cache.empty() || !read_file(path, tok_blob)) {
            printf("    %-26s SKIP (no %s/tokenizer.json in the cache)\n", name.c_str(), cdir.c_str());
            continue;
        }
        if (sha256_hex(tok_blob) != sha) {
            printf("    %-26s SKIP (cached tokenizer.json differs from the fixture's)\n", name.c_str());
            continue;
        }
        lembed::detail::TokenizerWrapper tok;
        try {
            tok.load_from_blob(tok_blob, 1 << 20);   /* no truncation, like the fixture */
        } catch (const std::exception& e) {
            printf("    %-26s load failed: %s\n", name.c_str(), e.what());
            if (supported) g_fail++;
            continue;
        }
        int differ = 0;
        std::map<std::string, std::pair<int, int>> by_script;
        for (const CorpusText& t : corpus) {
            cJSON* want_arr = cJSON_GetObjectItem(ids, t.id.c_str());
            std::vector<int> want;
            for (cJSON* x = want_arr ? want_arr->child : nullptr; x; x = x->next)
                want.push_back(x->valueint);
            const std::vector<int> got = tok.encode(t.text);
            auto& s = by_script[t.script];
            s.first++;
            if (got == want) {
                if (supported) g_pass++;
                continue;
            }
            differ++;
            s.second++;
            if (supported) {
                g_fail++;
                std::string g, w;
                for (int x : got) g += std::to_string(x) + " ";
                for (int x : want) w += std::to_string(x) + " ";
                fprintf(stderr, "FAIL: ids %s/%s:\n  got  %s\n  want %s\n", name.c_str(),
                        t.id.c_str(), g.c_str(), w.c_str());
            }
        }
        printf("    %-26s %2d/%zu texts differ (%5.1f%%)  %s\n", name.c_str(), differ,
               corpus.size(), 100.0 * differ / corpus.size(),
               supported ? "[implemented: must match]" : "[not implemented: reported only]");
        for (const auto& s : by_script)
            if (s.second.second)
                printf("        %-18s %d/%d\n", s.first.c_str(), s.second.second, s.second.first);
    }
    cJSON_Delete(root);
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : getenv("LEMBED_TOKENIZER_FIXTURES");
    if (!dir || !dir[0]) dir = LEMBED_TOKENIZER_FIXTURES_DEFAULT;

    std::vector<CorpusText> corpus;
    if (!load_corpus(dir, corpus)) {
        fprintf(stderr, "FAIL: cannot read the corpus in %s\n", dir);
        return 1;
    }
    printf("fixtures: %s (%zu texts)\n", dir, corpus.size());

    test_normalizer(dir, "hf_normalizer.json", "strip_accents + lowercase", true, true, false);
    test_normalizer(dir, "hf_normalizer_lower.json", "lowercase only", true, false, true);
    test_normalizer(dir, "hf_normalizer_strip.json", "strip_accents only", false, true, false);
    test_pretokenizer(dir, corpus);
    test_ids(dir, corpus);

    printf("\n%s: %d passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
