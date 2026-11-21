/*
 * test_tokenizer.cpp - HuggingFace parity for the BERT normaliser.
 *
 * Why this test exists: the reference SPLADE model is a HuggingFace
 * BertTokenizer. Without accent stripping, this tokeniser disagreed with HF on
 * accented text while llama.cpp's own WPM tokeniser stripped accents -- so the
 * GGUF and ONNX backends were not comparable, and the "cosine GGUF vs ONNX"
 * acceptance criterion would have measured the tokeniser gap instead of the
 * sparse implementation.
 *
 * The expectations below are HuggingFace's, not this implementation's: they
 * follow from `BertTokenizer.tokenize()`, which per whitespace token does
 * `token.lower()` then `_run_strip_accents()` (NFD, drop the Mn category).
 * Hermetic and offline: the vocabulary is built from a synthetic
 * tokenizer.json in memory.
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/detail/tokenizer_impl.hpp>

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); g_fail++; } \
} while (0)

#define CHECK_STR(actual, expected, msg) do { \
    const std::string a_ = (actual); \
    const std::string e_ = (expected); \
    if (a_ == e_) { g_pass++; } \
    else { fprintf(stderr, "FAIL: %s (line %d): got \"%s\", want \"%s\"\n", \
                  msg, __LINE__, a_.c_str(), e_.c_str()); g_fail++; } \
} while (0)

/* ==========================================================================
 * Normaliser
 * ========================================================================== */

static void test_plain_ascii(void) {
    CHECK_STR(lembed::detail::bert_normalize("The quick brown Fox"),
              "the quick brown fox", "ASCII is lowercased");
    CHECK_STR(lembed::detail::bert_normalize("MACHINE learning"),
              "machine learning", "uppercase ASCII is lowercased");
}

static void test_accents_are_stripped(void) {
    /* The cases that motivated the change. Every one of these diverged from HF
     * before, and each produced [UNK] or a wrong token. */
    CHECK_STR(lembed::detail::bert_normalize("Caf\xc3\xa9"), "cafe",
              "e-acute folds to e");
    CHECK_STR(lembed::detail::bert_normalize("CAF\xc3\x89"), "cafe",
              "uppercase accents fold too");
    CHECK_STR(lembed::detail::bert_normalize("\xc3\xa9" "l\xc3\xa9" "phant"),
              "elephant", "leading and inner accents fold");
    CHECK_STR(lembed::detail::bert_normalize("na\xc3\xaf" "ve"), "naive",
              "diaeresis folds");
    CHECK_STR(lembed::detail::bert_normalize("\xc3\x96" "l"), "ol",
              "O-diaeresis folds");
}

static void test_sharp_s_and_long_s_differ(void) {
    /* Neither letter has a canonical decomposition (long s only has a
     * compatibility one, which BertNormalizer's NFD does not apply), so HF
     * keeps both. An earlier draft folded long s to "s" and sharp s to "ss";
     * the codepoint-by-codepoint comparison with HF tokenizers (T26) shows HF
     * does neither. */
    CHECK_STR(lembed::detail::bert_normalize("Stra\xc3\x9f" "e"), "stra\xc3\x9f" "e",
              "sharp-s is not decomposable, only lowercased around it");
    CHECK_STR(lembed::detail::bert_normalize("\xc5\xbf"), "\xc5\xbf",
              "long s has no canonical decomposition, so HF keeps it");
}

static void test_letters_without_decomposition_are_only_lowercased(void) {
    /* HF has no canonical decomposition for these, so it does not fold them to
     * "ae"/"oe"/"d"/"th" -- but it still lowercases them (verified against HF
     * tokenizers by test_tokenizer_fidelity). An earlier version of this test
     * expected them to stay uppercase, which was wrong. */
    CHECK_STR(lembed::detail::bert_normalize("\xc3\x86" "ther"), "\xc3\xa6" "ther",
              "AE is lowercased to ae-ligature, not folded");
    CHECK_STR(lembed::detail::bert_normalize("\xc5\x92" "uvre"), "\xc5\x93" "uvre",
              "OE is lowercased to oe-ligature, not folded");
    CHECK_STR(lembed::detail::bert_normalize("\xc3\x9e" "orn"), "\xc3\xbe" "orn",
              "Thorn is lowercased, not folded");
    CHECK_STR(lembed::detail::bert_normalize("\xc3\x90" "an"), "\xc3\xb0" "an",
              "Eth is lowercased, not folded");
}

static void test_punctuation_and_digits_untouched(void) {
    CHECK_STR(lembed::detail::bert_normalize("a,b.c!"), "a,b.c!",
              "punctuation survives normalisation");
    CHECK_STR(lembed::detail::bert_normalize("2024"), "2024",
              "digits survive normalisation");
    CHECK_STR(lembed::detail::bert_normalize(""), "", "empty input is stable");
}

static void test_non_latin_scripts_fold_like_hf(void) {
    /* Regression for T26: Greek, Cyrillic and the rest used to pass through
     * with their case; HF lowercases and strips them. Expected values come from
     * HF tokenizers' BertNormalizer. */
    CHECK_STR(lembed::detail::bert_normalize("\xce\x91\xce\x92"), "\xce\xb1\xce\xb2",
              "Greek capitals are lowercased");
    CHECK_STR(lembed::detail::bert_normalize("\xce\x86"), "\xce\xb1",
              "Greek tonos is stripped (U+0386 -> alpha)");
    CHECK_STR(lembed::detail::bert_normalize("\xe1\xbc\x88"), "\xce\xb1",
              "polytonic Greek folds to the base letter (U+1F08)");
    CHECK_STR(lembed::detail::bert_normalize("\xd0\x9f\xd1\x80\xd0\x98\xd0\x92\xd0\x95\xd0\xa2"),
              "\xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82", "Cyrillic is lowercased");
    CHECK_STR(lembed::detail::bert_normalize("\xd0\x99"), "\xd0\xb8",
              "short I loses its breve like HF (U+0419 -> U+0438)");
    CHECK_STR(lembed::detail::bert_normalize("\xe1\xbb\x86"), "e",
              "Vietnamese stacked diacritics fold (U+1EC6)");
    CHECK_STR(lembed::detail::bert_normalize("\xe2\x89\xa0"), "=",
              "not-equal decomposes to = + combining slash, which is stripped");
    CHECK_STR(lembed::detail::bert_normalize("\xd7\xa9\xd6\xb8\xd7\x9c"), "\xd7\xa9\xd7\x9c",
              "Hebrew points are stripped");
    CHECK_STR(lembed::detail::bert_normalize("\xea\xb0\x80"), "\xe1\x84\x80\xe1\x85\xa1",
              "a Hangul syllable decomposes to jamo");
    CHECK_STR(lembed::detail::bert_normalize("\xf0\x9f\x98\x80"), "\xf0\x9f\x98\x80",
              "a 4-byte emoji round-trips");
    CHECK_STR(lembed::detail::bert_normalize("\xe4\xb8\xad"), "\xe4\xb8\xad",
              "a CJK ideograph is unchanged by the normaliser");
}

/* ==========================================================================
 * End to end, through a synthetic BertTokenizer
 * ========================================================================== */

static std::string make_tokenizer_json() {
    return std::string("{") +
        "\"normalizer\":{\"type\":\"BertNormalizer\",\"lowercase\":true}," +
        "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"[UNK]\","
            "\"continuing_subword_prefix\":\"##\",\"vocab\":{"
            "\"[PAD]\":0,\"[UNK]\":1,\"[CLS]\":2,\"[SEP]\":3,"
            "\"cafe\":4,\"the\":5,\"quick\":6,\"brown\":7,\"fox\":8,"
            "\"machine\":9,\"learn\":10,\"##ing\":11,\"elephant\":12,"
            "\"naive\":13,\".\":14,\"strasse\":15}}," +
        "\"post_processor\":{\"type\":\"TemplateProcessing\"}," +
        "\"added_tokens\":["
            "{\"id\":0,\"content\":\"[PAD]\"},"
            "{\"id\":1,\"content\":\"[UNK]\"},"
            "{\"id\":2,\"content\":\"[CLS]\"},"
            "{\"id\":3,\"content\":\"[SEP]\"}]}";
}

static std::vector<int64_t> ids(std::initializer_list<int> v) {
    return std::vector<int64_t>(v.begin(), v.end());
}

/* encode_single() is private, so go through the public batch API. For a single
 * text there is no padding to strip: seq_length is that text's own length. */
static std::vector<int64_t> encode_one(lembed::detail::TokenizerWrapper& tok,
                                       const std::string& text) {
    const lembed::detail::EncodingBatch b = tok.encode_batch({text});
    return b.input_ids.empty() ? std::vector<int64_t>() : b.input_ids[0];
}

static void test_encode_matches_huggingface(void) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_tokenizer_json(), 512);

    /* [CLS] ... [SEP], ids straight out of the synthetic vocab. */
    CHECK(encode_one(tok, "the quick brown fox") == ids({2, 5, 6, 7, 8, 3}),
          "plain ASCII sentence");
    CHECK(encode_one(tok, "Machine learning") == ids({2, 9, 10, 11, 3}),
          "WordPiece split on ##ing");
    CHECK(encode_one(tok, "Caf\xc3\xa9") == ids({2, 4, 3}),
          "accented word now resolves to the 'cafe' token instead of [UNK]");
    CHECK(encode_one(tok, "CAF\xc3\x89") == ids({2, 4, 3}),
          "uppercase accented word behaves identically");
    CHECK(encode_one(tok, "na\xc3\xaf" "ve") == ids({2, 13, 3}),
          "diaeresis folds before the vocabulary lookup");
    CHECK(encode_one(tok, "\xc3\xa9" "l\xc3\xa9" "phant") == ids({2, 12, 3}),
          "leading accent no longer blocks a full-word match");
    CHECK(encode_one(tok, "fox.") == ids({2, 8, 14, 3}),
          "punctuation splits into its own token");
    /* "Straße" normalises to "straße", which stays ONE word (the pre-tokeniser
     * only splits on whitespace, punctuation and CJK). Its sharp-s is not in the
     * vocabulary, so HF emits a single [UNK] for the whole word. */
    CHECK(encode_one(tok, "Stra\xc3\x9f" "e") == ids({2, 1, 3}),
          "sharp-s is absent from a BERT vocab: the whole word is one [UNK]");
}

static void test_unfoldable_letters_become_unk(void) {
    /* The honest consequence of HF parity: a letter with no decomposition is
     * not in a BERT vocabulary either, so the word is [UNK]. Before T26 the
     * pre-tokeniser cut the word at the ae-ligature and "the" still matched. */
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_tokenizer_json(), 512);
    CHECK(encode_one(tok, "\xc3\x86" "ther") == ids({2, 1, 3}),
          "AE makes the whole word one [UNK]");
}

static void test_failed_word_is_one_unk(void) {
    /* Regression (T26): when a later piece of a word is missing from the
     * vocabulary, HF discards the pieces already matched and emits ONE [UNK];
     * this tokenizer used to keep the matched prefix pieces. "learnxyz" matches
     * "learn" and then fails on "##xyz". */
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_tokenizer_json(), 512);
    CHECK(encode_one(tok, "learnxyz") == ids({2, 1, 3}),
          "a partially matched word is a single [UNK]");
    CHECK(encode_one(tok, "learning") == ids({2, 10, 11, 3}),
          "a fully matched word still splits into pieces");
}

static void test_long_word_is_unk(void) {
    /* Regression (T26): HF's WordPiece maps a word of more than
     * max_input_chars_per_word (100) characters to [UNK] even when every piece
     * exists. */
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_tokenizer_json(), 512);
    std::string longw = "learn";
    while (longw.size() < 101) longw += "ing";
    CHECK(encode_one(tok, longw) == ids({2, 1, 3}),
          "a word over 100 characters is one [UNK]");
}

static void test_cjk_and_unicode_punct_split(void) {
    /* Regression (T26): every CJK ideograph is its own word and Unicode
     * punctuation splits words; format characters vanish without splitting. */
    const auto w = lembed::detail::basic_tokenize("a\xe4\xb8\xad\xe6\x96\x87" "b");
    CHECK(w.size() == 4 && w[0] == "a" && w[1] == "\xe4\xb8\xad" &&
          w[2] == "\xe6\x96\x87" && w[3] == "b",
          "ideographs are single-character words");
    const auto p = lembed::detail::basic_tokenize("a\xe2\x80\x94" "b");   /* em dash */
    CHECK(p.size() == 3 && p[1] == "\xe2\x80\x94", "Unicode punctuation splits a word");
    const auto z = lembed::detail::basic_tokenize("a\xe2\x80\x8b" "b");   /* zero-width space */
    CHECK(z.size() == 1 && z[0] == "ab", "a zero-width space is removed, not a separator");
}

static void test_lowercase_disabled_keeps_accents(void) {
    /* With do_lower_case=false HF skips the accent strip, so the accented form
     * survives normalisation and is looked up as is: with lowercase off, an
     * accented word does NOT resolve to its ASCII counterpart. */
    std::string json = make_tokenizer_json();
    const size_t lc = json.find("\"lowercase\":true");
    json.replace(lc, strlen("\"lowercase\":true"), "\"lowercase\":false");
    const size_t el = json.find("\"elephant\":12");
    json.replace(el, strlen("\"elephant\":12"),
                 "\"caf\xc3\xa9\":16,\"el\xc3\xa9phant\":17");

    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(json, 512);
    CHECK(encode_one(tok, "caf\xc3\xa9") != ids({2, 4, 3}),
          "with lowercase off, an accented word does not collapse to 'cafe'");
}

static void test_special_tokens_are_added_once(void) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_tokenizer_json(), 512);
    const std::vector<int64_t> out = encode_one(tok, "fox");
    CHECK(out.size() == 3, "[CLS] + token + [SEP]");
    CHECK(out.front() == 2 && out.back() == 3, "[CLS] opens, [SEP] closes");
}

/* ==========================================================================
 * Pre-tokenisation of non-ASCII letters
 * ========================================================================== */

/* basic_tokenize() used to treat every byte >= 0x80 as a standalone token, so
 * "café" arrived as ["caf", "é"] and missed the vocabulary twice. HuggingFace's
 * BertPreTokenizer splits on whitespace, punctuation and CJK ideographs only,
 * so "café" is one word (T26). */
static void test_multibyte_letters_stay_in_the_word(void) {
    const std::vector<std::string> parts = lembed::detail::basic_tokenize("caf\xc3\xa9");
    CHECK(parts.size() == 1 && parts[0] == "caf\xc3\xa9",
          "an accented letter does not cut the word");
}

/* ==========================================================================
 * Independent normaliser switches (lowercase / strip_accents / Sequence)
 * ========================================================================== */

/* The synthetic vocabulary plus "café" (16) and "Cafe" (17), with the given
 * normalizer JSON object in place of the default BertNormalizer. */
static std::string make_json_with_normalizer(const std::string& normalizer) {
    std::string json = make_tokenizer_json();
    const std::string dflt = "{\"type\":\"BertNormalizer\",\"lowercase\":true}";
    json.replace(json.find(dflt), dflt.size(), normalizer);
    const std::string tail = "\"strasse\":15}";
    json.replace(json.find(tail), tail.size(), "\"strasse\":15,\"caf\xc3\xa9\":16,\"Cafe\":17}");
    return json;
}

static std::vector<int64_t> encode_with(const std::string& normalizer, const std::string& text) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_json_with_normalizer(normalizer), 512);
    return encode_one(tok, text);
}

static void test_strip_accents_is_independent_of_lowercase(void) {
    /* Regression (T26 review): accent stripping used to follow lowercase, and
     * strip_accents / clean_text / handle_chinese_chars were never read. */
    CHECK(encode_with("{\"type\":\"BertNormalizer\",\"lowercase\":true,\"strip_accents\":false}",
                      "Caf\xc3\x89") == ids({2, 16, 3}),
          "lowercase:true + strip_accents:false lowercases but keeps the accent");
    CHECK(encode_with("{\"type\":\"BertNormalizer\",\"lowercase\":false,\"strip_accents\":true}",
                      "Caf\xc3\xa9") == ids({2, 17, 3}),
          "lowercase:false + strip_accents:true strips the accent but keeps the case");
    CHECK(encode_with("{\"type\":\"BertNormalizer\",\"lowercase\":true,\"strip_accents\":null}",
                      "Caf\xc3\x89") == ids({2, 4, 3}),
          "strip_accents:null follows lowercase");
    CHECK(encode_with("{\"type\":\"BertNormalizer\",\"lowercase\":false}", "Caf\xc3\x89") ==
              ids({2, 1, 3}),
          "lowercase:false alone changes nothing");
}

static void test_handle_chinese_and_clean_text_switches(void) {
    const std::string zh = "\xe4\xb8\xad\xe6\x96\x87";
    /* With handle_chinese_chars off an ideograph run is one word. */
    lembed::detail::TokenizerWrapper on, off;
    on.load_from_blob(make_json_with_normalizer("{\"type\":\"BertNormalizer\"}"), 512);
    off.load_from_blob(make_json_with_normalizer(
        "{\"type\":\"BertNormalizer\",\"handle_chinese_chars\":false}"), 512);
    CHECK(encode_one(on, zh) == ids({2, 1, 1, 3}), "each ideograph is its own [UNK] by default");
    CHECK(encode_one(off, zh) == ids({2, 1, 3}), "handle_chinese_chars:false keeps the run whole");
    const std::vector<std::string> kept =
        lembed::detail::basic_tokenize("a\xe2\x80\x8b" "b", false, true);
    CHECK(kept.size() == 1 && kept[0] == "a\xe2\x80\x8b" "b",
          "clean_text:false keeps format characters");
}

static void test_sequence_normalizer(void) {
    CHECK(encode_with("{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"NFD\"},"
                      "{\"type\":\"Lowercase\"},{\"type\":\"StripAccents\"}]}",
                      "Caf\xc3\x89") == ids({2, 4, 3}),
          "Sequence NFD + Lowercase + StripAccents lowercases and strips");
    CHECK(encode_with("{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"Lowercase\"}]}",
                      "Caf\xc3\x89") == ids({2, 16, 3}),
          "Sequence with Lowercase only keeps the accent");
    CHECK(encode_with("{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"NFD\"}]}",
                      "Cafe") == ids({2, 17, 3}),
          "Sequence without Lowercase keeps the case");
}

/* ==========================================================================
 * Malformed UTF-8
 * ========================================================================== */

static void test_malformed_utf8_is_kept_verbatim(void) {
    /* Regression (T26 review): a lone byte 0x80-0xFF used to be read as a
     * Latin-1 codepoint (0xC9 -> E-acute -> "e"), and a lead byte followed by a
     * non-continuation byte swallowed it. */
    CHECK_STR(lembed::detail::bert_normalize("\xc3" "A"), "\xc3" "a",
              "a lead byte before an ASCII letter does not swallow the letter");
    CHECK_STR(lembed::detail::bert_normalize("\xc9"), "\xc9",
              "a lone 0xC9 is not E-acute");
    CHECK_STR(lembed::detail::bert_normalize("\xc0\x80"), "\xc0\x80", "overlong NUL is rejected");
    CHECK_STR(lembed::detail::bert_normalize("\xed\xa0\x80"), "\xed\xa0\x80",
              "an encoded surrogate is rejected");
    CHECK_STR(lembed::detail::bert_normalize("\xf4\x90\x80\x80"), "\xf4\x90\x80\x80",
              "a codepoint above U+10FFFF is rejected");
    const std::vector<std::string> w = lembed::detail::basic_tokenize("a\xc9" "b c");
    CHECK(w.size() == 2 && w[0] == "a\xc9" "b" && w[1] == "c",
          "a stray byte stays inside its word");
}

/* ==========================================================================
 * Byte-level BPE: legacy behaviour, pinned
 *
 * BPE keeps the pre-T26 normalisation and pre-tokenisation until its own
 * rewrite. These ids are today's behaviour, not HF's: when BPE is rewritten
 * they are meant to change, and this test is where that becomes explicit.
 * ========================================================================== */

static std::string make_bpe_json() {
    return std::string("{") +
        "\"normalizer\":{\"type\":\"BertNormalizer\",\"lowercase\":true}," +
        "\"model\":{\"type\":\"BPE\",\"unk_token\":\"[UNK]\",\"vocab\":{"
            "\"[PAD]\":0,\"[UNK]\":1,\"[CLS]\":2,\"[SEP]\":3,"
            "\"a\":4,\"b\":5,\"c\":6,\"ab\":7,\"e\":8,\"\xc3\xa6\":9,\"\xc3\xa6" "c\":10},"
            "\"merges\":[\"a b\",\"\xc3\xa6 c\"]}," +
        "\"post_processor\":{\"type\":\"TemplateProcessing\"}," +
        "\"added_tokens\":["
            "{\"id\":0,\"content\":\"[PAD]\"},"
            "{\"id\":1,\"content\":\"[UNK]\"},"
            "{\"id\":2,\"content\":\"[CLS]\"},"
            "{\"id\":3,\"content\":\"[SEP]\"}]}";
}

static void test_bpe_legacy_behaviour_is_pinned(void) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_bpe_json(), 512);
    CHECK(encode_one(tok, "Ab c") == ids({2, 7, 6, 3}), "BPE merges within a word");
    CHECK(encode_one(tok, "\xc3\x89") == ids({2, 8, 3}),
          "legacy: BPE folds E-acute to e (HF byte-level BPE would not)");
    CHECK(encode_one(tok, "\xc3\xa6" "c") == ids({2, 9, 6, 3}),
          "legacy: a non-ASCII character is its own word, so ae-ligature + c never merge");
    CHECK(encode_one(tok, "\xc3") == ids({2, 1, 3}),
          "a truncated multi-byte sequence is handled without reading past the end");
}

/* ==========================================================================
 * Sequence pairs (reranker input)
 * ========================================================================== */

static std::string with_post_processor(const std::string& pp) {
    std::string json = make_tokenizer_json();
    const std::string old = "\"post_processor\":{\"type\":\"TemplateProcessing\"}";
    json.replace(json.find(old), old.size(), "\"post_processor\":" + pp);
    return json;
}

static const char* kBertTemplate =
    "{\"type\":\"TemplateProcessing\","
    "\"single\":[{\"SpecialToken\":{\"id\":\"[CLS]\",\"type_id\":0}},"
        "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},"
        "{\"SpecialToken\":{\"id\":\"[SEP]\",\"type_id\":0}}],"
    "\"pair\":[{\"SpecialToken\":{\"id\":\"[CLS]\",\"type_id\":0}},"
        "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},"
        "{\"SpecialToken\":{\"id\":\"[SEP]\",\"type_id\":0}},"
        "{\"Sequence\":{\"id\":\"B\",\"type_id\":1}},"
        "{\"SpecialToken\":{\"id\":\"[SEP]\",\"type_id\":1}}],"
    "\"special_tokens\":{\"[CLS]\":{\"id\":\"[CLS]\",\"ids\":[2],\"tokens\":[\"[CLS]\"]},"
        "\"[SEP]\":{\"id\":\"[SEP]\",\"ids\":[3],\"tokens\":[\"[SEP]\"]}}}";

static void check_pair(const lembed::detail::TokenizerWrapper& tok,
                       const char* a, const char* b,
                       const std::vector<int>& want_ids,
                       const std::vector<int>& want_types, const char* msg) {
    std::vector<int> i, t;
    tok.encode_pair(a, b, i, t);
    CHECK(i == want_ids, msg);
    CHECK(t == want_types, msg);
}

static void test_pair_bert_template(void) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(with_post_processor(kBertTemplate), 512);
    check_pair(tok, "the fox", "quick brown",
               {2, 5, 8, 3, 6, 7, 3}, {0, 0, 0, 0, 1, 1, 1},
               "[CLS] q [SEP] d [SEP], types 0..0 1..1");
    /* Single sequence is unchanged by the pair support. */
    CHECK(encode_one(tok, "the fox") == ids({2, 5, 8, 3}), "single encoding unchanged");
}

static void test_pair_fallback_without_template(void) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_tokenizer_json(), 512); /* post_processor has no pair */
    check_pair(tok, "the fox", "brown",
               {2, 5, 8, 3, 7, 3}, {0, 0, 0, 0, 1, 1}, "BERT fallback");
}

static void test_pair_bert_processing_and_roberta(void) {
    lembed::detail::TokenizerWrapper bert;
    bert.load_from_blob(with_post_processor(
        "{\"type\":\"BertProcessing\",\"sep\":[\"[SEP]\",3],\"cls\":[\"[CLS]\",2]}"), 512);
    check_pair(bert, "fox", "the", {2, 8, 3, 5, 3}, {0, 0, 0, 1, 1}, "BertProcessing");

    /* RoBERTa / XLM-R: <s> A </s></s> B </s>, type ids all 0. Reuse ids 2/3. */
    lembed::detail::TokenizerWrapper rob;
    rob.load_from_blob(with_post_processor(
        "{\"type\":\"Sequence\",\"processors\":[{\"type\":\"ByteLevel\"},"
        "{\"type\":\"RobertaProcessing\",\"sep\":[\"</s>\",3],\"cls\":[\"<s>\",2]}]}"), 512);
    check_pair(rob, "fox", "the", {2, 8, 3, 3, 5, 3}, {0, 0, 0, 0, 0, 0}, "RobertaProcessing");
}

static void test_pair_longest_first_truncation(void) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(with_post_processor(kBertTemplate), 8); /* 3 specials: 5 body */
    /* a=2 tokens, b=6 tokens: the longer (b) is trimmed to 3. */
    check_pair(tok, "fox the", "quick brown fox the quick brown",
               {2, 8, 5, 3, 6, 7, 8, 3}, {0, 0, 0, 0, 1, 1, 1, 1},
               "longest_first trims the longer segment");
    /* a=4, b=4: tie trims b first, ends 3/... total 5 -> a=3,b=2 per HF loop */
    check_pair(tok, "the quick brown fox", "the quick brown fox",
               {2, 5, 6, 7, 3, 5, 6, 3}, {0, 0, 0, 0, 0, 1, 1, 1},
               "equal lengths: result fits max_length with both specials kept");
}

int main(void) {
    test_pair_bert_template();
    test_pair_fallback_without_template();
    test_pair_bert_processing_and_roberta();
    test_pair_longest_first_truncation();
    test_plain_ascii();
    test_accents_are_stripped();
    test_sharp_s_and_long_s_differ();
    test_letters_without_decomposition_are_only_lowercased();
    test_punctuation_and_digits_untouched();
    test_non_latin_scripts_fold_like_hf();

    test_encode_matches_huggingface();
    test_unfoldable_letters_become_unk();
    test_failed_word_is_one_unk();
    test_long_word_is_unk();
    test_cjk_and_unicode_punct_split();
    test_lowercase_disabled_keeps_accents();
    test_special_tokens_are_added_once();
    test_multibyte_letters_stay_in_the_word();
    test_strip_accents_is_independent_of_lowercase();
    test_handle_chinese_and_clean_text_switches();
    test_sequence_normalizer();
    test_malformed_utf8_is_kept_verbatim();
    test_bpe_legacy_behaviour_is_pinned();

    printf("%s: %d passed, %d failed\n",
           g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}