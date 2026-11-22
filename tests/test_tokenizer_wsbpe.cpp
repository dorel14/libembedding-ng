/*
 * test_tokenizer_wsbpe.cpp - non byte-level BPE with the `Whitespace`
 * pre-tokenizer (jina-reranker-v1-turbo-en style) on a tiny synthetic
 * tokenizer.json. HF splits on `\w+|[^\w\s]+` and lowercases with Unicode
 * rules; the legacy path made every non-ASCII character a word of its own, so
 * merges never applied across Cyrillic / CJK characters. Hermetic and offline.
 *
 * Auteur: David Orel
 * Version: 1.16.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/detail/tokenizer_impl.hpp>

#include <stdio.h>

#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

static void check_ids(const char* name, const std::vector<int>& got, const std::vector<int>& want) {
    if (got == want) { g_pass++; return; }
    g_fail++;
    fprintf(stderr, "FAIL: %s\n  got ", name);
    for (int x : got) fprintf(stderr, "%d ", x);
    fprintf(stderr, "\n  want ");
    for (int x : want) fprintf(stderr, "%d ", x);
    fprintf(stderr, "\n");
}

/* ids: 0 <s>, 1 <pad>, 2 </s>, 3 <unk>, 4 m (U+043C), 5 o (U+043E), 6 s (U+0441),
 *      7 mo, 8 mos, 9 U+673A, 10 U+5668, 11 U+673AU+5668, 12 !, 13 !!, 14 a, 15 b, 16 ab,
 *      17 U+00E9, 18 _ (word character). */
static std::string make_json() {
    return std::string(
        "{\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"NFC\"},{\"type\":\"Lowercase\"}]},"
        "\"pre_tokenizer\":{\"type\":\"Whitespace\"},"
        "\"post_processor\":{\"type\":\"RobertaProcessing\",\"sep\":[\"</s>\",2],\"cls\":[\"<s>\",0],"
        "\"trim_offsets\":true,\"add_prefix_space\":false},"
        "\"added_tokens\":[{\"id\":0,\"content\":\"<s>\",\"special\":true},"
        "{\"id\":1,\"content\":\"<pad>\",\"special\":true},"
        "{\"id\":2,\"content\":\"</s>\",\"special\":true},"
        "{\"id\":3,\"content\":\"<unk>\",\"special\":true}],"
        "\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"continuing_subword_prefix\":null,"
        "\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,"
        "\"vocab\":{\"<s>\":0,\"<pad>\":1,\"</s>\":2,\"<unk>\":3,"
        "\"\xD0\xBC\":4,\"\xD0\xBE\":5,\"\xD1\x81\":6,\"\xD0\xBC\xD0\xBE\":7,\"\xD0\xBC\xD0\xBE\xD1\x81\":8,"
        "\"\xE6\x9C\xBA\":9,\"\xE5\x99\xA8\":10,\"\xE6\x9C\xBA\xE5\x99\xA8\":11,"
        "\"!\":12,\"!!\":13,\"a\":14,\"b\":15,\"ab\":16,\"\xC3\xA9\":17,\"_\":18},"
        "\"merges\":[\"\xD0\xBC \xD0\xBE\",\"\xD0\xBC\xD0\xBE \xD1\x81\",\"\xE6\x9C\xBA \xE5\x99\xA8\","
        "\"! !\",\"a b\"]}}");
}

static std::vector<int> enc(const std::string& text) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_json(), 1 << 20);
    return tok.encode(text);
}

int main() {
    /* Latin baseline. */
    check_ids("latin merge", enc("ab"), {0, 16, 2});
    check_ids("two words", enc("ab a"), {0, 16, 14, 2});

    /* Cyrillic: lowercased with Unicode rules, one word, merges apply. */
    check_ids("cyrillic upper-case merges", enc("\xD0\x9C\xD0\x9E\xD0\xA1"), {0, 8, 2});
    /* Unknown Cyrillic letter: one unk, not dropped, the others still merge. */
    check_ids("cyrillic unknown letter", enc("\xD0\xBC\xD0\xBE\xD0\xB6"), {0, 7, 3, 2});

    /* CJK: letters of one word, the merge applies across characters. */
    check_ids("cjk merge", enc("\xE6\x9C\xBA\xE5\x99\xA8"), {0, 11, 2});

    /* [^\w\s]+ groups a run of punctuation into one word. */
    check_ids("punctuation run", enc("!!"), {0, 13, 2});
    check_ids("punctuation after word", enc("ab!!"), {0, 16, 13, 2});
    check_ids("underscore is a word char", enc("a_b"), {0, 14, 18, 15, 2});

    /* NFC: e + combining acute composes before the vocabulary lookup. */
    check_ids("nfc composition", enc("e\xCC\x81"), {0, 17, 2});

    check_ids("empty text", enc(""), {0, 2});

    printf("%s: %d passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
