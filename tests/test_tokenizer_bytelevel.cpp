/*
 * test_tokenizer_bytelevel.cpp - byte-level BPE (CLIP style) on a tiny
 * synthetic tokenizer.json: GPT-2 byte mapping of non-ASCII text, the </w>
 * end-of-word suffix, contractions, digits split one by one, special tokens
 * cut out of the text, whitespace collapsing and truncation keeping the last
 * token. Hermetic and offline. The real 49k vocabulary is covered by
 * test_tokenizer_fidelity against HF fixtures.
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

/* Byte-level characters: 'e acute' is the bytes C3 A9, which GPT-2 maps to the
 * characters U+00C3 and U+00A9 (printable Latin-1 maps to itself).
 * ids: 0 a, 1 b, 2 c, 3 a</w>, 4 b</w>, 5 c</w>, 6 ab</w>, 7 U+00C3, 8 U+00A9</w>,
 *      9 U+00C3U+00A9</w>, 10 1</w>, 11 2</w>, 12 ', 13 s</w>, 14 's</w>,
 *      15 <|endoftext|> (unk and sep), 16 <|startoftext|> (cls). */
static std::string make_json() {
    return std::string(
        "{\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"NFC\"},"
        "{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"\\\\s+\"},\"content\":\" \"},{\"type\":\"Lowercase\"}]},"
        "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
        "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"<\\\\|startoftext\\\\|>|<\\\\|endoftext\\\\|>|'s|'t|'re|'ve|'m|'ll|'d"
        "|[\\\\p{L}]+|[\\\\p{N}]|[^\\\\s\\\\p{L}\\\\p{N}]+\"},\"behavior\":\"Removed\",\"invert\":true},"
        "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":true}]},"
        "\"post_processor\":{\"type\":\"RobertaProcessing\",\"sep\":[\"<|endoftext|>\",15],"
        "\"cls\":[\"<|startoftext|>\",16],\"trim_offsets\":false,\"add_prefix_space\":false},"
        "\"added_tokens\":[{\"id\":16,\"content\":\"<|startoftext|>\",\"special\":true},"
        "{\"id\":15,\"content\":\"<|endoftext|>\",\"special\":true}],"
        "\"model\":{\"type\":\"BPE\",\"unk_token\":\"<|endoftext|>\",\"continuing_subword_prefix\":\"\","
        "\"end_of_word_suffix\":\"</w>\",\"fuse_unk\":false,"
        "\"vocab\":{\"a\":0,\"b\":1,\"c\":2,\"a</w>\":3,\"b</w>\":4,\"c</w>\":5,\"ab</w>\":6,"
        "\"\xC3\x83\":7,\"\xC2\xA9</w>\":8,\"\xC3\x83\xC2\xA9</w>\":9,\"1</w>\":10,\"2</w>\":11,"
        "\"'\":12,\"s</w>\":13,\"'s</w>\":14,\"<|endoftext|>\":15,\"<|startoftext|>\":16},"
        "\"merges\":[\"a b</w>\",\"\xC3\x83 \xC2\xA9</w>\",\"' s</w>\"]}}");
}

static std::vector<int> enc(const std::string& text, int max_length = 1 << 20) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(make_json(), max_length);
    return tok.encode(text);
}

int main() {
    /* </w> goes on the last symbol of the word, then merges apply. */
    check_ids("merge with suffix", enc("ab"), {16, 6, 15});
    check_ids("suffix per word", enc("a b"), {16, 3, 4, 15});
    check_ids("lowercase", enc("AB"), {16, 6, 15});
    check_ids("whitespace runs collapse", enc("a \t\n b"), {16, 3, 4, 15});

    /* Non-ASCII goes through the byte mapping, never through code points. */
    check_ids("non-ascii bytes", enc("\xC3\xA9"), {16, 9, 15});
    /* NFC: e + combining acute composes to the precomposed letter. */
    check_ids("nfc composition", enc("e\xCC\x81"), {16, 9, 15});

    /* Contractions are their own pre-token; digits are split one by one. */
    check_ids("contraction", enc("a's"), {16, 3, 14, 15});
    check_ids("digits one by one", enc("12"), {16, 10, 11, 15});

    /* Special tokens are cut out of the text and emitted as their id. */
    check_ids("special token in text", enc("a<|endoftext|>b"), {16, 3, 15, 4, 15});
    check_ids("empty text", enc(""), {16, 15});

    /* Truncation keeps the closing special token. */
    check_ids("truncation keeps last token", enc("a b c", 3), {16, 3, 15});
    check_ids("no truncation", enc("a b c", 5), {16, 3, 4, 5, 15});

    printf("%s: %d passed, %d failed\n", g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
