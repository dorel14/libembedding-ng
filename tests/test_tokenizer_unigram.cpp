/*
 * test_tokenizer_unigram.cpp - Unigram (SentencePiece) tokenizer on a tiny
 * synthetic tokenizer.json: Viterbi choice, unk fusion, byte fallback,
 * Metaspace splitting and the <s> $A </s> template. Hermetic and offline.
 * The charsmap and the real 250k vocabulary are covered by
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

/* ids: 0 <unk>, 1 <s>, 2 </s>, 3 U+2581, 4 "_a", 5 "_ab", 6 "b", 7 "a", 8 "_b",
 *      9 "<0x5A>" ('Z'), 10 "<0x5B>" ('['). "_" is U+2581. */
static std::string make_json(const char* ab_score, bool byte_fallback, const char* pre) {
    const std::string S = "\xE2\x96\x81";
    std::string j = "{\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":["
                    "{\"type\":\"Replace\",\"pattern\":{\"Regex\":\" {2,}\"},\"content\":\" \"}]},"
                    "\"pre_tokenizer\":" + std::string(pre) + ","
                    "\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":["
                    "{\"SpecialToken\":{\"id\":\"<s>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},"
                    "{\"SpecialToken\":{\"id\":\"</s>\",\"type_id\":0}}],\"pair\":[],"
                    "\"special_tokens\":{\"<s>\":{\"id\":\"<s>\",\"ids\":[1],\"tokens\":[\"<s>\"]},"
                    "\"</s>\":{\"id\":\"</s>\",\"ids\":[2],\"tokens\":[\"</s>\"]}}},"
                    "\"added_tokens\":[{\"id\":1,\"content\":\"<s>\"},{\"id\":2,\"content\":\"</s>\"},{\"id\":0,\"content\":\"<unk>\"}],"
                    "\"model\":{\"type\":\"Unigram\",\"unk_id\":0,\"byte_fallback\":";
    j += byte_fallback ? "true" : "false";
    j += ",\"vocab\":[[\"<unk>\",0.0],[\"<s>\",0.0],[\"</s>\",0.0],[\"" + S + "\",-4.0],"
         "[\"" + S + "a\",-1.0],[\"" + S + "ab\"," + ab_score + "],[\"b\",-1.0],[\"a\",-2.0],[\"" + S + "b\",-2.0],"
         "[\"<0x5A>\",-5.0],[\"<0x5B>\",-5.0]]}}";
    return j;
}

static const char* kMeta = "{\"type\":\"Metaspace\",\"replacement\":\"\xE2\x96\x81\",\"prepend_scheme\":\"always\"}";

static std::vector<int> enc(const std::string& json, const std::string& text) {
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_blob(json, 1 << 20);
    return tok.encode(text);
}

int main() {
    /* Viterbi: "_ab" (-1.5) beats "_a"+"b" (-2.0) ... */
    check_ids("viterbi prefers whole piece", enc(make_json("-1.5", false, kMeta), "ab"), {1, 5, 2});
    /* ... and loses when it scores -2.5. */
    check_ids("viterbi prefers split", enc(make_json("-2.5", false, kMeta), "ab"), {1, 4, 6, 2});

    /* Metaspace: one word per space, prefix added, runs of spaces collapsed. */
    check_ids("metaspace split", enc(make_json("-1.5", false, kMeta), "ab  ab"), {1, 5, 5, 2});
    check_ids("metaspace first word", enc(make_json("-1.5", false, kMeta), "ab b"), {1, 5, 8, 2});
    check_ids("empty text", enc(make_json("-1.5", false, kMeta), ""), {1, 2});
    check_ids("only space", enc(make_json("-1.5", false, kMeta), " "), {1, 3, 2});
    {
        const char* never = "{\"type\":\"Metaspace\",\"replacement\":\"\xE2\x96\x81\",\"prepend_scheme\":\"never\"}";
        check_ids("prepend never", enc(make_json("-1.5", false, never), "ab"), {1, 7, 6, 2});
    }

    /* Sequence[WhitespaceSplit, Metaspace] (paraphrase-multilingual): any
     * Unicode whitespace run separates words and is dropped. */
    {
        const char* wss = "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"WhitespaceSplit\"},"
                          "{\"type\":\"Metaspace\",\"replacement\":\"\xE2\x96\x81\",\"add_prefix_space\":true}]}";
        check_ids("whitespace split", enc(make_json("-1.5", false, wss), "ab\t\r\n b"), {1, 5, 8, 2});
        check_ids("whitespace split nbsp", enc(make_json("-1.5", false, wss), " ab\xC2\xA0" "b "), {1, 5, 8, 2});
    }

    /* Unknown characters: consecutive unks fuse into one. */
    check_ids("unk fused", enc(make_json("-1.5", false, kMeta), "aq?q"), {1, 4, 0, 2});
    check_ids("unk split by known piece", enc(make_json("-1.5", false, kMeta), "?b?"), {1, 3, 0, 6, 0, 2});

    /* Byte fallback: bytes known to the vocabulary replace the unk. */
    check_ids("byte fallback", enc(make_json("-1.5", true, kMeta), "aZ["), {1, 4, 9, 10, 2});
    check_ids("byte fallback missing byte -> unk", enc(make_json("-1.5", true, kMeta), "aZq"), {1, 4, 0, 2});

    /* No unk_id: an unknown character is an error, never id -1 (which would
     * be an out-of-range index for the model's embedding gather). */
    {
        std::string json = make_json("-1.5", false, kMeta);
        const std::string key = "\"unk_id\":0,";
        json.erase(json.find(key), key.size());
        check_ids("no unk_id, known text", enc(json, "ab"), {1, 5, 2});
        bool threw = false;
        try { enc(json, "a?"); } catch (const std::runtime_error&) { threw = true; }
        if (threw) g_pass++;
        else { g_fail++; fprintf(stderr, "FAIL: no unk_id, unknown char did not throw\n"); }
    }

    printf("test_tokenizer_unigram: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
