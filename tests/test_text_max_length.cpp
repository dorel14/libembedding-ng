/*
 * test_text_max_length.cpp - default truncation length of ONNX dense models
 *
 * SentenceTransformers and FastEmbed truncate each model at the max_seq_length
 * of its sentence_bert_config.json (all-MiniLM-L6-v2: 256, not 512). The
 * registry max_tokens must match, and the length is resolved as:
 *   options.max_length > sentence_bert_config.json > registry / config.json.
 *
 * Part A needs no model. Part B and C use all-MiniLM-L6-v2 from the local cache
 * (~/.cache/libembedding/models--Qdrant-all-MiniLM-L6-v2-onnx) and are skipped
 * when it is absent. Never downloads.
 *
 * Auteur: David Orel
 * Version: 1.16.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

static int passes = 0, failures = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, lembed_last_error()); \
        failures++; \
    } else { \
        passes++; \
    } \
} while (0)

/* max_seq_length of the model's sentence_bert_config.json (checked against
 * huggingface.co on 2026-11-23); 512 models are listed too so a regression to
 * another value is caught. */
struct Expected { lembed_text_model_t model; int max_tokens; const char* name; };
static const Expected kExpected[] = {
    {LEMBED_TEXT_ALL_MINILM_L6_V2, 256, "all-MiniLM-L6-v2"},
    {LEMBED_TEXT_ALL_MINILM_L6_V2_Q, 256, "all-MiniLM-L6-v2 quantized"},
    {LEMBED_TEXT_ALL_MINILM_L12_V2, 128, "all-MiniLM-L12-v2"},
    {LEMBED_TEXT_ALL_MINILM_L12_V2_Q, 128, "all-MiniLM-L12-v2 quantized"},
    {LEMBED_TEXT_ALL_MPNET_BASE_V2, 384, "all-mpnet-base-v2"},
    {LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2, 128, "paraphrase-multilingual-MiniLM-L12-v2"},
    {LEMBED_TEXT_PARAPHRASE_ML_MINILM_L12_V2_Q, 128, "paraphrase-multilingual-MiniLM-L12-v2 fp16"},
    {LEMBED_TEXT_PARAPHRASE_ML_MPNET_BASE_V2, 128, "paraphrase-multilingual-mpnet-base-v2"},
    {LEMBED_TEXT_BGE_SMALL_EN_V15, 512, "bge-small-en-v1.5"},
    {LEMBED_TEXT_BGE_BASE_EN_V15, 512, "bge-base-en-v1.5"},
    {LEMBED_TEXT_BGE_LARGE_EN_V15, 512, "bge-large-en-v1.5"},
    {LEMBED_TEXT_BGE_M3, 8192, "bge-m3"},
    {LEMBED_TEXT_MULTILINGUAL_E5_SMALL, 512, "multilingual-e5-small"},
    {LEMBED_TEXT_MXBAI_EMBED_LARGE_V1, 512, "mxbai-embed-large-v1"},
    {LEMBED_TEXT_GTE_BASE_EN_V15, 8192, "gte-base-en-v1.5"},
    {LEMBED_TEXT_NOMIC_EMBED_TEXT_V15, 8192, "nomic-embed-text-v1.5"},
    {LEMBED_TEXT_SNOWFLAKE_ARCTIC_EMBED_M, 512, "snowflake-arctic-embed-m"},
};

static std::string home_dir() {
    const char* home = getenv("USERPROFILE");
    if (!home || !home[0]) home = getenv("HOME");
    return home ? home : "";
}

static void write_file(const fs::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary);
    f << text;
}

/* Hard link when possible (no 90 MB copy), copy otherwise. */
static bool link_or_copy(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::create_hard_link(from, to, ec);
    if (!ec) return true;
    ec.clear();
    return fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec) && !ec;
}

static std::string long_text(int words) {
    static const char* vocab[] = {"the", "model", "of", "and", "data", "in", "to", "is",
                                  "for", "with", "on", "that", "this", "as", "are", "by"};
    std::string s;
    unsigned x = 12345u;
    for (int i = 0; i < words; i++) {
        x = x * 1103515245u + 12345u;
        if (i) s += ' ';
        s += vocab[(x >> 16) % 16];
    }
    return s;
}

static bool embed_one(lembed_text_embedding_t* ctx, const std::string& text, std::vector<float>& out) {
    const char* t = text.c_str();
    lembed_embeddings_t r;
    r.data = nullptr; r.num_embeddings = 0; r.dim = 0;
    if (lembed_text_embedding_embed(ctx, &t, 1, 1, &r) != LEMBED_OK) return false;
    out.assign(r.data, r.data + r.dim);
    lembed_embeddings_free(&r);
    return true;
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) m = std::fmax(m, std::fabs((double)a[i] - b[i]));
    return m;
}

int main(void) {
    /* Part A: registry values */
    for (const Expected& e : kExpected) {
        lembed_model_info_t info;
        ASSERT(lembed_get_text_model_info(e.model, &info) == LEMBED_OK, e.name);
        if (info.max_tokens != e.max_tokens) {
            fprintf(stderr, "  %s: max_tokens %d, expected %d\n", e.name, info.max_tokens, e.max_tokens);
        }
        ASSERT(info.max_tokens == e.max_tokens, "registry max_tokens equals sentence_bert max_seq_length");
    }

    /* Part B and C need the cached MiniLM */
    const fs::path src = fs::path(home_dir()) / ".cache" / "libembedding" / "models--Qdrant-all-MiniLM-L6-v2-onnx";
    if (!fs::exists(src / "model.onnx") || !fs::exists(src / "tokenizer.json")) {
        printf("SKIP parts B and C: %s not cached\n", src.string().c_str());
        printf("\n%d passed, %d failed\n", passes, failures);
        return failures > 0 ? 1 : 0;
    }

    /* Part B: priority user > sentence_bert_config.json > registry / config.json */
    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path() / ("lembed_maxlen_" + std::to_string((long long)std::rand()));
    const fs::path model_dir = tmp / "models--Qdrant-all-MiniLM-L6-v2-onnx";
    fs::create_directories(model_dir, ec);
    bool ok = link_or_copy(src / "model.onnx", model_dir / "model.onnx") &&
              link_or_copy(src / "tokenizer.json", model_dir / "tokenizer.json");
    if (fs::exists(src / "config.json")) ok = link_or_copy(src / "config.json", model_dir / "config.json") && ok;
    ASSERT(ok, "temporary model directory prepared");
    if (ok) {
        const std::string tmp_str = tmp.generic_string();
        const std::string dir_str = model_dir.generic_string();
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;
        opts.cache_dir = tmp_str.c_str();
        opts.offline = 1;

        lembed_text_embedding_t* ctx = nullptr;
        ASSERT(lembed_text_embedding_create(&opts, &ctx) == LEMBED_OK, "create without sentence_bert_config.json");
        if (ctx) {
            ASSERT(lembed_text_embedding_max_length(ctx) == 256, "registry value 256 without the file");
            lembed_text_embedding_free(ctx);
            ctx = nullptr;
        }

        write_file(model_dir / "sentence_bert_config.json", "{\"max_seq_length\": 64, \"do_lower_case\": false}");
        ASSERT(lembed_text_embedding_create(&opts, &ctx) == LEMBED_OK, "create with sentence_bert_config.json");
        if (ctx) {
            ASSERT(lembed_text_embedding_max_length(ctx) == 64, "sentence_bert_config.json beats the registry");
            lembed_text_embedding_free(ctx);
            ctx = nullptr;
        }

        opts.max_length = 100;
        ASSERT(lembed_text_embedding_create(&opts, &ctx) == LEMBED_OK, "create with explicit max_length");
        if (ctx) {
            ASSERT(lembed_text_embedding_max_length(ctx) == 100, "explicit max_length beats the file");
            lembed_text_embedding_free(ctx);
            ctx = nullptr;
        }

        /* Local directory path: user > sentence_bert > config.json */
        lembed_text_options_t popts = lembed_text_options_default();
        ASSERT(lembed_text_embedding_create_from_path(dir_str.c_str(), &popts, &ctx) == LEMBED_OK,
               "create_from_path with sentence_bert_config.json");
        if (ctx) {
            ASSERT(lembed_text_embedding_max_length(ctx) == 64, "path: sentence_bert_config.json beats config.json");
            lembed_text_embedding_free(ctx);
            ctx = nullptr;
        }
        popts.max_length = 128;
        ASSERT(lembed_text_embedding_create_from_path(dir_str.c_str(), &popts, &ctx) == LEMBED_OK,
               "create_from_path with explicit max_length");
        if (ctx) {
            ASSERT(lembed_text_embedding_max_length(ctx) == 128, "path: explicit max_length beats everything");
            lembed_text_embedding_free(ctx);
            ctx = nullptr;
        }

        /* Unusable file values are ignored */
        write_file(model_dir / "sentence_bert_config.json", "{\"max_seq_length\": \"x\"}");
        popts.max_length = 0;
        ASSERT(lembed_text_embedding_create_from_path(dir_str.c_str(), &popts, &ctx) == LEMBED_OK,
               "create_from_path with a malformed value");
        if (ctx) {
            int ml = lembed_text_embedding_max_length(ctx);
            ASSERT(ml == 512 || ml == 0 || ml == LEMBED_DEFAULT_MAX_LENGTH,
                   "path: malformed sentence_bert_config.json falls back to config.json / default");
            lembed_text_embedding_free(ctx);
            ctx = nullptr;
        }
    }
    fs::remove_all(tmp, ec);

    /* Part C: with the registry default (256), text beyond 256 tokens is cut:
     * 600 words and their first 300 words give the same vector, while an
     * explicit 512 sees the difference. */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.model = LEMBED_TEXT_ALL_MINILM_L6_V2;
        opts.offline = 1;
        lembed_text_embedding_t* ctx = nullptr;
        if (lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
            printf("SKIP part C: %s\n", lembed_last_error());
        } else {
            const std::string w600 = long_text(600);
            const std::string w300 = long_text(300);
            std::vector<float> a, b;
            ASSERT(embed_one(ctx, w600, a) && embed_one(ctx, w300, b), "embed long texts");
            ASSERT(lembed_text_embedding_max_length(ctx) == 256, "default max_length is 256");
            ASSERT(max_abs_diff(a, b) < 1e-6, "600 words == first 300 words once cut at 256 tokens");
            lembed_text_embedding_free(ctx);

            opts.max_length = 512;
            ctx = nullptr;
            std::vector<float> c, d;
            if (lembed_text_embedding_create(&opts, &ctx) == LEMBED_OK) {
                ASSERT(embed_one(ctx, w600, c) && embed_one(ctx, w300, d), "embed long texts at 512");
                ASSERT(max_abs_diff(c, d) > 1e-4, "at 512 the extra tokens change the vector");
                ASSERT(max_abs_diff(a, c) > 1e-4, "256 and 512 truncations differ on a long text");
                lembed_text_embedding_free(ctx);
            }
        }
    }

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
