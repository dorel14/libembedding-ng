/*
 * test_dynamic_quant_sequential.cpp - dynamic int8 text models run one text
 * per ONNX call (T46).
 *
 * DynamicQuantizeLinear computes its int8 scale over the whole batch tensor, so
 * batching such a model makes a text's vector depend on its neighbours. The
 * library detects the operator in the graph and keeps one text per run, whatever
 * batch_strategy the caller asked for.
 *
 * Part A needs no model: the byte detector, including a marker split across two
 * read chunks. Part B uses the dynamic int8 variant of all-MiniLM-L6-v2 from the
 * local cache and is skipped when it is absent. Never downloads.
 *
 * Auteur: David Orel
 * Version: 1.16.7
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _WIN32
#define LIBEMBEDDING_IMPLEMENTATION
#endif
#include <libembedding/libembedding.h>
#include <libembedding/detail/onnx_quant_detect.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
using lembed::detail::onnx_bytes_use_dynamic_quantization;
using lembed::detail::onnx_file_uses_dynamic_quantization;

static int passes = 0, failures = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, lembed_last_error()); \
        failures++; \
    } else { \
        passes++; \
    } \
} while (0)

static void write_file(const fs::path& p, const std::string& data) {
    std::ofstream out(p, std::ios::binary);
    out.write(data.data(), (std::streamsize)data.size());
}

static void part_a_detector() {
    const std::string marker = "DynamicQuantizeLinear";
    ASSERT(!onnx_bytes_use_dynamic_quantization(nullptr, 0), "null buffer");
    ASSERT(!onnx_bytes_use_dynamic_quantization("MatMul", 6), "plain op");
    ASSERT(!onnx_bytes_use_dynamic_quantization("DynamicQuantiz", 14), "truncated marker");
    std::string blob = std::string(100, '\0') + marker + std::string(10, 'x');
    ASSERT(onnx_bytes_use_dynamic_quantization(blob.data(), blob.size()), "marker in blob");

    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / "lembed_t46_detect";
    fs::create_directories(dir, ec);

    /* The file reader works in 1 MiB chunks: put the marker across the first
     * chunk boundary, then past it, then nowhere. */
    const size_t chunk = (size_t)1 << 20;
    const fs::path split = dir / "split.onnx";
    write_file(split, std::string(chunk - 7, 'a') + marker + std::string(64, 'b'));
    ASSERT(onnx_file_uses_dynamic_quantization(split.string()), "marker across chunk boundary");

    const fs::path late = dir / "late.onnx";
    write_file(late, std::string(3 * chunk + 11, 'a') + marker);
    ASSERT(onnx_file_uses_dynamic_quantization(late.string()), "marker in a later chunk");

    const fs::path plain = dir / "plain.onnx";
    write_file(plain, std::string(2 * chunk + 3, 'a') + "MatMul");
    ASSERT(!onnx_file_uses_dynamic_quantization(plain.string()), "FP32-like file");

    const fs::path empty = dir / "empty.onnx";
    write_file(empty, "");
    ASSERT(!onnx_file_uses_dynamic_quantization(empty.string()), "empty file");

    ASSERT(onnx_file_uses_dynamic_quantization((dir / "missing.onnx").string()),
           "unreadable file counts as quantized");
    fs::remove_all(dir, ec);
}

static bool embed(lembed_text_embedding_t* ctx, const std::vector<const char*>& texts,
                  std::vector<float>& out, int* dim) {
    lembed_embeddings_t r;
    r.data = nullptr; r.num_embeddings = 0; r.dim = 0;
    if (lembed_text_embedding_embed(ctx, texts.data(), (int)texts.size(), 32, &r) != LEMBED_OK) {
        return false;
    }
    *dim = r.dim;
    out.assign(r.data, r.data + (size_t)r.num_embeddings * (size_t)r.dim);
    lembed_embeddings_free(&r);
    return true;
}

static void part_b_registry_variant() {
    const int variant = lembed_find_text_model_variant("sentence-transformers/all-MiniLM-L6-v2",
                                                       LEMBED_QUANTIZATION_DYNAMIC);
    if (variant < 0) {
        printf("SKIP part B: no dynamic all-MiniLM-L6-v2 entry in the registry\n");
        return;
    }
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = (lembed_text_model_t)variant;
    opts.offline = 1;
    opts.batch_strategy = LEMBED_BATCH_LENGTH_BUCKET;
    lembed_text_embedding_t* ctx = nullptr;
    if (lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK) {
        printf("SKIP part B: %s\n", lembed_last_error());
        return;
    }

    const std::vector<const char*> texts = {
        "Hi.",
        "The quick brown fox jumps over the lazy dog.",
        "Embeddings map text to vectors.",
        "performance performance performance performance performance performance "
        "performance performance performance performance performance performance.",
        "Short one",
        "A medium sentence about batching several sequences into a single run.",
    };

    lembed_stats_t before, after;
    lembed_text_embedding_stats(ctx, &before);
    std::vector<float> batch;
    int dim = 0;
    ASSERT(embed(ctx, texts, batch, &dim), "embed the batch");
    lembed_text_embedding_stats(ctx, &after);
    ASSERT(after.batches_run - before.batches_run == texts.size(),
           "one ONNX run per text despite LENGTH_BUCKET");

    /* One text per run: the batch result is the single-text result, bit for bit
     * up to float noise. Batched, the int8 scales would move it (cosine ~0.98). */
    float worst = 0.0f;
    for (size_t i = 0; i < texts.size() && dim > 0; i++) {
        std::vector<float> one;
        int d = 0;
        if (!embed(ctx, {texts[i]}, one, &d) || d != dim) {
            ASSERT(false, "embed one text");
            continue;
        }
        for (int k = 0; k < dim; k++) {
            worst = std::fmax(worst, std::fabs(one[(size_t)k] - batch[i * (size_t)dim + (size_t)k]));
        }
    }
    ASSERT(dim > 0 && worst < 1e-6f, "batch vectors equal single-text vectors");
    lembed_text_embedding_free(ctx);
}

int main() {
    part_a_detector();
    part_b_registry_variant();
    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
