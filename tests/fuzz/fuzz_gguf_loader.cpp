/*
 * libembedding - tests/fuzz/fuzz_gguf_loader.cpp
 * libFuzzer harness for the sparse GGUF loader (roadmap T2b).
 *
 * Target: lembed::gguf::SparseSession::load_from_buffer(), which runs exactly
 * the checks of the public lembed_sparse_text_embedding_create_from_gguf_path()
 * after the parse: gguf's reader (gguf_init_from_buffer and gguf_init_from_file
 * share gguf_init_from_reader), then Weights::read_hparams() bounds,
 * the capability check, Weights::validate_tensors() shapes and types, the
 * tensor copy, the vocabulary and the tokenizer. Hardened in T2 (PR #19).
 *
 * One input is one GGUF image in memory: no temp file per run.
 *
 * When a file is accepted, one short text is embedded with it. T2's shape
 * checks exist so that an accepted file can never reach a GGML_ASSERT inside
 * the graph; running the graph is what puts that claim under test.
 *
 * Build: clang only, -DLIBEMBEDDING_BUILD_FUZZERS=ON (docs/gguf_sparse_runtime.md,
 * section 4.1).
 * Seeds: `test_sparse_gguf --write-fuzz-seeds <dir>` writes the fixtures that
 * the unit tests use.
 *
 * Auteur: David Orel
 * Version: 1.14.0
 *
 * SPDX-License-Identifier: MIT
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

/* gguf logs every refusal to stderr. Millions of runs of that drown the one
 * report that matters, and slow the loop down. */
void quiet_log(enum ggml_log_level, const char*, void*) {}

} /* namespace */

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    ggml_log_set(quiet_log, nullptr);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    lembed::gguf::SparseSession session;
    std::string err;
    /* One thread, one worker: no thread pool to start per input, and any
     * finding replays deterministically. max_length 8 keeps the graph small
     * whatever context length the input declares. */
    if (!session.load_from_buffer(data, size, 1, 1, 8, err)) return 0;

    const std::vector<std::string> texts{"hello world"};
    (void)session.embed(texts, 1);
    return 0;
}
