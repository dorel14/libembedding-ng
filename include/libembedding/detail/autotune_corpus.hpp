/*
 * libembedding - detail/autotune_corpus.hpp
 * Synthetic workloads shared by the auto-tuners.
 *
 * The corpora are deterministic (same input -> same texts), so a tuning is
 * reproducible and its corpus fingerprint is stable across runs and machines,
 * which is what lets the cache recognise it.
 *
 * Auteur: David Orel
 * Version: 1.13.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_CORPUS_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_CORPUS_HPP

#include <algorithm>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Mixed-length text corpus: 40% short (16 words), 40% medium (64), 20% long
 * (128). Real embedding traffic is not uniform, and padding inside a batch
 * makes the length mix part of what is being tuned. */
inline std::vector<std::string> synthetic_text_corpus(int n_samples) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "algorithms", "process", "data", "efficiently",
        "natural", "language", "processing", "enables", "understanding", "semantic",
        "embeddings", "represent", "meaningful", "vector", "representations",
        "transformer", "models", "utilize", "attention", "mechanisms",
        "deep", "neural", "networks", "learn", "patterns", "from", "training",
        "optimization", "techniques", "improve", "convergence", "accuracy",
        "inference", "latency", "throughput", "benchmark", "performance",
        "production", "deployment", "scalable", "reliable", "systems"
    };
    const int n_words = (int)(sizeof(words) / sizeof(words[0]));
    static const int lengths[] = {16, 16, 16, 16, 64, 64, 64, 64, 128, 128};
    const int n_lengths = (int)(sizeof(lengths) / sizeof(lengths[0]));

    std::vector<std::string> corpus;
    corpus.reserve(n_samples > 0 ? (size_t)n_samples : 0);
    for (int i = 0; i < n_samples; i++) {
        int n_tokens = lengths[i % n_lengths];
        std::string text;
        for (int j = 0; j < n_tokens; j++) {
            if (j > 0) text += " ";
            text += words[(i * 7 + j) % n_words];
        }
        corpus.push_back(text);
    }
    return corpus;
}

/* One reranker document of roughly `target_tokens` tokens (0.75 word/token). */
inline std::string synthetic_reranker_doc(int target_tokens, int seed) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "data", "model", "system", "algorithm",
        "search", "query", "document", "text", "information", "result",
        "process", "analysis", "method", "approach", "technique",
        "application", "performance", "evaluation", "research", "study",
    };
    const int n_words = (int)(sizeof(words) / sizeof(words[0]));
    std::string doc;
    int target_words = std::max(1, (int)(target_tokens * 0.75));
    for (int i = 0; i < target_words; i++) {
        if (i > 0) doc += " ";
        doc += words[(seed + i) % n_words];
    }
    return doc;
}

/* Borrowed C view of a corpus; valid while `corpus` is alive and unchanged. */
inline std::vector<const char*> corpus_c_strings(const std::vector<std::string>& corpus) {
    std::vector<const char*> out;
    out.reserve(corpus.size());
    for (const auto& s : corpus) out.push_back(s.c_str());
    return out;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_CORPUS_HPP */
