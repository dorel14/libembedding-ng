/*
 * libembedding - detail/quantization_auto_entry.hpp
 * Definition of the quantization auto-selection entry point.
 *
 * Kept free of the tuning machinery on purpose: the model creation path resolves
 * LEMBED_QUANTIZATION_AUTO, so this header has to be includable from
 * text_embedding_onnx_impl.hpp without pulling autotune_bench_text.hpp (which
 * itself needs text_embedding.h and would close an include cycle).
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_QUANTIZATION_AUTO_ENTRY_HPP
#define LIBEMBEDDING_QUANTIZATION_AUTO_ENTRY_HPP

#include "quantization_auto_impl.hpp"

#ifdef __cplusplus
/* The includer may sit inside an extern "C" region; every C++ helper below
 * needs C++ linkage, and the public entry point has to be defined with C linkage
 * to match its declaration in autotuner.h. The header therefore dictates its own
 * linkage instead of relying on the includer. */
extern "C++" {
#endif

namespace lembed {
namespace detail {

/* Short synthetic corpus for the variant benchmark. Kept local rather than
 * reusing generate_corpus() from autotune_bench_text.hpp to avoid the include
 * cycle described above. Lengths are mixed so a variant cannot look good only
 * because the corpus happened to be all padding. */
inline std::vector<std::string> quant_auto_corpus(int n_samples) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "algorithms", "process", "data", "efficiently",
        "natural", "language", "processing", "enables", "understanding", "semantic",
        "embeddings", "represent", "meaningful", "vector", "representations",
        "transformer", "models", "utilize", "attention", "mechanisms",
        "deep", "neural", "networks", "learn", "patterns", "from", "training",
        "inference", "latency", "throughput", "benchmark", "performance"
    };
    const int nwords = (int)(sizeof(words) / sizeof(words[0]));
    const int lengths[] = {16, 16, 16, 64, 64, 64, 128, 128};
    const int n_lengths = (int)(sizeof(lengths) / sizeof(lengths[0]));

    std::vector<std::string> corpus;
    corpus.reserve(n_samples);
    for (int i = 0; i < n_samples; i++) {
        const int ntok = lengths[i % n_lengths];
        std::string text;
        for (int j = 0; j < ntok; j++) {
            if (j > 0) text += " ";
            text += words[(i * 7 + j) % nwords];
        }
        corpus.push_back(text);
    }
    return corpus;
}

} /* namespace detail */
} /* namespace lembed */

#ifdef __cplusplus
} /* extern "C++" */

extern "C" {
#endif

lembed_status_t lembed_quantization_auto_select(
    lembed_text_model_t model,
    int num_threads,
    int batch_size,
    int num_docs,
    int dry_run,
    lembed_quantization_choice_t* result) {

    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;

    memset(result, 0, sizeof(*result));

    if (batch_size <= 0) batch_size = LEMBED_DEFAULT_BATCH_SIZE;

    lembed_model_info_t info;
    if (lembed_get_text_model_info(model, &info) != LEMBED_OK) {
        return LEMBED_ERROR_MODEL_NOT_FOUND;
    }

    auto id = lembed::detail::quant_auto_identity(info.model_code);

    /* Cache first: a recorded decision is free, a benchmark costs seconds. */
    if (lembed::detail::quant_auto_load(id, result)) {
        int variant = (result->quantization == (int)info.quantization)
                          ? (int)model
                          : lembed_find_text_model_variant(info.model_name,
                                                           result->quantization);
        if (variant >= 0) {
            result->variant_model = variant;
            return LEMBED_OK;
        }
        /* The winning variant is no longer in the registry: measure again
         * rather than silently load something else. */
        memset(result, 0, sizeof(*result));
    }

    int variants[lembed::detail::kQuantMaxVariants + 1];
    const int max_variants = (int)(sizeof(variants) / sizeof(variants[0]));
    int n_variants =
        lembed::detail::quant_auto_collect_variants(model, variants, max_variants);
    if (n_variants <= 0) return LEMBED_ERROR_MODEL_NOT_FOUND;

    /* Phase 2 runs on this corpus, so its size is the real measurement cost.
     * kMeasureDocs is the default: enough to separate variants that differ by
     * tens of percent, small enough that the first model load does not stall.
     * A caller asking for more gets more. */
    int corpus_size = (num_docs > 0) ? num_docs : lembed::detail::kMeasureDocs;
    if (corpus_size < lembed::detail::kMeasureDocs) corpus_size = lembed::detail::kMeasureDocs;
    if (corpus_size > 256) corpus_size = 256;
    std::vector<std::string> corpus =
        lembed::detail::quant_auto_corpus(corpus_size);

    /* The baseline always has to be measured, whatever the probe says about it,
     * because the decision rule compares everything to it. Slot 0 is the FP32
     * entry of the family, not `model`: the caller may have pointed at an
     * already-quantized sibling, and comparing the siblings against that would
     * leave the table with no baseline to compare against. */
    const int baseline_entry = variants[0];
    std::vector<int> candidates;
    candidates.push_back(baseline_entry);

    /* ---- Phase 1: probe, and drop the hopeless ----
     *
     * A surviving variant is loaded twice, once here and once in phase 2. The
     * probe session is freed before phase 2 opens its own so the two never
     * overlap in memory, but the file is read twice. That is the price of never
     * materialising more than one session at a time, and it is bounded: the
     * probe is only reached on a cache miss, i.e. once per machine per model. */
    double best_rate = 0.0;
    for (int i = 1; i < n_variants; i++) {
        lembed_text_embedding_t* ctx =
            lembed::detail::quant_auto_open((lembed_text_model_t)variants[i],
                                            num_threads, batch_size);
        if (!ctx) continue; /* weights absent: skip, never download */

        double rate = lembed::detail::quant_auto_probe(
            ctx, corpus, batch_size, best_rate);
        lembed_text_embedding_free(ctx);
        if (rate <= 0.0) continue; /* eliminated */
        if (rate > best_rate) best_rate = rate;
        candidates.push_back(variants[i]);
    }

    /* ---- Phase 2: real measurement on the survivors ---- */
    lembed_quantization_choice_t measured;
    memset(&measured, 0, sizeof(measured));

    for (size_t i = 0; i < candidates.size(); i++) {
        if (measured.num_measured >= lembed::detail::kQuantMaxVariants) break;
        lembed_model_info_t vi;
        if (lembed_get_text_model_info((lembed_text_model_t)candidates[i], &vi)
            != LEMBED_OK) {
            continue;
        }
        lembed_text_embedding_t* ctx =
            lembed::detail::quant_auto_open((lembed_text_model_t)candidates[i],
                                            num_threads, batch_size);
        if (!ctx) continue;

        int slot = measured.num_measured++;
        measured.measured[slot].quantization = (int)vi.quantization;
        measured.measured[slot].file_mb =
            lembed::detail::quant_auto_file_mb((lembed_text_model_t)candidates[i]);
        measured.measured[slot].docs_per_sec =
            lembed::detail::quant_auto_measure(ctx, corpus, batch_size,
                                               measured.measured[slot].latency_ms);
        lembed_text_embedding_free(ctx);

        /* A variant that could not be measured is dropped rather than recorded
         * as a zero: a zero reads as "infinitely slow" and, under a max() rule,
         * would win by being the only entry left. */
        if (measured.measured[slot].docs_per_sec <= 0.0) {
            measured.num_measured--;
        }
    }

    if (measured.num_measured <= 0) {
        /* Nothing measurable: keep the requested entry and say so, rather than
         * reporting a mode nothing can load. */
        measured.quantization = (int)info.quantization;
        measured.variant_model = (int)model;
        snprintf(measured.reason, sizeof(measured.reason),
                 "no variant could be measured, kept %s",
                 lembed__quantization_name((lembed_quantization_t)info.quantization));
        *result = measured;
        return LEMBED_OK;
    }

    /* quant_auto_pick answers NONE both for "FP32 won" and for "no FP32 row could be
     * measured". Those two must not collapse: the second happens exactly when the
     * FP32 weights are absent from disk, and answering NONE there would make AUTO
     * pick -- and the caller's final, non-offline create then download -- FP32
     * weights for a caller who asked for a quantized sibling. With no baseline
     * there is nothing to compare, so the entry the caller named stands. */
    bool have_baseline = false;
    for (int i = 0; i < measured.num_measured && i < lembed::detail::kQuantMaxVariants;
         i++) {
        if (measured.measured[i].quantization == LEMBED_QUANTIZATION_NONE &&
            measured.measured[i].docs_per_sec > 0.0) {
            have_baseline = true;
            break;
        }
    }

    int winner = have_baseline ? lembed::detail::quant_auto_pick(measured)
                               : (int)info.quantization;
    int variant = (winner == (int)info.quantization)
                      ? (int)model
                      : lembed_find_text_model_variant(info.model_name, winner);

    measured.quantization = (variant >= 0) ? winner : (int)info.quantization;
    measured.variant_model = (variant >= 0) ? variant : (int)model;

    std::string why = lembed::detail::quant_auto_reason(measured);
    snprintf(measured.reason, sizeof(measured.reason), "%s", why.c_str());

    if (!dry_run) {
        lembed::detail::quant_auto_save(id, measured);
    }
    measured.from_cache = 0;
    *result = measured;
    return LEMBED_OK;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_QUANTIZATION_AUTO_ENTRY_HPP */