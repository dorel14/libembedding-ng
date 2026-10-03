/*
 * libembedding - detail/quantization_auto_impl.hpp
 * Automatic selection of the best available quantization variant for a model.
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_QUANTIZATION_AUTO_IMPL_HPP
#define LIBEMBEDDING_QUANTIZATION_AUTO_IMPL_HPP

#include "../downloader.h"
#include "../model_registry.h"
#include "../text_embedding.h"
/* detail/autotune_cache.hpp only, deliberately not ../autotune_cache.h: that
 * header pulls in detail/autotune_cache_impl.hpp under LIBEMBEDDING_IMPLEMENTATION,
 * and this file is reached from inside an extern "C" region where those C
 * functions would be compiled with C++ linkage and clash with their declarations.
 * Everything needed here -- the identity, the fingerprinting file name, the
 * fault-tolerant JSON accessors and the atomic write -- lives in the detail
 * header. */#include "autotune_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace lembed {
namespace detail {

/* A variant must beat the FP32 baseline by more than this to be worth a weights
 * swap. Below that the difference is inside the noise of a short benchmark. */
static const double kQuantMinGain = 1.05;

/* Bias toward dynamic INT8: another variant has to be this much faster to be
 * preferred instead. Dynamic INT8 is the safer of the quantized options quality
 * wise, so the default only moves when the win is clear. */
static const double kQuantDynamicPreference = 1.15;

/* Cap on recorded variants: NONE, STATIC, DYNAMIC, FP16. */
static const int kQuantMaxVariants = 4;

/* Cache layout: the hardware fingerprint (CPU, ORT version, library version) is
 * baked into the file name by get_cache_key(), and the identity fields are
 * re-checked on read. A decision measured on another machine or another library
 * version therefore cannot be served from cache. The variant tag keeps the
 * quantization decision in its own slot so it can never overwrite a
 * threads/batch tuning entry for the same model. */
inline autotune_cache_identity quant_auto_identity(const char* model_code) {
    autotune_cache_identity id;
    id.model = model_code ? model_code : "";
    id.variant = "quantization";
    id.corpus = "";
    id.objective = -1;
    id.mode = -1;
    return id;
}

inline std::string quant_auto_path(const autotune_cache_identity& id) {
    std::string variant = cache_variant_key(id.variant.c_str(), id.corpus, id.objective, id.mode);
    return get_cache_path(id.model.c_str(), "quantization", variant);
}

/* =========================================================================
 * Measurement
 * ========================================================================= */

/* Size of the resolved weights, in MB.
 *
 * Deliberately the on-disk size and not the resident set: peak RSS is a
 * high-water mark per process, so once the first variant has been loaded the
 * figure can never go down and cannot be attributed to a later variant. That is
 * the measurement bug fixed for the quantization benchmark (LE-7.5); reusing the
 * pattern here would make every variant after the first look like it used no
 * memory at all. File size is deterministic, per variant, and ordered the right
 * way (FP32 > FP16 > INT8 for the same architecture).
 *
 * External data is counted too. An ONNX model above the protobuf size limit is
 * stored as a small graph file plus a weights sidecar -- bge-small INT8 is a
 * 0.4 MB `model_quantized.onnx` next to a 33 MB `model_quantized.onnx_data`.
 * Stat'ing only the graph file would report the quantized variant as three
 * hundred times smaller than FP32, which is exactly backwards.
 */
inline double quant_auto_file_mb(lembed_text_model_t variant) {
    char* dir = nullptr;
    if (lembed_ensure_text_model(variant, nullptr, 0, 1, &dir) != LEMBED_OK) {
        return 0.0;
    }
    double bytes = 0.0;
    lembed_model_info_t info;
    if (dir && lembed_get_text_model_info(variant, &info) == LEMBED_OK) {
        std::filesystem::path p = std::filesystem::path(dir) / info.model_file;
        std::error_code ec;
        auto sz = std::filesystem::file_size(p, ec);
        if (!ec) bytes += (double)sz;

        /* Sidecars: "<file>_data", then the sharded "<file>_data_<n>" layout. */
        std::filesystem::path base = p;
        base += "_data";
        sz = std::filesystem::file_size(base, ec);
        if (!ec) bytes += (double)sz;
        for (int i = 0; i < 16; i++) {
            std::error_code ec2;
            std::filesystem::path sharded =
                std::filesystem::path(std::string(base.string()) + "_" + std::to_string(i));
            sz = std::filesystem::file_size(sharded, ec2);
            if (!ec2) bytes += (double)sz;
        }
    }
    if (dir) free(dir);
    return bytes / (1024.0 * 1024.0);
}

/* =========================================================================
 * Measurement
 *
 * Two phases, because the variants are not comparable in cost: on the same
 * machine FP32 and INT8 of one model can differ by 3x, and a 12x-slower FP16
 * variant would otherwise dominate the whole selection while proving nothing
 * that a fraction of the time would not.
 *
 *   Phase 1 -- probe. A rejector, not a ranker. Each variant runs until it has
 *     done `kProbeMaxDocs` documents or `kProbeBudgetMs` has elapsed, whichever
 *     comes first, and is dropped if it is more than `kProbeRejectRatio` slower
 *     than the best rate seen so far. The time budget is the important half: a
 *     doc-count-only probe still pays full price on a pathological variant,
 *     which is exactly the case worth avoiding.
 *
 *   Phase 2 -- measure. The FP32 baseline and the survivors only, on the full
 *     corpus with a separate warmup, so the numbers the decision rule compares
 *     come from identical work.
 * ========================================================================= */

/* Probe limits. 4 documents is enough to rank; 250 ms caps what a slow variant
 * can cost. Together they bound phase 1 at ~750 ms for three variants whatever
 * their relative speed. */
static const int    kProbeMaxDocs  = 4;
static const double kProbeBudgetMs = 250.0;
static const double kProbeRejectRatio = 3.0;

/* Phase 2 corpus. Only two or three variants ever reach it, and the warmup is a
 * small fixed slice rather than the whole corpus: warming up on 8 documents to
 * then time 8 doubles the cost of the measurement for no extra stability. */
static const int kMeasureDocs = 16;
static const int kWarmupDocs  = 2;

/* Opens a session for one variant. Returns nullptr when the weights are absent
 * or the provider is unavailable, which the caller reads as "skip", not "slow":
 * AUTO must not download weights just to measure them. */
inline lembed_text_embedding_t* quant_auto_open(lembed_text_model_t variant,
                                                int threads, int batch_size) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = variant;
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* ctx = nullptr;
    if (lembed_text_embedding_create(&opts, &ctx) != LEMBED_OK) return nullptr;
    return ctx;
}

/* Phase 1. Returns docs/sec, or 0.0 when the variant could not be timed at all.
 * `best_so_far` is the running best rate; a variant below best/kProbeRejectRatio
 * stops early and reports 0.0, which the caller treats as "eliminated". */
inline double quant_auto_probe(lembed_text_embedding_t* ctx,
                               const std::vector<std::string>& corpus,
                               int batch_size, double best_so_far) {
    if (!ctx || corpus.empty()) return 0.0;
    const int n = (int)corpus.size();

    /* One untimed document first: the first embed pays kernel selection and
     * buffer allocation, and timing that would reject a variant for being
     * first rather than for being slow. */
    const char* first = corpus[0].c_str();
    lembed_embeddings_t w = {0};
    if (lembed_text_embedding_embed(ctx, &first, 1, batch_size, &w)
        != LEMBED_OK) {
        return 0.0;
    }
    lembed_embeddings_free(&w);

    auto t0 = std::chrono::high_resolution_clock::now();
    int done = 0;
    bool ok = true;
    for (int i = 1; i < n && done < kProbeMaxDocs; i++) {
        int chunk = 1;
        /* Stop as soon as the budget is spent, so a very slow variant cannot
         * make the probe unbounded. */
        double elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - t0)
                             .count();
        if (elapsed >= kProbeBudgetMs) break;

        const char* text = corpus[i].c_str();
        lembed_embeddings_t r = {0};
        ok = (lembed_text_embedding_embed(ctx, &text, chunk,
                                          batch_size, &r) == LEMBED_OK);
        lembed_embeddings_free(&r);
        if (!ok) return 0.0;
        done += chunk;

        /* Early rejection once enough documents have been timed to be worth it. */
        if (done >= 2 && best_so_far > 0.0) {
            elapsed = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0)
                          .count();
            if (elapsed > 0.0) {
                double rate = (double)done / (elapsed / 1000.0);
                if (rate * kProbeRejectRatio < best_so_far) return 0.0;
            }
        }
    }
    if (done <= 0) return 0.0;

    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0)
                    .count();
    if (ms <= 0.0) return 0.0;
    return (double)done / (ms / 1000.0);
}

/* Phase 2. Full-corpus measurement with a separate warmup slice. */
inline double quant_auto_measure(lembed_text_embedding_t* ctx,
                                 const std::vector<std::string>& corpus,
                                 int batch_size,
                                 double& out_latency_ms) {
    out_latency_ms = 0.0;
    if (!ctx || corpus.empty()) return 0.0;

    int n = (int)corpus.size();
    std::vector<const char*> ct;
    ct.reserve(n);
    for (int i = 0; i < n; i++) ct.push_back(corpus[i].c_str());

    /* Warmup is a fixed small slice, not the whole corpus: it only has to pay
     * the first-call costs, and warming up on the full corpus would double the
     * cost of every measurement. */
    int warm = n < kWarmupDocs ? n : kWarmupDocs;
    lembed_embeddings_t w = {0};
    lembed_text_embedding_embed(ctx, ct.data(), warm, batch_size, &w);
    lembed_embeddings_free(&w);

    auto t0 = std::chrono::high_resolution_clock::now();
    lembed_embeddings_t r = {0};
    lembed_status_t st = lembed_text_embedding_embed(ctx, ct.data(), n, batch_size, &r);
    auto t1 = std::chrono::high_resolution_clock::now();
    lembed_embeddings_free(&r);

    if (st != LEMBED_OK) return 0.0;
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms <= 0.0) return 0.0;
    out_latency_ms = ms / (double)n;
    return (double)n / (ms / 1000.0);
}

/* =========================================================================
 * Decision
 * ========================================================================= */

/* Picks a winner from a measured table.
 *
 * Separate from the measurement on purpose: the rule is the part worth testing,
 * and it can be exercised with a synthetic table without loading a model. */
inline int quant_auto_pick(const lembed_quantization_choice_t& measured) {
    double baseline = 0.0;
    double dynamic = 0.0;
    bool have_dynamic = false;

    for (int i = 0; i < measured.num_measured && i < kQuantMaxVariants; i++) {
        int q = measured.measured[i].quantization;
        double dps = measured.measured[i].docs_per_sec;
        if (dps <= 0.0) continue;
        if (q == LEMBED_QUANTIZATION_NONE) {
            baseline = dps;
        } else if (q == LEMBED_QUANTIZATION_DYNAMIC) {
            dynamic = dps;
            have_dynamic = true;
        }
    }

    /* No usable FP32 baseline: stay on FP32 rather than trust the table. */
    if (baseline <= 0.0) return LEMBED_QUANTIZATION_NONE;

    /* Nothing clears the noise floor: FP32 stays. */
    int best = LEMBED_QUANTIZATION_NONE;
    double best_dps = baseline;
    for (int i = 0; i < measured.num_measured && i < kQuantMaxVariants; i++) {
        int q = measured.measured[i].quantization;
        double dps = measured.measured[i].docs_per_sec;
        if (q == LEMBED_QUANTIZATION_NONE || dps <= 0.0) continue;
        if (dps / baseline >= kQuantMinGain && dps > best_dps) {
            best = q;
            best_dps = dps;
        }
    }
    if (best == LEMBED_QUANTIZATION_NONE) return LEMBED_QUANTIZATION_NONE;

    /* Dynamic INT8 measured and cleared the floor: it is the default choice
     * unless something is clearly faster still. */
    if (have_dynamic && best != LEMBED_QUANTIZATION_DYNAMIC &&
        best_dps / dynamic < kQuantDynamicPreference) {
        return LEMBED_QUANTIZATION_DYNAMIC;
    }
    return best;
}

inline std::string quant_auto_reason(const lembed_quantization_choice_t& c) {
    double baseline = 0.0;
    double winner = 0.0;
    for (int i = 0; i < c.num_measured && i < kQuantMaxVariants; i++) {
        double dps = c.measured[i].docs_per_sec;
        if (dps <= 0.0) continue;
        if (c.measured[i].quantization == LEMBED_QUANTIZATION_NONE) baseline = dps;
        if (c.measured[i].quantization == c.quantization) winner = dps;
    }
    char buf[192];
    if (baseline <= 0.0 || winner <= 0.0) {
        snprintf(buf, sizeof(buf), "no baseline measured, kept FP32");
    } else {
        snprintf(buf, sizeof(buf), "%.2fx the fp32 variant (%.0f -> %.0f docs/s)",
                 winner / baseline, baseline, winner);
    }
    return std::string(buf);
}

/* =========================================================================
 * Cache read / write
 * ========================================================================= */

/* Returns false on a miss, on a foreign entry and on a partial entry, so the
 * caller never applies a half-filled choice.
 *
 * Parsing is per line and per field rather than per line and per variant: the
 * four fields of a variant live on four different lines, so a reader that looked
 * for them all on the docs_per_sec line would silently recover throughput and
 * drop latency and file size. */
inline bool quant_auto_load(const autotune_cache_identity& id,
                            lembed_quantization_choice_t* out) {
    if (!out) return false;
    std::string path = quant_auto_path(id);
    if (!cache_identity_matches(path, id)) return false;

    std::ifstream f(path);
    if (!f.is_open()) return false;

    memset(out, 0, sizeof(*out));
    out->from_cache = 1;

    int best = -1;
    int n_measured = -1;
    bool seen[kQuantMaxVariants] = {false, false, false, false};
    int highest = -1;
    std::string reason;
    std::string line;
    while (std::getline(f, line)) {
        if (json_number(line, "quantization", best)) continue;
        if (json_number(line, "num_measured", n_measured)) continue;
        if (json_string(line, "reason", reason)) continue;

        for (int i = 0; i < kQuantMaxVariants; i++) {
            const std::string prefix = "q" + std::to_string(i) + "_";
            double v = 0.0;
            if (!seen[i]) {
                if (json_number(line, (prefix + "docs_per_sec").c_str(), v)) {
                    out->measured[i].docs_per_sec = v;
                    seen[i] = true;
                } else if (i > highest) {
                    continue;
                }
            }
            if (json_number(line, (prefix + "quantization").c_str(), v)) {
                out->measured[i].quantization = (int)v;
            }
            if (json_number(line, (prefix + "latency_ms").c_str(), v)) {
                out->measured[i].latency_ms = v;
            }
            if (json_number(line, (prefix + "file_mb").c_str(), v)) {
                out->measured[i].file_mb = v;
            }
            if (seen[i] && i > highest) highest = i;
        }
    }

    if (best < 0 || highest < 0) {
        memset(out, 0, sizeof(*out));
        return false;
    }
    out->num_measured = highest + 1;
    if (n_measured != out->num_measured) {
        memset(out, 0, sizeof(*out));
        return false;
    }
    out->quantization = best;
    out->variant_model = best; /* registry index is filled in by the caller */
    snprintf(out->reason, sizeof(out->reason), "%s", reason.c_str());
    return true;
}

inline void quant_auto_save(const autotune_cache_identity& id,
                            const lembed_quantization_choice_t& choice) {
    std::ostringstream os;
    os << "{\n";
    write_cache_identity(os, id);
    os << "  \"quantization\": " << choice.quantization << ",\n";
    os << "  \"num_measured\": " << choice.num_measured << ",\n";
    os << "  \"reason\": \"" << choice.reason << "\",\n";
    for (int i = 0; i < choice.num_measured && i < kQuantMaxVariants; i++) {
        std::string p = "\"q" + std::to_string(i) + "_";
        os << "  " << p << "quantization\": " << choice.measured[i].quantization << ",\n";
        os << "  " << p << "docs_per_sec\": " << choice.measured[i].docs_per_sec << ",\n";
        os << "  " << p << "latency_ms\": " << choice.measured[i].latency_ms << ",\n";
        os << "  " << p << "file_mb\": " << choice.measured[i].file_mb;
        if (i + 1 < choice.num_measured && i + 1 < kQuantMaxVariants) os << ",";
        os << "\n";
    }
    os << "}\n";
    write_file_atomic(quant_auto_path(id), os.str());
}

/* =========================================================================
 * Entry point
 * ========================================================================= */

/* Collects the variants the registry actually offers for this model, in a fixed
 * order (NONE first so the FP32 baseline is always present in the table).
 * A mode with no sibling entry is simply absent: the registry has no STATIC
 * variants at all today, so a rule that assumed three would compare against a
 * variant that cannot exist. */
inline int quant_auto_collect_variants(lembed_text_model_t model,
                                       int* out, int max_out) {
    lembed_model_info_t info;
    int n = 0;
    if (lembed_get_text_model_info(model, &info) != LEMBED_OK) return 0;

    /* The baseline is the entry the caller asked for. */
    out[n++] = (int)model;

    static const int kOrder[] = {
        LEMBED_QUANTIZATION_DYNAMIC,
        LEMBED_QUANTIZATION_FP16,
        LEMBED_QUANTIZATION_STATIC,
    };
    for (size_t k = 0; k < sizeof(kOrder) / sizeof(kOrder[0]); k++) {
        if (n >= max_out) break;
        int v = lembed_find_text_model_variant(info.model_name, kOrder[k]);
        if (v >= 0) out[n++] = v;
    }
    return n;
}

} /* namespace detail */
} /* namespace lembed */

#endif /* LIBEMBEDDING_QUANTIZATION_AUTO_IMPL_HPP */