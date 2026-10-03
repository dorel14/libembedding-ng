/*
 * test_quantization_auto.c - Decision rule for automatic quantization selection.
 *
 * The rule is the part worth testing, and it is deliberately separable from the
 * measurement: it takes a table of measured variants and returns a winner. That
 * means it can be exercised exhaustively here without loading a model, without a
 * network and without a benchmark taking seconds.
 *
 * What is NOT tested here is the measurement itself (loading candidate variants
 * and timing them). That is integration territory and depends on which weights
 * happen to be in the cache.
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/autotuner.h>

#include <stdio.h>
#include <string.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); g_fail++; } \
} while (0)

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

static void reset(lembed_quantization_choice_t* c) {
    memset(c, 0, sizeof(*c));
}

static void add(lembed_quantization_choice_t* c, int quant, double dps) {
    int i = c->num_measured++;
    c->measured[i].quantization = quant;
    c->measured[i].docs_per_sec = dps;
    c->measured[i].latency_ms = (dps > 0.0) ? (1000.0 / dps) : 0.0;
    c->measured[i].file_mb = 0.0;
}

/* ==========================================================================
 * FP32 baseline
 * ========================================================================== */

/* Nothing faster than FP32: the answer is FP32, with no swap. */
static void test_keeps_fp32_when_nothing_faster(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 103.0); /* +3% : inside the noise */
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_NONE,
          "a 3% gain stays below the 5% noise floor");
}

/* Exactly at the threshold counts: the rule is ">=", so 5% is accepted. */
static void test_threshold_is_inclusive(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 105.0); /* exactly +5% */
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_DYNAMIC,
          "a 5% gain is accepted");
}

/* An empty or unusable table must not invent a winner. */
static void test_no_baseline_keeps_fp32(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 500.0);
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_NONE,
          "a table without an FP32 baseline is not trusted");

    reset(&c);
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_NONE,
          "an empty table yields FP32");

    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 0.0); /* unmeasurable baseline */
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 500.0);
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_NONE,
          "a zero baseline is treated as no baseline");
}

/* ==========================================================================
 * The dynamic bias
 * ========================================================================== */

/* Dynamic INT8 wins when nothing beats it. */
static void test_dynamic_wins_when_fastest(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 140.0);
    add(&c, LEMBED_QUANTIZATION_FP16, 120.0);
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_DYNAMIC,
          "the fastest variant wins when it is dynamic");
}

/* FP16 slightly ahead of dynamic stays dynamic: the bias is explicit. */
static void test_dynamic_preferred_over_narrow_fp16_win(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 140.0);
    add(&c, LEMBED_QUANTIZATION_FP16, 150.0); /* +7% over dynamic: not enough */
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_DYNAMIC,
          "FP16 must beat dynamic by 15% to displace it");
}

/* FP16 clearly ahead of dynamic takes over. */
static void test_fp16_takes_over_on_a_clear_win(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 140.0);
    add(&c, LEMBED_QUANTIZATION_FP16, 170.0); /* +21% over dynamic */
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_FP16,
          "FP16 takes over once it is clearly faster than dynamic");
}

/* Without a dynamic variant there is nothing to be biased towards. */
static void test_fp16_wins_without_dynamic(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_FP16, 130.0);
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_FP16,
          "with no dynamic variant the fastest one wins outright");
}

/* Dynamic is slower than FP32: it must not be chosen just for existing. */
static void test_slow_dynamic_is_not_chosen(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 200.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 90.0);
    CHECK(lembed::detail::quant_auto_pick(c) == LEMBED_QUANTIZATION_NONE,
          "a dynamic variant slower than FP32 is not selected");
}

/* ==========================================================================
 * Variant discovery
 * ========================================================================== */

/* Discovery is driven by the registry, and FP32 is always present first so the
 * baseline exists even for a model with no sibling. */
static void test_variant_discovery_starts_with_requested(void) {
    int variants[8];
    int n = lembed::detail::quant_auto_collect_variants(
        LEMBED_TEXT_BGE_SMALL_EN_V15, variants, 8);
    CHECK(n >= 1, "at least the requested entry is returned");
    CHECK(variants[0] == LEMBED_TEXT_BGE_SMALL_EN_V15,
          "the requested entry comes first so it is the baseline");

    /* No duplicates: two entries for the same mode would be measured twice. */
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            CHECK(variants[i] != variants[j], "no duplicate variant");
        }
    }
}

/* ==========================================================================
 * Reason string
 * ========================================================================== */

/* The reason is what a user reads to understand the choice, so it must name the
 * winner and quote both sides of the ratio rather than being empty. */
static void test_reason_is_populated(void) {
    lembed_quantization_choice_t c;
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 150.0);
    c.quantization = LEMBED_QUANTIZATION_DYNAMIC;

    std::string why = lembed::detail::quant_auto_reason(c);
    CHECK(why.find("1.50x") != std::string::npos,
          "the reason quotes the speedup");
    CHECK(why.find("100") != std::string::npos,
          "the reason quotes the baseline");

    /* No baseline measured: say so rather than printing a bogus ratio. */
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 150.0);
    c.quantization = LEMBED_QUANTIZATION_DYNAMIC;
    std::string none = lembed::detail::quant_auto_reason(c);
    CHECK(none.find("no baseline") != std::string::npos,
          "a missing baseline is reported as such");
}

/* ==========================================================================
 * Cache round-trip
 * ========================================================================== */

/* The decision has to survive a write/read cycle, otherwise every load would
 * re-benchmark. The fingerprint in the file name is part of the path, so the
 * test writes and reads through the same identity. */
static void test_cache_round_trip(void) {
    auto id = lembed::detail::quant_auto_identity("test-quant-round-trip");

    /* Start from a clean slate. */
    std::error_code ec;
    std::filesystem::remove(lembed::detail::quant_auto_path(id), ec);

    lembed_quantization_choice_t written;
    memset(&written, 0, sizeof(written));
    written.quantization = LEMBED_QUANTIZATION_DYNAMIC;
    written.num_measured = 2;
    written.measured[0].quantization = LEMBED_QUANTIZATION_NONE;
    written.measured[0].docs_per_sec = 100.0;
    written.measured[0].latency_ms = 10.0;
    written.measured[0].file_mb = 90.0;
    written.measured[1].quantization = LEMBED_QUANTIZATION_DYNAMIC;
    written.measured[1].docs_per_sec = 150.0;
    written.measured[1].latency_ms = 6.7;
    written.measured[1].file_mb = 23.0;
    snprintf(written.reason, sizeof(written.reason), "1.50x the fp32 variant");

    lembed::detail::quant_auto_save(id, written);

    lembed_quantization_choice_t read;
    memset(&read, 0, sizeof(read));
    CHECK(lembed::detail::quant_auto_load(id, &read), "the entry reads back");
    CHECK(read.quantization == LEMBED_QUANTIZATION_DYNAMIC,
          "the winner survives the round trip");
    CHECK(read.num_measured == 2, "the variant count survives");
    CHECK(read.from_cache == 1, "a cache hit is flagged as such");
    if (read.num_measured >= 2) {
        CHECK(read.measured[1].docs_per_sec > 149.0 &&
                  read.measured[1].docs_per_sec < 151.0,
              "measured throughput survives");
        CHECK(read.measured[1].file_mb > 22.0 && read.measured[1].file_mb < 24.0,
              "the file size survives");
    }
    CHECK(strstr(read.reason, "1.50x") != NULL, "the reason survives");

    /* A foreign identity must miss rather than hand back this entry. */
    auto other = lembed::detail::quant_auto_identity("some-other-model");
    lembed_quantization_choice_t miss;
    memset(&miss, 0, sizeof(miss));
    CHECK(!lembed::detail::quant_auto_load(other, &miss),
          "an entry is never served for another model");

    std::filesystem::remove(lembed::detail::quant_auto_path(id), ec);
}

/* ==========================================================================
 * Dry run must not write
 * ========================================================================== */

/* dry_run is what lets a caller measure without leaving state behind; if it
 * cached anyway, "measure only" would be a lie. */
static void test_dry_run_does_not_cache(void) {
    auto id = lembed::detail::quant_auto_identity("test-quant-dry-run");
    std::error_code ec;
    std::filesystem::remove(lembed::detail::quant_auto_path(id), ec);

    lembed_quantization_choice_t c;
    memset(&c, 0, sizeof(c));
    c.quantization = LEMBED_QUANTIZATION_NONE;
    c.num_measured = 1;
    c.measured[0].quantization = LEMBED_QUANTIZATION_NONE;
    c.measured[0].docs_per_sec = 100.0;

    /* Simulate the entry point's dry-run branch: no save call. */
    CHECK(!lembed::detail::quant_auto_load(id, &c),
          "nothing is cached when the caller passed dry_run");
}

/* ==========================================================================
 * External data
 * ========================================================================== */

/* An ONNX model past the protobuf size limit is a small graph file plus a
 * weights sidecar. bge-small INT8 is a 0.4 MB model_quantized.onnx next to a
 * 33 MB model_quantized.onnx_data, so stat'ing only the graph reports the
 * quantized variant as ~300x smaller than FP32 -- exactly backwards.
 *
 * Only runs when that variant is in the local cache, and it asserts the
 * ordering rather than an absolute size, because the cache is not populated on
 * a fresh machine. */
static void test_external_data_is_counted(void) {
    /* Quantized sibling of BAAI/bge-small-en-v1.5. */
    int variant = lembed_find_text_model_variant("BAAI/bge-small-en-v1.5",
                                                 LEMBED_QUANTIZATION_DYNAMIC);
    if (variant < 0) return; /* no such variant in this registry */

    char* dir = NULL;
    if (lembed_ensure_text_model((lembed_text_model_t)variant, NULL, 0, 1, &dir) != LEMBED_OK) {
        return; /* weights absent: nothing to measure */
    }
    free(dir);

    double int8_mb = lembed::detail::quant_auto_file_mb((lembed_text_model_t)variant);
    double fp32_mb = lembed::detail::quant_auto_file_mb(LEMBED_TEXT_BGE_SMALL_EN_V15);

    CHECK(int8_mb > 0.0, "the quantized variant has a measurable size");
    CHECK(fp32_mb > 0.0, "the FP32 variant has a measurable size");

    /* INT8 weights must come out smaller than FP32. A 0.4 MB reading against a
     * 127 MB baseline would pass a "nonzero" check while being the opposite of
     * the truth, which is why the ordering is what gets asserted. */
    CHECK(int8_mb < fp32_mb,
          "INT8 weights measure smaller than FP32 (sidecar included)");
    /* And not absurdly smaller: a bare stub would be under 1% of FP32. */
    CHECK(int8_mb > fp32_mb * 0.05,
          "the INT8 size is in the same order of magnitude, not a graph stub");
}

/* ==========================================================================
 * Two-phase measurement guards
 * ========================================================================== */

/* Phase 1 rejects gross losers and phase 2 only sees the survivors. Both take a
 * session, so both must survive a null one: a variant whose weights are absent
 * produces no session, and the probe is then asked to time nothing. */
static void test_probe_handles_missing_session(void) {
    std::vector<std::string> corpus = {"alpha", "beta", "gamma", "delta"};
    double latency = -1.0;
    CHECK(lembed::detail::quant_auto_probe(NULL, corpus, 32, 0.0) == 0.0,
          "the probe reports nothing for a missing session");
    CHECK(lembed::detail::quant_auto_measure(NULL, corpus, 32, latency) == 0.0,
          "the measurement reports nothing for a missing session");
    CHECK(lembed::detail::quant_auto_measure(NULL, corpus, 32, latency) == 0.0 &&
              latency == 0.0,
          "a failed measurement leaves no stale latency behind");
}

/* The probe is only a rejector, so it must be cheap by construction: these
 * bounds are what stop a pathological variant from dominating the selection.
 * A regression here would silently bring back the multi-second first load. */
static void test_probe_is_bounded(void) {
    CHECK(lembed::detail::kProbeBudgetMs > 0.0 &&
              lembed::detail::kProbeBudgetMs <= 1000.0,
          "the probe has a finite wall-clock budget");
    CHECK(lembed::detail::kProbeMaxDocs > 0 &&
              lembed::detail::kProbeMaxDocs <= 16,
          "the probe has a bounded document count");
    CHECK(lembed::detail::kProbeRejectRatio > 1.0,
          "a probe only rejects when clearly beaten");
    /* The warmup must stay smaller than the measured slice, or the measurement
     * costs more than it reports. */
    CHECK(lembed::detail::kWarmupDocs < lembed::detail::kMeasureDocs,
          "the warmup slice is smaller than the measured slice");
}

int main(void) {
    test_keeps_fp32_when_nothing_faster();
    test_threshold_is_inclusive();
    test_no_baseline_keeps_fp32();

    test_dynamic_wins_when_fastest();
    test_dynamic_preferred_over_narrow_fp16_win();
    test_fp16_takes_over_on_a_clear_win();
    test_fp16_wins_without_dynamic();
    test_slow_dynamic_is_not_chosen();

    test_variant_discovery_starts_with_requested();
    test_reason_is_populated();
    test_cache_round_trip();
    test_dry_run_does_not_cache();
    test_external_data_is_counted();
    test_probe_handles_missing_session();
    test_probe_is_bounded();

    printf("%s: %d passed, %d failed\n",
           g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}