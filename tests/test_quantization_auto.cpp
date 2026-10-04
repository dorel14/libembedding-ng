/*
 * test_quantization_auto.cpp - Decision rule for automatic quantization selection.
 *
 * The rule is the part worth testing, and it is deliberately separable from the
 * measurement: it takes a table of measured variants and returns a winner. That
 * means it can be exercised exhaustively here without loading a model, without a
 * network and without a benchmark taking seconds.
 *
 * The two cases that genuinely need a session -- the cache round trip through
 * the entry point, and the dry-run guarantee -- are skipped unless the weights
 * are already in the local cache, which is the same pattern the sidecar test
 * below uses. A test that "passes" because it never reached the branch it names
 * is worse than no test at all.
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/autotuner.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

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

/* Discovery is driven by the registry, and slot 0 is the FP32 entry of the
 * family: the decision rule compares everything to it, so a table without an FP32
 * row degenerates to "no baseline, stay safe". Anchoring slot 0 on the requested
 * entry instead meant a request for an already-quantized sibling under AUTO
 * produced a table with no baseline at all. */
static void test_variant_discovery_baselines_on_fp32(void) {
    int variants[8];
    int n = lembed::detail::quant_auto_collect_variants(
        LEMBED_TEXT_BGE_SMALL_EN_V15, variants, 8);
    CHECK(n >= 1, "at least one variant is returned");
    if (n < 1) return;

    int fp32 = lembed_find_text_model_variant("BAAI/bge-small-en-v1.5",
                                              LEMBED_QUANTIZATION_NONE);
    CHECK(variants[0] == fp32,
          "the FP32 entry is first so it can serve as the baseline");

    /* No duplicates: two entries for the same mode would be measured twice, and
     * the duplicate baseline in particular would be counted twice in the rule. */
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            CHECK(variants[i] != variants[j], "no duplicate variant");
        }
    }

    /* Every returned entry must belong to the same family, or the table compares
     * two unrelated models. */
    for (int i = 0; i < n; i++) {
        lembed_model_info_t vi;
        CHECK(lembed_get_text_model_info((lembed_text_model_t)variants[i], &vi) == LEMBED_OK,
              "each variant is a registry entry");
        CHECK(vi.model_name == std::string("BAAI/bge-small-en-v1.5"),
              "each variant belongs to the requested family");
    }
}

/* Asking for an already-quantized entry must not lose the FP32 baseline: that is
 * the case that used to promote the caller to FP32 without measuring anything. */
static void test_variant_discovery_keeps_baseline_from_quantized_request(void) {
    int dynamic = lembed_find_text_model_variant("BAAI/bge-small-en-v1.5",
                                                 LEMBED_QUANTIZATION_DYNAMIC);
    if (dynamic < 0) return; /* no such variant in this registry */

    int variants[8];
    int n = lembed::detail::quant_auto_collect_variants(
        (lembed_text_model_t)dynamic, variants, 8);
    CHECK(n >= 1, "a quantized request still returns variants");
    if (n < 1) return;

    int fp32 = lembed_find_text_model_variant("BAAI/bge-small-en-v1.5",
                                              LEMBED_QUANTIZATION_NONE);
    CHECK(variants[0] == fp32,
          "requesting a quantized sibling still baselines on FP32");
}

/* max_out is a hard bound: writing slot 0 before honouring it would overrun a
 * caller-sized buffer. */
static void test_variant_discovery_honours_max_out(void) {
    int one = 0;
    CHECK(lembed::detail::quant_auto_collect_variants(
              LEMBED_TEXT_BGE_SMALL_EN_V15, &one, 0) == 0,
          "a zero-sized output returns nothing");
    CHECK(lembed::detail::quant_auto_collect_variants(
              LEMBED_TEXT_BGE_SMALL_EN_V15, NULL, 4) == 0,
          "a null output returns nothing");

    int one_slot = -1;
    int n = lembed::detail::quant_auto_collect_variants(
        LEMBED_TEXT_BGE_SMALL_EN_V15, &one_slot, 1);
    CHECK(n == 1, "a single slot yields the baseline only");
    CHECK(one_slot >= 0, "the single slot holds the baseline");
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

    /* No baseline measured: say so rather than printing a bogus ratio, and do
     * not claim FP32 was kept -- which mode stands depends on the caller. */
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 150.0);
    c.quantization = LEMBED_QUANTIZATION_DYNAMIC;
    std::string none = lembed::detail::quant_auto_reason(c);
    CHECK(none.find("no fp32 baseline") != std::string::npos,
          "a missing baseline is reported as such");
    CHECK(none.find("dynamic") != std::string::npos,
          "a missing baseline names the mode that was kept");

    /* FP32 kept: "1.00x" would read as a measurement to trust rather than as
     * "nothing cleared the noise floor". */
    reset(&c);
    add(&c, LEMBED_QUANTIZATION_NONE, 100.0);
    add(&c, LEMBED_QUANTIZATION_DYNAMIC, 102.0); /* +2% : under the floor */
    c.quantization = LEMBED_QUANTIZATION_NONE;
    std::string kept = lembed::detail::quant_auto_reason(c);
    CHECK(kept.find("1.00x") == std::string::npos,
          "keeping FP32 does not report a 1.00x ratio");
    CHECK(kept.find("under the 1.05x floor") != std::string::npos,
          "keeping FP32 explains that the noise floor was not cleared");
}

/* The reason is written verbatim into a JSON cache file, so a quote or a
 * backslash in it would corrupt the entry. Today only digits and mode names reach
 * it, which is exactly why nothing noticed. */
static void test_reason_with_json_metacharacters(void) {
    auto id = lembed::detail::quant_auto_identity("test-quant-escape");
    std::error_code ec;
    std::filesystem::remove(lembed::detail::quant_auto_path(id), ec);

    const char* raw = "he said \"2.5x\\fast\" on <fp32>\nsecond line\tend";
    CHECK(lembed::detail::quant_auto_json_escape(std::string(raw)) !=
              std::string(raw),
          "JSON metacharacters are escaped on the way out");
    CHECK(lembed::detail::quant_auto_json_unescape(
              lembed::detail::quant_auto_json_escape(std::string(raw))) ==
              std::string(raw),
          "escaping and unescaping are inverses");

    /* And the escaped form must survive an actual save/load cycle. */
    lembed_quantization_choice_t written;
    memset(&written, 0, sizeof(written));
    written.quantization = LEMBED_QUANTIZATION_FP16;
    written.num_measured = 1;
    written.measured[0].quantization = LEMBED_QUANTIZATION_NONE;
    written.measured[0].docs_per_sec = 100.0;
    written.measured[0].latency_ms = 10.0;
    written.measured[0].file_mb = 90.0;
    snprintf(written.reason, sizeof(written.reason), "%s", raw);

    lembed::detail::quant_auto_save(id, written);

    lembed_quantization_choice_t read;
    memset(&read, 0, sizeof(read));
    CHECK(lembed::detail::quant_auto_load(id, &read),
          "an entry with a quoted reason still reads back");
    if (read.from_cache) {
        CHECK(std::string(read.reason) == std::string(raw),
              "the reason survives escaping and unescaping intact");
    }

    std::filesystem::remove(lembed::detail::quant_auto_path(id), ec);
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

/* Counts how many of a family's variants already have weights on disk. AUTO
 * never downloads anything to measure it, so without at least two of them there
 * is nothing to measure and the save branch is never reached. */
static int measurable_variants(int model) {
    int variants[8];
    int n = lembed::detail::quant_auto_collect_variants((lembed_text_model_t)model,
                                                        variants, 8);
    int found = 0;
    for (int i = 0; i < n; i++) {
        char* dir = NULL;
        if (lembed_ensure_text_model((lembed_text_model_t)variants[i], NULL, 0, 1,
                                     &dir) == LEMBED_OK) {
            free(dir);
            found++;
        }
    }
    return found;
}

/* dry_run is what lets a caller measure without leaving state behind; if it
 * cached anyway, "measure only" would be a lie.
 *
 * The branch sits in the entry point behind a successful measurement, so this
 * needs the weights. The previous version called quant_auto_load() on a path it
 * had never written, which passed whether or not dry_run was honoured -- and it
 * would equally have passed with the save removed. So the contrast is asserted
 * here: dry_run=1 leaves nothing, dry_run=0 does write. Without the second half
 * the first proves nothing.
 *
 * Skipped on a machine that does not already have the variants cached. */
static void test_dry_run_does_not_cache(void) {
    if (measurable_variants(LEMBED_TEXT_BGE_SMALL_EN_V15) < 2) return;

    lembed_model_info_t info;
    if (lembed_get_text_model_info(LEMBED_TEXT_BGE_SMALL_EN_V15, &info) != LEMBED_OK)
        return;

    auto id = lembed::detail::quant_auto_identity(info.model_code);
    std::string path = lembed::detail::quant_auto_path(id);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    /* --- dry run: measure, report, write nothing --- */
    lembed_quantization_choice_t dry;
    memset(&dry, 0, sizeof(dry));
    CHECK(lembed_quantization_auto_select(LEMBED_TEXT_BGE_SMALL_EN_V15, 0, 8, 0,
                                          /*dry_run=*/1, &dry) == LEMBED_OK,
          "a dry run succeeds");
    CHECK(dry.num_measured > 0, "a dry run measures at least one variant");
    CHECK(dry.from_cache == 0, "a dry run is not served from the cache");
    CHECK(dry.reason[0] != '\0', "a dry run reports a reason");

    lembed_quantization_choice_t probe;
    memset(&probe, 0, sizeof(probe));
    CHECK(!lembed::detail::quant_auto_load(id, &probe),
          "dry_run left no cache entry behind");

    /* --- same call without dry_run: now it must write, or the check above is
     *     vacuous --- */
    lembed_quantization_choice_t wet;
    memset(&wet, 0, sizeof(wet));
    CHECK(lembed_quantization_auto_select(LEMBED_TEXT_BGE_SMALL_EN_V15, 0, 8, 0,
                                          /*dry_run=*/0, &wet) == LEMBED_OK,
          "the persisting run succeeds");

    lembed_quantization_choice_t probe2;
    memset(&probe2, 0, sizeof(probe2));
    CHECK(lembed::detail::quant_auto_load(id, &probe2),
          "without dry_run the decision is cached, so the check above has teeth");
    if (probe2.from_cache) {
        CHECK(probe2.quantization == wet.quantization,
              "the cached decision is the one that was just measured");
    }

    std::filesystem::remove(path, ec);
}

/* ==========================================================================
 * A truncated cache entry must not be trusted
 * ========================================================================== */

/* write_file_atomic makes a partial file unlikely, but a full disk or a
 * hand-edited file is not impossible, and the reader used to accept a slot that
 * carried a throughput next to three zeros -- the half-filled choice the loader
 * promises never to return. */
static void test_partial_entry_is_rejected(void) {
    auto id = lembed::detail::quant_auto_identity("test-quant-partial");
    std::string path = lembed::detail::quant_auto_path(id);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    /* Slot 0 complete, slot 1 missing its latency, file size and quantization --
     * i.e. the file stopped after q1_docs_per_sec. */
    std::ostringstream os;
    os << "{\n";
    lembed::detail::write_cache_identity(os, id);
    os << "  \"quantization\": 1,\n";
    os << "  \"num_measured\": 2,\n";
    os << "  \"reason\": \"truncated\",\n";
    os << "  \"q0_quantization\": 0,\n";
    os << "  \"q0_docs_per_sec\": 100,\n";
    os << "  \"q0_latency_ms\": 10,\n";
    os << "  \"q0_file_mb\": 90,\n";
    os << "  \"q1_docs_per_sec\": 150\n";
    os << "}\n";
    lembed::detail::write_file_atomic(path, os.str());

    lembed_quantization_choice_t c;
    memset(&c, 0, sizeof(c));
    CHECK(!lembed::detail::quant_auto_load(id, &c),
          "a slot missing three of its four fields is not served");
    CHECK(c.num_measured == 0 && c.from_cache == 0,
          "a rejected entry leaves no half-filled table behind");

    std::filesystem::remove(path, ec);
}

/* A gap in the table is as suspicious as a truncated tail: slot 1 complete with
 * slot 0 empty is not something the writer can produce. */
static void test_entry_with_a_gap_is_rejected(void) {
    auto id = lembed::detail::quant_auto_identity("test-quant-gap");
    std::string path = lembed::detail::quant_auto_path(id);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    std::ostringstream os;
    os << "{\n";
    lembed::detail::write_cache_identity(os, id);
    os << "  \"quantization\": 1,\n";
    os << "  \"num_measured\": 2,\n";
    os << "  \"reason\": \"gap\",\n";
    os << "  \"q1_quantization\": 1,\n";
    os << "  \"q1_docs_per_sec\": 150,\n";
    os << "  \"q1_latency_ms\": 6.7,\n";
    os << "  \"q1_file_mb\": 23\n";
    os << "}\n";
    lembed::detail::write_file_atomic(path, os.str());

    lembed_quantization_choice_t c;
    memset(&c, 0, sizeof(c));
    CHECK(!lembed::detail::quant_auto_load(id, &c),
          "a table whose slots do not start at 0 is not served");

    std::filesystem::remove(path, ec);
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

/* ==========================================================================
 * Cache invalidation reaches the quantization sub-directory
 * ========================================================================== */

/*lembed_autotune_clear_cache(model) resolves the name and then sweeps. The
 * sweep used to iterate the cache root only, and the quantization decisions live
 * one level down in <root>/quantization -- so clearing a model's tuning left its
 * recorded quantization decision in place, with no way to invalidate it short of
 * deleting the whole cache by hand.
 *
 * This writes a real entry through quant_auto_save(), so the file carries the
 * identity fields the sweeper looks for rather than a hand-rolled approximation. */
static void test_clear_cache_reaches_quantization_subdir(void) {
    lembed_model_info_t info;
    if (lembed_get_text_model_info(LEMBED_TEXT_BGE_SMALL_EN_V15, &info) != LEMBED_OK) {
        return;
    }

    auto id = lembed::detail::quant_auto_identity(info.model_code);
    std::string path = lembed::detail::quant_auto_path(id);
    std::error_code ec;

    /* Somewhere under <root>/quantization, not in the root itself: that is the
     * whole point of the test. */
    CHECK(path.find("quantization") != std::string::npos,
          "the decision lives in the quantization sub-directory");

    /* Never clobber a real recorded decision: remember it and put it back. */
    bool had_entry = std::filesystem::exists(path, ec);
    std::string saved;
    if (had_entry) {
        std::ifstream in(path, std::ios::binary);
        saved.assign(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
    }

    lembed_quantization_choice_t c;
    memset(&c, 0, sizeof(c));
    c.quantization = LEMBED_QUANTIZATION_DYNAMIC;
    c.num_measured = 1;
    c.measured[0].quantization = LEMBED_QUANTIZATION_NONE;
    c.measured[0].docs_per_sec = 100.0;
    c.measured[0].latency_ms = 10.0;
    c.measured[0].file_mb = 90.0;
    snprintf(c.reason, sizeof(c.reason), "sentinel");
    lembed::detail::quant_auto_save(id, c);

    lembed::detail::clear_autotune_cache(info.model_code);

    CHECK(!std::filesystem::exists(path, ec),
          "clear_autotune_cache(model) removes the recorded quantization decision");

    if (had_entry) {
        lembed::detail::write_file_atomic(path, saved);
    }
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

    test_variant_discovery_baselines_on_fp32();
    test_variant_discovery_keeps_baseline_from_quantized_request();
    test_variant_discovery_honours_max_out();
    test_reason_is_populated();
    test_reason_with_json_metacharacters();
    test_cache_round_trip();
    test_partial_entry_is_rejected();
    test_entry_with_a_gap_is_rejected();
    test_dry_run_does_not_cache();
    test_clear_cache_reaches_quantization_subdir();
    test_external_data_is_counted();
    test_probe_handles_missing_session();
    test_probe_is_bounded();

    printf("%s: %d passed, %d failed\n",
           g_fail == 0 ? "PASS" : "FAIL", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}