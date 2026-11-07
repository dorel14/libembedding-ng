/*
 * libembedding - tests/test_autotune_core.cpp
 * Unit tests for the shared auto-tuning engine (no model, no network):
 *   - measurement protocol (percentiles, failures, throughput derivation)
 *   - objective costs, constraints and selection
 *   - generic grid search with a deterministic fake benchmark (racing, ties,
 *     constraint fallback)
 *   - session reuse across grid points
 *   - generic cache entries (round trip, identity, partial entry, purge)
 *
 * The cache root is redirected to a temporary directory so the test never
 * touches the user's real tuning results.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/autotuner.h>
#include <libembedding/detail/autotune_cache.hpp>
#include <libembedding/detail/autotune_measure.hpp>
#include <libembedding/detail/autotune_objective.hpp>
#include <libembedding/detail/autotune_search.hpp>
#include <libembedding/detail/autotune_session.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace lembed::detail;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); } \
} while (0)

static bool approx_eq(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

/* Build a measurement directly, as a bench callable would return it. */
static autotune_measurement meas(double p50, double p95, double thr, double mem = 0.0,
                                 bool valid = true) {
    autotune_measurement m;
    m.valid = valid;
    m.p50_ms = p50;
    m.p95_ms = p95;
    m.throughput = thr;
    m.memory_mb = mem;
    m.items_per_call = 1;
    m.calls_ok = valid ? 5 : 0;
    return m;
}

/* -------------------------------------------------------------------------
 * Measurement protocol
 * ------------------------------------------------------------------------- */

static void test_percentiles() {
    std::vector<double> empty;
    CHECK(percentile_sorted(empty, 0.5) == 0.0, "empty sample -> 0");
    std::vector<double> one = {7.0};
    CHECK(percentile_sorted(one, 0.95) == 7.0, "single sample -> itself");
    std::vector<double> four = {1.0, 2.0, 3.0, 4.0};
    CHECK(approx_eq(percentile_sorted(four, 0.5), 2.5), "median interpolates");
    CHECK(approx_eq(percentile_sorted(four, 0.0), 1.0), "p0 is the minimum");
    CHECK(approx_eq(percentile_sorted(four, 1.0), 4.0), "p100 is the maximum");
    CHECK(approx_eq(percentile_sorted(four, 0.95), 3.85), "p95 interpolates");
}

static void test_measure_calls() {
    autotune_protocol proto;
    proto.warmup = 2;
    proto.iters = 6;

    int calls = 0;
    auto all_fail = autotune_measure_calls(proto, 10, -1.0, [&]() { calls++; return false; });
    CHECK(!all_fail.valid, "every call failing -> invalid measurement");
    CHECK(all_fail.calls_ok == 0, "no successful call counted");
    CHECK(calls == 8, "warmup + iters calls are made");

    int n = 0;
    auto some_fail = autotune_measure_calls(proto, 10, -1.0, [&]() {
        volatile double x = 0.0;
        for (int i = 0; i < 20000; i++) x += i * 0.5;
        return (n++ % 2) == 0;
    });
    CHECK(some_fail.calls_ok > 0 && some_fail.calls_ok < proto.iters,
          "failed timed calls are excluded");
    CHECK(some_fail.valid, "a measurement with successes is valid");
    CHECK(some_fail.p95_ms >= some_fail.p50_ms, "p95 >= p50");
    CHECK(approx_eq(some_fail.throughput, 10 * 1000.0 / some_fail.p50_ms, 1e-6),
          "throughput derived from the P50 and items per call");

    auto no_mem = autotune_measure_calls(proto, 1, -1.0, []() { return true; });
    CHECK(no_mem.memory_mb == 0.0, "negative rss_before skips the memory estimate");
    CHECK(autotune_current_rss_mb() > 0.0, "resident set size is readable on this platform");
}

/* -------------------------------------------------------------------------
 * Objectives and constraints
 * ------------------------------------------------------------------------- */

static void test_objectives() {
    /* 0: high throughput, slow requests   (big batches)
     * 1: fast requests, low throughput    (small batches)
     * 2: in between on both axes
     * 3: smallest memory footprint, otherwise mediocre */
    std::vector<autotune_measurement> ms = {
        meas(100.0, 120.0, 1000.0, 400.0),
        meas(10.0, 12.0, 300.0, 300.0),
        meas(30.0, 33.0, 800.0, 350.0),
        meas(60.0, 70.0, 500.0, 50.0),
    };
    autotune_constraints none;
    CHECK(autotune_select_best(ms, LEMBED_OBJECTIVE_THROUGHPUT, none) == 0, "throughput picks 0");
    CHECK(autotune_select_best(ms, LEMBED_OBJECTIVE_LATENCY, none) == 1, "latency picks 1");
    CHECK(autotune_select_best(ms, LEMBED_OBJECTIVE_MEMORY, none) == 3, "memory picks 3");
    CHECK(autotune_select_best(ms, LEMBED_OBJECTIVE_BALANCED, none) == 2,
          "balanced picks the compromise");

    /* Dimensionless: rescaling an axis by a constant must not change a choice. */
    std::vector<autotune_measurement> scaled = ms;
    for (auto& m : scaled) { m.p50_ms *= 1000.0; m.p95_ms *= 1000.0; m.throughput /= 1000.0; }
    CHECK(autotune_select_best(scaled, LEMBED_OBJECTIVE_BALANCED, none) == 2,
          "balanced choice is independent of units");

    /* Constraint filters before normalisation. */
    autotune_constraints budget;
    budget.max_p95_ms = 50.0;
    CHECK(autotune_select_best(ms, LEMBED_OBJECTIVE_THROUGHPUT, budget) == 2,
          "throughput within a 50 ms p95 budget picks 2");
    budget.max_p95_ms = 5.0;
    CHECK(autotune_select_best(ms, LEMBED_OBJECTIVE_THROUGHPUT, budget) == -1,
          "no configuration within budget -> -1");

    /* Invalid measurements never win, even with flattering numbers. */
    std::vector<autotune_measurement> with_invalid = {
        meas(0.001, 0.001, 1e9, 0.0, false),
        meas(10.0, 11.0, 100.0),
    };
    CHECK(autotune_select_best(with_invalid, LEMBED_OBJECTIVE_THROUGHPUT, none) == 1,
          "invalid measurement is skipped");
    std::vector<autotune_measurement> all_invalid = {meas(1, 1, 1, 0, false)};
    CHECK(autotune_select_best(all_invalid, LEMBED_OBJECTIVE_BALANCED, none) == -1,
          "all invalid -> -1");

    /* Ties keep the earliest index. */
    std::vector<autotune_measurement> ties = {meas(10, 10, 100), meas(10, 10, 100)};
    CHECK(autotune_select_best(ties, LEMBED_OBJECTIVE_BALANCED, none) == 0, "tie keeps first");

    /* Within the tolerance, a slightly better later point does not displace the
     * earlier (preferred) one; beyond it, it does. */
    std::vector<autotune_measurement> noisy = {meas(10, 10, 100), meas(10, 10, 103)};
    CHECK(autotune_select_best(noisy, LEMBED_OBJECTIVE_THROUGHPUT, none, 0.0) == 1,
          "no tolerance: 3% better wins");
    CHECK(autotune_select_best(noisy, LEMBED_OBJECTIVE_THROUGHPUT, none, 0.05) == 0,
          "5% tolerance: 3% better is a tie, earliest wins");
    std::vector<autotune_measurement> clear_win = {meas(10, 10, 100), meas(10, 10, 110)};
    CHECK(autotune_select_best(clear_win, LEMBED_OBJECTIVE_THROUGHPUT, none, 0.05) == 1,
          "5% tolerance: 10% better still wins");
}

/* -------------------------------------------------------------------------
 * Session reuse
 * ------------------------------------------------------------------------- */

struct FakeSession {
    int key;
};
static int g_sessions_alive = 0;
static void free_fake_session(FakeSession* s) {
    g_sessions_alive--;
    delete s;
}

static void test_session_cache() {
    int created = 0;
    auto make = [&](int key) {
        return [&created, key]() -> FakeSession* {
            created++;
            g_sessions_alive++;
            return new FakeSession{key};
        };
    };
    {
        autotune_session_cache<int, FakeSession> cache(free_fake_session);
        FakeSession* a = cache.get(4, make(4));
        FakeSession* b = cache.get(4, make(4));
        CHECK(a == b && created == 1, "same key reuses the session");
        FakeSession* c = cache.get(8, make(8));
        CHECK(c != nullptr && c->key == 8 && created == 2, "new key creates a new session");
        CHECK(g_sessions_alive == 1, "previous session freed on key change");
        FakeSession* failed = cache.get(16, []() -> FakeSession* { return nullptr; });
        CHECK(failed == nullptr && g_sessions_alive == 0, "failed creation leaves nothing alive");
        FakeSession* again = cache.get(16, make(16));
        CHECK(again != nullptr && created == 3, "a failed key is retried, not cached");
        CHECK(cache.rss_before_mb() >= 0.0, "rss sampled before creation");
    }
    CHECK(g_sessions_alive == 0, "destructor frees the last session");
}

/* -------------------------------------------------------------------------
 * Grid search
 * ------------------------------------------------------------------------- */

struct FakeConfig {
    int threads;
    int batch;
};

static void test_grid_search() {
    std::vector<FakeConfig> grid = {{1, 4}, {2, 4}, {4, 4}, {4, 16}};
    /* Throughput grows with threads*batch; latency grows with batch. One
     * configuration fails to run. */
    auto bench = [](const FakeConfig& c, const autotune_protocol&) {
        if (c.threads == 2) return meas(0, 0, 0, 0, false);
        double thr = 100.0 * c.threads * c.batch;
        double p50 = 1.0 * c.batch;
        return meas(p50, p50 * 1.1, thr);
    };
    auto describe = [](const FakeConfig& c) {
        return "threads=" + std::to_string(c.threads) + " batch=" + std::to_string(c.batch);
    };

    autotune_search_options o;
    o.verbose = false;
    o.objective = LEMBED_OBJECTIVE_THROUGHPUT;
    auto r = autotune_grid_search(grid, bench, describe, o);
    CHECK(r.found(), "a feasible configuration is found");
    CHECK(r.measurements.size() == grid.size(), "every grid point is measured");
    CHECK(r.feasible == 3, "the failed point is not feasible");
    CHECK(r.best_config().threads == 4 && r.best_config().batch == 16,
          "throughput objective picks the largest threads*batch");

    o.objective = LEMBED_OBJECTIVE_LATENCY;
    r = autotune_grid_search(grid, bench, describe, o);
    CHECK(r.best_config().batch == 4 && r.best_config().threads == 1,
          "latency objective picks the first smallest-batch point");

    o.objective = LEMBED_OBJECTIVE_THROUGHPUT;
    o.constraints.max_p95_ms = 5.0;
    r = autotune_grid_search(grid, bench, describe, o);
    CHECK(r.best_config().threads == 4 && r.best_config().batch == 4,
          "constraint excludes batch 16");

    CHECK(r.constraints_met, "a feasible pick meets the constraints");

    /* Budget nobody meets: fall back to the unconstrained optimum of the same
     * measurements (no second benchmark), and say so. */
    int calls = 0;
    auto counting_bench = [&](const FakeConfig& c, const autotune_protocol& p) {
        calls++;
        return bench(c, p);
    };
    o.constraints.max_p95_ms = 0.5;
    r = autotune_grid_search(grid, counting_bench, describe, o);
    CHECK(r.found() && !r.constraints_met, "infeasible budget -> relaxed pick, flagged");
    CHECK(r.best_config().threads == 4 && r.best_config().batch == 16,
          "relaxed pick is the unconstrained optimum");
    CHECK(calls == (int)grid.size(), "fallback re-uses measurements");

    o.relax_if_infeasible = false;
    r = autotune_grid_search(grid, bench, describe, o);
    CHECK(!r.found(), "no relaxation -> nothing found");

    std::vector<FakeConfig> empty;
    r = autotune_grid_search(empty, bench, describe, o);
    CHECK(!r.found(), "empty grid -> nothing found");

    /* Racing: the first point is measured in full, later points receive a
     * prune threshold of prune_factor x the best P50 so far; MEMORY disables it. */
    std::vector<double> thresholds;
    auto spy = [&](const FakeConfig& c, const autotune_protocol& p) {
        thresholds.push_back(p.prune_above_ms);
        return bench(c, p);
    };
    autotune_search_options race;
    race.verbose = false;
    race.objective = LEMBED_OBJECTIVE_THROUGHPUT;
    race.prune_factor = 2.0;
    autotune_grid_search(grid, spy, describe, race);
    CHECK(thresholds.size() == grid.size() && thresholds[0] == 0.0,
          "first point is never pruned");
    CHECK(approx_eq(thresholds[1], 8.0) && approx_eq(thresholds[3], 8.0),
          "later points get 2 x best P50 (4 ms)");

    thresholds.clear();
    race.objective = LEMBED_OBJECTIVE_MEMORY;
    autotune_grid_search(grid, spy, describe, race);
    CHECK(std::all_of(thresholds.begin(), thresholds.end(), [](double t) { return t == 0.0; }),
          "racing is off for the memory objective");
}

static void test_racing_protocol() {
    autotune_protocol proto;
    proto.warmup = 0;
    proto.iters = 5;
    proto.prune_above_ms = 1e-6; /* any real call is slower */
    int calls = 0;
    auto m = autotune_measure_calls(proto, 1, -1.0, [&]() {
        calls++;
        volatile double x = 0.0;
        for (int i = 0; i < 20000; i++) x += i;
        return true;
    });
    CHECK(calls == 2 && m.calls_ok == 2, "a pruned configuration stops after two timed calls");
    CHECK(m.valid, "a pruned measurement is still valid");
}

/* -------------------------------------------------------------------------
 * Generic cache entries
 * ------------------------------------------------------------------------- */

static void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static autotune_cache_identity ident(const char* corpus) {
    autotune_cache_identity id;
    id.model = "test/core-model";
    id.variant = "default";
    id.corpus = corpus;
    id.objective = (int)LEMBED_OBJECTIVE_BALANCED;
    id.mode = (int)LEMBED_AUTOTUNE_QUICK;
    return id;
}

static void test_cache_entries() {
    namespace fs = std::filesystem;
    fs::path root = fs::temp_directory_path() / "lembed_test_autotune_core";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    set_env("LOCALAPPDATA", root.string());
    set_env("HOME", root.string());

    int threads = 4, batch = 32;
    float weight = 0.25f;
    double thr = 123.5;
    std::vector<cache_field> out_fields = {
        cache_int("threads", threads), cache_int("batch_size", batch),
        cache_float("min_weight", weight), cache_double("throughput_docs_sec", thr)};
    autotune_cache_identity id = ident("abc");
    CHECK(write_cache_entry("sparse", id, out_fields), "entry is written");

    int r_threads = -1, r_batch = -1;
    float r_weight = -1.0f;
    double r_thr = -1.0;
    std::vector<cache_field> in_fields = {
        cache_int("threads", r_threads), cache_int("batch_size", r_batch),
        cache_float("min_weight", r_weight), cache_double("throughput_docs_sec", r_thr)};
    CHECK(read_cache_entry("sparse", id, in_fields), "entry is read back");
    CHECK(r_threads == 4 && r_batch == 32, "int fields round-trip");
    CHECK(approx_eq(r_weight, 0.25, 1e-6) && approx_eq(r_thr, 123.5, 1e-9), "float/double round-trip");

    /* Another corpus is another identity: a miss, never the first corpus' answer. */
    r_threads = -1;
    CHECK(!read_cache_entry("sparse", ident("other"), in_fields), "foreign corpus -> miss");
    CHECK(r_threads == -1, "outputs untouched on a miss");

    /* Same identity in another sub-directory is a different entry. */
    CHECK(!read_cache_entry("image", id, in_fields), "other task sub-directory -> miss");

    /* A partial entry (field removed) is a miss and leaves outputs untouched. */
    std::string path = cache_entry_path("sparse", id);
    {
        std::ifstream f(path);
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();
        size_t pos = content.find("\"batch_size\"");
        size_t eol = content.find('\n', pos);
        content.erase(pos, eol - pos + 1);
        std::ofstream w(path, std::ios::trunc);
        w << content;
    }
    r_threads = -1;
    CHECK(!read_cache_entry("sparse", id, in_fields), "partial entry -> miss");
    CHECK(r_threads == -1, "partial entry leaves outputs untouched");

    /* Purge by model reaches task sub-directories. */
    CHECK(write_cache_entry("sparse", id, out_fields), "entry rewritten");
    CHECK(write_cache_entry("image", id, out_fields), "image entry written");
    clear_autotune_cache("test/core-model");
    CHECK(!fs::exists(cache_entry_path("sparse", id)), "per-model purge removes sparse entry");
    CHECK(!fs::exists(cache_entry_path("image", id)), "per-model purge removes image entry");

    /* Root-only purge (task-scoped "all embedding entries") spares sub-directories. */
    CHECK(write_cache_entry("", id, out_fields), "root entry written");
    CHECK(write_cache_entry("sparse", id, out_fields), "sparse entry written again");
    clear_autotune_cache_root_entries();
    CHECK(!fs::exists(cache_entry_path("", id)), "root purge removes root entries");
    CHECK(fs::exists(cache_entry_path("sparse", id)), "root purge keeps sparse entries");

    fs::remove_all(root, ec);
}

int main() {
    test_percentiles();
    test_measure_calls();
    test_objectives();
    test_grid_search();
    test_racing_protocol();
    test_session_cache();
    test_cache_entries();
    std::printf("test_autotune_core: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
