/*
 * libembedding - detail/autotune_search.hpp
 * Generic grid search shared by every auto-tuner.
 *
 * A tuner is reduced to three task-specific pieces:
 *   - a Config struct and the list of configurations to try (the grid);
 *   - a bench callable: Config -> autotune_measurement (see autotune_measure.hpp);
 *   - a describe callable: Config -> short text for the progress log.
 * Everything else -- running the grid, logging, applying constraints, choosing
 * with the shared objective -- lives here, once.
 *
 * The whole grid is measured before choosing because the objective costs are
 * normalised by the best value on the grid (autotune_objective.hpp). The grids
 * are small (a few to a few dozen points), so early stopping would save little
 * and make the choice depend on the visiting order.
 *
 * Auteur: David Orel
 * Version: 1.12.2
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_SEARCH_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_SEARCH_HPP

#include "autotune_measure.hpp"
#include "autotune_objective.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Outcome of a grid search. `best` indexes `grid`/`measurements`, -1 when
 * nothing usable was measured. `constraints_met` is false when the best entry
 * comes from the unconstrained fallback (see autotune_search_options). */
template <class Config>
struct autotune_search_result {
    std::vector<Config> grid;
    std::vector<autotune_measurement> measurements;
    int best = -1;
    int feasible = 0;
    bool constraints_met = true;

    bool found() const { return best >= 0; }
    const Config& best_config() const { return grid[(size_t)best]; }
    const autotune_measurement& best_measurement() const { return measurements[(size_t)best]; }
};

/* Search options. `label` prefixes every progress line ("reranker_autotune",
 * ...) so logs stay attributable; `verbose` = false silences them (tests).
 *
 * `relax_if_infeasible`: when configurations ran but none met the
 * constraints, choose among all of them as if unconstrained, and report it
 * through constraints_met = false. This is the public API's documented
 * behaviour (the caller can compare p95 with its budget). It re-uses the
 * measurements already taken instead of benchmarking the grid a second time. */
struct autotune_search_options {
    lembed_objective_t objective = LEMBED_OBJECTIVE_BALANCED;
    autotune_constraints constraints;
    bool relax_if_infeasible = true;
    autotune_protocol protocol;          /* see autotune_protocol_for() */
    double prune_factor = 2.5;           /* racing threshold, x best P50; 0 = off */
    double tie_tolerance = 0.05;         /* costs within 5% are a tie -> earliest point */
    const char* label = "autotune";
    bool verbose = true;
};

/* `bench(cfg, protocol)` measures one configuration with the given protocol.
 *
 * Racing: once a configuration has been measured, later ones stop after two
 * timed calls if both are slower than prune_factor x the best P50 so far.
 * Every timed call processes the same workload, so such a configuration loses
 * on latency and throughput alike. Racing is disabled for MEMORY, where a slow
 * configuration can still win. */
template <class Config, class Bench, class Describe>
autotune_search_result<Config> autotune_grid_search(const std::vector<Config>& grid,
                                                    Bench&& bench,
                                                    Describe&& describe,
                                                    const autotune_search_options& opts) {
    autotune_search_result<Config> res;
    res.grid = grid;
    res.measurements.reserve(grid.size());

    if (opts.verbose)
        std::fprintf(stderr, "%s: testing %d configurations (objective=%s)...\n", opts.label,
                     (int)grid.size(), autotune_objective_name(opts.objective));

    const bool racing = opts.prune_factor > 0.0 && opts.objective != LEMBED_OBJECTIVE_MEMORY;
    double best_p50 = 0.0;
    for (const Config& cfg : grid) {
        autotune_protocol proto = opts.protocol;
        proto.prune_above_ms = (racing && best_p50 > 0.0) ? best_p50 * opts.prune_factor : 0.0;
        autotune_measurement m = bench(cfg, proto);
        if (m.valid && (best_p50 <= 0.0 || m.p50_ms < best_p50)) best_p50 = m.p50_ms;
        if (opts.verbose) {
            std::string what = describe(cfg);
            if (m.valid)
                std::fprintf(stderr, "  %s: %s -> %.1f items/s (P50=%.1fms P95=%.1fms, +%.0fMB)\n",
                             opts.label, what.c_str(), m.throughput, m.p50_ms, m.p95_ms,
                             m.memory_mb);
            else
                std::fprintf(stderr, "  %s: %s -> FAILED\n", opts.label, what.c_str());
        }
        if (autotune_satisfies(m, opts.constraints)) res.feasible++;
        res.measurements.push_back(m);
    }

    res.best = autotune_select_best(res.measurements, opts.objective, opts.constraints,
                                    opts.tie_tolerance);
    if (!res.found() && opts.relax_if_infeasible) {
        res.best = autotune_select_best(res.measurements, opts.objective, autotune_constraints(),
                                        opts.tie_tolerance);
        if (res.found()) {
            res.constraints_met = false;
            if (opts.verbose)
                std::fprintf(stderr, "%s: no config satisfies constraints, falling back...\n",
                             opts.label);
        }
    }

    if (opts.verbose) {
        if (res.found())
            std::fprintf(stderr, "%s: best config: %s (%.1f items/s, P50=%.1fms, P95=%.1fms)\n",
                         opts.label, describe(res.best_config()).c_str(),
                         res.best_measurement().throughput, res.best_measurement().p50_ms,
                         res.best_measurement().p95_ms);
        else
            std::fprintf(stderr, "%s: no feasible configuration (%d tried)\n", opts.label,
                         (int)grid.size());
    }
    return res;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_SEARCH_HPP */
