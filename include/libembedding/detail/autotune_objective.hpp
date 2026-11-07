/*
 * libembedding - detail/autotune_objective.hpp
 * Objective scoring and constraints shared by every auto-tuner.
 *
 * Why normalised costs: the tuners used to rank configurations with formulas
 * that added docs/s to milliseconds (or inverted one of them with an arbitrary
 * constant), so the weight of each term depended on the model's speed. Here
 * every term is divided by the best value observed on the feasible grid, which
 * makes each term dimensionless and equal to 1.0 for the best configuration on
 * that axis. A cost is "how many times worse than the best", lower is better.
 *
 *   LATENCY     tail / tail_min,           tail = P50 + 0.5 * (P95 - P50)
 *   THROUGHPUT  thr_max / thr
 *   BALANCED    0.5 * latency cost + 0.5 * throughput cost
 *   MEMORY      (mem + floor) / (mem_min + floor) + 0.1 * latency cost
 *
 * The tail term keeps the latency objective honest about jitter without being
 * dominated by one outlier. The memory floor (16 MB) stops a configuration that
 * happened to measure ~0 MB of growth from making every other cost explode; the
 * small latency term breaks ties between configurations of equal footprint.
 *
 * Constraints are hard filters applied before normalisation, so an infeasible
 * configuration can neither win nor distort the reference values.
 *
 * Auteur: David Orel
 * Version: 1.12.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_OBJECTIVE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_OBJECTIVE_HPP

#include "libembedding/autotuner.h"
#include "autotune_measure.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace lembed { namespace detail {

/* Hard limits a configuration must meet to be eligible. 0 disables a limit. */
struct autotune_constraints {
    double max_p95_ms = 0.0;   /* request tail latency budget */
};

inline bool autotune_satisfies(const autotune_measurement& m, const autotune_constraints& c) {
    if (!m.valid) return false;
    if (c.max_p95_ms > 0.0 && m.p95_ms > c.max_p95_ms) return false;
    return true;
}

/* Tail-aware latency of one call. */
inline double autotune_tail_ms(const autotune_measurement& m) {
    return m.p50_ms + 0.5 * (m.p95_ms - m.p50_ms);
}

/* Best values over the feasible set, used to normalise every cost. */
struct autotune_reference {
    double tail_min = 0.0;
    double thr_max = 0.0;
    double mem_min = 0.0;
};

inline autotune_reference autotune_make_reference(const std::vector<autotune_measurement>& ms,
                                                  const autotune_constraints& c) {
    autotune_reference ref;
    bool first = true;
    for (const auto& m : ms) {
        if (!autotune_satisfies(m, c)) continue;
        double tail = autotune_tail_ms(m);
        if (first) {
            ref.tail_min = tail;
            ref.thr_max = m.throughput;
            ref.mem_min = m.memory_mb;
            first = false;
        } else {
            ref.tail_min = std::min(ref.tail_min, tail);
            ref.thr_max = std::max(ref.thr_max, m.throughput);
            ref.mem_min = std::min(ref.mem_min, m.memory_mb);
        }
    }
    return ref;
}

/* Cost given to an axis that cannot be computed (zero throughput). Finite on
 * purpose: the library is built with -ffast-math / /fp:fast, under which the
 * compiler may assume no infinity exists and fold comparisons with one, which
 * broke selection on macOS (Clang) in both directions. */
constexpr double kAutotuneWorstCost = 1.0e30;

/* Dimensionless cost of a feasible measurement, lower is better. */
inline double autotune_cost(const autotune_measurement& m, const autotune_reference& ref,
                            lembed_objective_t objective) {
    const double kMemFloorMb = 16.0;
    double tail = autotune_tail_ms(m);
    double lat_cost = (ref.tail_min > 0.0 && tail > 0.0) ? tail / ref.tail_min : 1.0;
    double thr_cost = (m.throughput > 0.0 && ref.thr_max > 0.0) ? ref.thr_max / m.throughput
                                                                : kAutotuneWorstCost;
    switch (objective) {
        case LEMBED_OBJECTIVE_LATENCY:
            return lat_cost;
        case LEMBED_OBJECTIVE_THROUGHPUT:
            return thr_cost;
        case LEMBED_OBJECTIVE_MEMORY:
            return (m.memory_mb + kMemFloorMb) / (ref.mem_min + kMemFloorMb) + 0.1 * lat_cost;
        case LEMBED_OBJECTIVE_BALANCED:
        default:
            return 0.5 * lat_cost + 0.5 * thr_cost;
    }
}

/* Index of the best feasible measurement, or -1 when none is feasible.
 *
 * `tie_tolerance` (e.g. 0.05) treats every cost within that fraction of the
 * minimum as a tie, and ties go to the earliest index. Grids list their
 * preferred configurations first (fewer sessions, less truncation, smaller
 * batches), so a difference that is within measurement noise no longer flips
 * the choice from one run to the next. */
inline int autotune_select_best(const std::vector<autotune_measurement>& ms,
                                lembed_objective_t objective,
                                const autotune_constraints& c,
                                double tie_tolerance = 0.0) {
    /* Feasibility is tracked explicitly, never through an infinite sentinel
     * (see kAutotuneWorstCost). */
    autotune_reference ref = autotune_make_reference(ms, c);
    std::vector<double> costs(ms.size(), 0.0);
    std::vector<char> feasible(ms.size(), 0);
    int best = -1;
    double best_cost = 0.0;
    for (size_t i = 0; i < ms.size(); i++) {
        if (!autotune_satisfies(ms[i], c)) continue;
        feasible[i] = 1;
        costs[i] = autotune_cost(ms[i], ref, objective);
        if (best < 0 || costs[i] < best_cost) {
            best = (int)i;
            best_cost = costs[i];
        }
    }
    if (best < 0) return -1;
    double limit = best_cost * (1.0 + std::max(0.0, tie_tolerance));
    for (size_t i = 0; i < ms.size(); i++)
        if (feasible[i] && costs[i] <= limit) return (int)i;
    return best;
}

inline const char* autotune_objective_name(lembed_objective_t objective) {
    switch (objective) {
        case LEMBED_OBJECTIVE_LATENCY:    return "latency";
        case LEMBED_OBJECTIVE_THROUGHPUT: return "throughput";
        case LEMBED_OBJECTIVE_BALANCED:   return "balanced";
        case LEMBED_OBJECTIVE_MEMORY:     return "memory";
        default:                          return "unknown";
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_OBJECTIVE_HPP */
