/*
 * libembedding - detail/enum_check_autotune.hpp
 * Range validation of the autotune enums that come from the caller.
 * Kept apart from enum_check.hpp: autotuner.h pulls the implementation chain
 * in, which model_registry.h (an includer of enum_check.hpp) must not.
 *
 * Auteur: David Orel
 * Version: 1.16.4
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_ENUM_CHECK_AUTOTUNE_HPP
#define LIBEMBEDDING_ENUM_CHECK_AUTOTUNE_HPP

#include "../autotuner.h"
#include "enum_check.hpp"

extern "C++" {
namespace lembed { namespace detail {

inline lembed_status_t check_autotune_mode(const lembed_autotune_mode_t& v) {
    return check_enum(v, LEMBED_AUTOTUNE_QUICK, LEMBED_AUTOTUNE_FULL, "autotune mode");
}
inline lembed_status_t check_objective(const lembed_objective_t& v) {
    return check_enum(v, LEMBED_OBJECTIVE_LATENCY, LEMBED_OBJECTIVE_MEMORY, "autotune objective");
}

}} /* namespace lembed::detail */
}

#endif /* LIBEMBEDDING_ENUM_CHECK_AUTOTUNE_HPP */
