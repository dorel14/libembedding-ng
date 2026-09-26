/*
 * libembedding - detail/status_helper.hpp
 * Common error handling helpers for C++ wrappers
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_STATUS_HELPER_HPP
#define LIBEMBEDDING_DETAIL_STATUS_HELPER_HPP

#include "libembedding/types.h"

#include <stdexcept>
#include <string>

namespace lembed { namespace detail {

/* Throw if status != LEMBED_OK */
inline void check_status(lembed_status_t status) {
    if (status != LEMBED_OK) {
        throw std::runtime_error(
            std::string("libembedding error: ") + lembed_last_error());
    }
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_STATUS_HELPER_HPP */




