/*
 * libembedding - detail/batch.hpp
 * Batch iteration utility
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#if defined(_WIN32) || defined(WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifndef LIBEMBEDDING_DETAIL_BATCH_HPP
#define LIBEMBEDDING_DETAIL_BATCH_HPP

#include <algorithm>

namespace lembed { namespace detail {

/* Generate batch ranges: [start, end) for each batch */
struct BatchRange {
    int start;
    int end;
};

inline int batch_count(int total, int batch_size) {
    if (batch_size <= 0) batch_size = total;
    return (total + batch_size - 1) / batch_size;
}

inline BatchRange get_batch(int index, int total, int batch_size) {
    if (batch_size <= 0) batch_size = total;
    BatchRange r;
    r.start = index * batch_size;
    r.end = std::min(r.start + batch_size, total);
    return r;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_BATCH_HPP */




