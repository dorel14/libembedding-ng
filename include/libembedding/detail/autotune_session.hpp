/*
 * libembedding - detail/autotune_session.hpp
 * Session reuse across grid points that differ only by per-call parameters.
 *
 * Creating an ORT session costs seconds (graph load, optimisation, weight
 * pre-packing: ~6 s for a small reranker on a 4-core laptop), while the batch
 * size of the reranker, sparse and image APIs is a per-call argument. Grid
 * points that share every session-level setting can therefore share one
 * session. The tuners order their grids so those points are consecutive, and
 * this cache keeps the current session alive until the key changes.
 *
 * Memory accounting: the resident set is sampled before a session is created
 * and that sample is handed to every grid point using the session, so each
 * point reports "session + activations so far". ORT arenas only grow, which
 * makes the figure an upper-bound estimate for later points; it is used only by
 * the MEMORY objective.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_SESSION_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_SESSION_HPP

#include "autotune_measure.hpp"

namespace lembed { namespace detail {

/* Holds at most one session of type Handle, identified by Key (must support
 * operator==). Not thread-safe: one cache per tuning run. */
template <class Key, class Handle>
class autotune_session_cache {
public:
    using deleter_t = void (*)(Handle*);

    explicit autotune_session_cache(deleter_t deleter) : deleter_(deleter) {}
    ~autotune_session_cache() { reset(); }

    autotune_session_cache(const autotune_session_cache&) = delete;
    autotune_session_cache& operator=(const autotune_session_cache&) = delete;

    /* Session for `key`: the current one when the key matches, otherwise a new
     * one from `create()` (returns Handle* or nullptr on failure). */
    template <class Create>
    Handle* get(const Key& key, Create&& create) {
        if (handle_ && key == key_) return handle_;
        reset();
        rss_before_mb_ = autotune_current_rss_mb();
        handle_ = create();
        if (handle_) key_ = key;
        return handle_;
    }

    /* Resident set sampled before the current session was created. */
    double rss_before_mb() const { return rss_before_mb_; }

    void reset() {
        if (handle_ && deleter_) deleter_(handle_);
        handle_ = nullptr;
    }

private:
    deleter_t deleter_;
    Handle* handle_ = nullptr;
    Key key_{};
    double rss_before_mb_ = 0.0;
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_SESSION_HPP */
