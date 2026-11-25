/*
 * libembedding - detail/log.hpp
 * Opt-in diagnostic output. The library prints nothing on stderr unless the
 * user asks for it: LIBEMBEDDING_VERBOSE=1 (or true/yes/on). Errors go
 * through lembed_last_error(), never through stderr.
 *
 * Download progress bars are not concerned: they have their own switch
 * (show_download_progress / show_progress), and llama.cpp's error-level log
 * lines are kept, as they report a failure the caller is about to see.
 *
 * Auteur: David Orel
 * Version: 1.16.5
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_LOG_HPP
#define LIBEMBEDDING_DETAIL_LOG_HPP

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace lembed { namespace detail {

/* Read once per process: the variable is meant to be set before the program
 * starts, and reading it on every call would cost a getenv per log line. */
inline bool verbose_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("LIBEMBEDDING_VERBOSE");
        if (!value) return false;
        std::string v(value);
        for (char& c : v) c = (char)std::tolower((unsigned char)c);
        return v == "1" || v == "true" || v == "yes" || v == "on";
    }();
    return enabled;
}

/* printf-style line on stderr, only when verbose_enabled(). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline void log_info(const char* fmt, ...) {
    if (!verbose_enabled()) return;
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_LOG_HPP */
