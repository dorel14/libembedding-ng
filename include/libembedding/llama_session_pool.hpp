/*
 * libembedding - llama_session_pool.hpp
 * Public header for LlamaSessionPool (llama.cpp backend)
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_LLAMA_SESSION_POOL_HPP
#define LIBEMBEDDING_LLAMA_SESSION_POOL_HPP

#include "detail/llama_session_impl.hpp"

namespace lembed {

/* Public alias for the llama.cpp session pool. */
using LlamaSessionPool = detail::LlamaSessionPool;

} /* namespace lembed */

#endif /* LIBEMBEDDING_LLAMA_SESSION_POOL_HPP */




