/*
 * libembedding - Single umbrella include
 *
 * Usage:
 *   // In exactly ONE .cpp file in your project:
 *   #define LIBEMBEDDING_IMPLEMENTATION
 *   #include <libembedding/libembedding.h>
 *
 *   // In all other files:
 *   #include <libembedding/libembedding.h>
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_H
#define LIBEMBEDDING_H

#include "detail/win_psapi_init.hpp"
#include "config.h"
#include "error.h"
#include "types.h"
#include "model_registry.h"
#include "downloader.h"
#include "text_embedding.h"          // Must come before benchmark (defines lembed_text_embedding_t)
#include "sparse_text_embedding.h"
#include "autotuner.h"

#ifndef LIBEMBEDDING_NO_IMAGE
#include "image_embedding.h"
#endif

#include "reranker.h"
#include "similarity.h"
#include "embedding_benchmark.h"     // Uses types from text_embedding.h
#include "gguf_registry.h"
#include "autotune_cache.h"
#include "unified_benchmark.h"
#include "worker_autotune.h"
#include "embedding_mode.h"
#include "embedding_cache.h"

#endif /* LIBEMBEDDING_H */




