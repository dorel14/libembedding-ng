/*
 * libembedding - embedding_cache.h
 * LRU cache for embeddings
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_EMBEDDING_CACHE_H
#define LIBEMBEDDING_EMBEDDING_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t capacity;
    size_t current_size;
    int ttl_seconds;
} lembed_cache_config_t;

lembed_cache_config_t lembed_cache_config_default(void);

typedef struct lembed_cache_t lembed_cache_t;

lembed_cache_t* lembed_cache_create(const lembed_cache_config_t* config);
void lembed_cache_free(lembed_cache_t* cache);
void lembed_cache_clear(lembed_cache_t* cache);
int lembed_cache_get(lembed_cache_t* cache, const char* text, float** out_vec, int* dim);
void lembed_cache_put(lembed_cache_t* cache, const char* text, const float* vec, int dim);
size_t lembed_cache_size(const lembed_cache_t* cache);
size_t lembed_cache_capacity(const lembed_cache_t* cache);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#if defined(__cplusplus)
extern "C++" {
#endif
#include "detail/embedding_cache_impl.hpp"
#if defined(__cplusplus)
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <string.h>

struct lembed_cache_t {
    lembed::detail::LRUCache cache;
};

lembed_cache_config_t lembed_cache_config_default(void) {
    lembed_cache_config_t cfg;
    cfg.capacity = 4096;
    cfg.current_size = 0;
    cfg.ttl_seconds = 0;
    return cfg;
}

lembed_cache_t* lembed_cache_create(const lembed_cache_config_t* config) {
    if (!config || config->capacity == 0) return NULL;
    lembed_cache_t* cache = (lembed_cache_t*)malloc(sizeof(lembed_cache_t));
    if (!cache) return NULL;
    new (&cache->cache) lembed::detail::LRUCache(config->capacity, config->ttl_seconds);
    return cache;
}

void lembed_cache_free(lembed_cache_t* cache) {
    if (!cache) return;
    cache->cache.~LRUCache();
    free(cache);
}

void lembed_cache_clear(lembed_cache_t* cache) {
    if (!cache) return;
    cache->cache.clear();
}

int lembed_cache_get(lembed_cache_t* cache, const char* text, float** out_vec, int* dim) {
    if (!cache || !text || !out_vec || !dim) return 0;
    return cache->cache.get(text, out_vec, dim) ? 1 : 0;
}

void lembed_cache_put(lembed_cache_t* cache, const char* text, const float* vec, int dim) {
    if (!cache || !text || !vec) return;
    cache->cache.put(text, vec, dim);
}

size_t lembed_cache_size(const lembed_cache_t* cache) {
    return cache ? cache->cache.size() : 0;
}

size_t lembed_cache_capacity(const lembed_cache_t* cache) {
    return cache ? cache->cache.capacity() : 0;
}

#ifdef __cplusplus
}
#endif
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_EMBEDDING_CACHE_H */
