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

/* Returns 1 and fills *out_vec / *dim on hit, 0 on miss.
 * *out_vec is a BORROWED pointer owned by the cache: it stays valid until the
 * entry is overwritten by lembed_cache_put(), evicted by LRU pressure,
 * cleared by lembed_cache_clear(), or released by lembed_cache_free().
 * The caller must copy the data it needs and must never free() that pointer
 * (doing so would be a double free at the next eviction).
 *
 * Because the cache keeps no lock once this function returns, a concurrent
 * put/eviction/clear on the same key can free the buffer while the caller
 * copies it. Prefer lembed_cache_get_copy(), which copies under the lock.
 *
 * DEPRECATED: use lembed_cache_get_copy(), which has no such hazard.
 * Kept for ABI compatibility with existing C callers. */
int lembed_cache_get(lembed_cache_t* cache, const char* text, float** out_vec, int* dim);

/* Copies the cached embedding into caller-owned storage, under the cache lock.
 * No cache-owned pointer escapes, so the copy is safe against a concurrent
 * eviction, overwrite or clear(). The caller keeps ownership of out_vec.
 *
 * `capacity` is the number of floats out_vec can hold. On a hit, *out_dim
 * always receives the stored dimension, even when the copy does not happen,
 * so a caller that does not know the dimension can query it by passing
 * capacity 0 and a NULL out_vec.
 *
 * Return value:
 *    1  hit, `capacity` was large enough and out_vec holds the embedding
 *    0  miss (absent key or expired entry), or invalid argument
 *   -1  hit, but `capacity` is too small (or out_vec is NULL): *out_dim holds
 *       the required size and out_vec is left untouched
 *
 * Thread-safe with respect to the other cache functions. */
int lembed_cache_get_copy(lembed_cache_t* cache, const char* text, float* out_vec,
                          int capacity, int* out_dim);
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

int lembed_cache_get_copy(lembed_cache_t* cache, const char* text, float* out_vec,
                          int capacity, int* out_dim) {
    if (!cache || !text || !out_dim) return 0;
    int dim = 0;
    int r = cache->cache.get_copy(text, out_vec, capacity, &dim);
    if (r == 0) return 0;
    *out_dim = dim;
    return r;
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
