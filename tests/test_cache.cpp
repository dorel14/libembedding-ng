/*
 * test_cache.cpp - LRU embedding cache, C API level.
 *
 * Focus: lembed_cache_get_copy(), the race-free read path. The value is copied
 * into caller-owned storage while the cache holds its lock, so a value read
 * stays valid after the entry is overwritten, evicted or cleared, and reading
 * the same key twice can never double free.
 *
 * The borrowed-pointer entry point lembed_cache_get() is exercised too: its
 * contract is that the caller must not free the pointer it returns.
 *
 * No model downloads required.
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

/* Required on Linux/macOS, where libembedding is a header-only INTERFACE
 * target: exactly one translation unit must define this to emit the code. */
#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cstdio>
#include <cstring>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        failures++; \
    } else { \
        passes++; \
    } \
} while (0)

static void fill(float* dst, int dim, int seed) {
    for (int i = 0; i < dim; ++i) dst[i] = (float)(seed + i);
}

static bool equals(const float* a, int dim, int seed) {
    for (int i = 0; i < dim; ++i) {
        if (a[i] != (float)(seed + i)) return false;
    }
    return true;
}

int main(void) {
    int passes = 0, failures = 0;
    const int dim = 4;
    float src[4];
    fill(src, dim, 1);

    lembed_cache_config_t cfg = lembed_cache_config_default();
    cfg.capacity = 2;
    cfg.ttl_seconds = 0;
    lembed_cache_t* cache = lembed_cache_create(&cfg);
    ASSERT(cache != NULL, "cache creation must succeed");
    if (!cache) return 1;

    /* --- miss ------------------------------------------------------- */
    {
        int out_dim = -1;
        float buf[4];
        ASSERT(lembed_cache_get_copy(cache, "absent", buf, dim, &out_dim) == 0,
               "unknown key must report a miss");
        ASSERT(lembed_cache_size(cache) == 0, "a miss must not insert anything");
    }

    /* --- put then get_copy ------------------------------------------ */
    lembed_cache_put(cache, "a", src, dim);
    ASSERT(lembed_cache_size(cache) == 1, "put must insert one entry");
    {
        int out_dim = 0;
        float buf[4];
        memset(buf, 0, sizeof(buf));
        ASSERT(lembed_cache_get_copy(cache, "a", buf, dim, &out_dim) == 1,
               "stored key must be a hit");
        ASSERT(out_dim == dim, "get_copy must report the stored dimension");
        ASSERT(equals(buf, dim, 1), "get_copy must return the stored values");
    }

    /* --- capacity query: capacity 0 reads the dim without copying --- */
    {
        int out_dim = 0;
        ASSERT(lembed_cache_get_copy(cache, "a", NULL, 0, &out_dim) == -1,
               "capacity 0 must report the required size");
        ASSERT(out_dim == dim, "capacity 0 must still report the dimension");
    }

    /* --- too small buffer ------------------------------------------- */
    {
        int out_dim = 0;
        float buf[4];
        memset(buf, 0xAB, sizeof(buf));
        ASSERT(lembed_cache_get_copy(cache, "a", buf, dim - 1, &out_dim) == -1,
               "a short buffer must report the required size");
        ASSERT(out_dim == dim, "a short buffer must report the stored dimension");
        ASSERT(((const unsigned char*)buf)[0] == 0xAB,
               "a short buffer must be left untouched");
    }

    /* --- reading the same key twice must stay stable ---------------- */
    {
        float first[4];
        float second[4];
        int d1 = 0, d2 = 0;
        ASSERT(lembed_cache_get_copy(cache, "a", first, dim, &d1) == 1, "first read");
        ASSERT(lembed_cache_get_copy(cache, "a", second, dim, &d2) == 1, "second read");
        ASSERT(equals(first, dim, 1), "first read must be intact");
        ASSERT(equals(second, dim, 1), "second read must be intact");
    }

    /* --- the copied value survives eviction and clear --------------- */
    {
        float snapshot[4];
        int d = 0;
        ASSERT(lembed_cache_get_copy(cache, "a", snapshot, dim, &d) == 1, "snapshot read");

        /* capacity 2: inserting "b" then "c" evicts "a" (least recently used) */
        lembed_cache_put(cache, "b", src, dim);
        lembed_cache_put(cache, "c", src, dim);
        int out_dim = 0;
        ASSERT(lembed_cache_get_copy(cache, "a", NULL, 0, &out_dim) == 0,
               "the LRU victim must be gone");
        ASSERT(lembed_cache_size(cache) == 2, "the cache must hold two entries");
        ASSERT(equals(snapshot, dim, 1),
               "a value copied before the eviction must still be valid");

        lembed_cache_clear(cache);
        ASSERT(lembed_cache_size(cache) == 0, "clear must drop every entry");
        ASSERT(equals(snapshot, dim, 1),
               "a value copied before clear must still be valid");
    }

    /* --- LRU order: a re-read entry survives the next insertion ----- */
    {
        lembed_cache_clear(cache);
        float s2[4];
        fill(s2, dim, 50);
        float s3[4];
        fill(s3, dim, 90);
        lembed_cache_put(cache, "x", s2, dim);
        lembed_cache_put(cache, "y", s3, dim);
        int d = 0;
        float buf[4];
        ASSERT(lembed_cache_get_copy(cache, "x", buf, dim, &d) == 1, "read x");
        /* promotes x, so y becomes the eviction victim */
        lembed_cache_put(cache, "z", s2, dim);
        /* A NULL buffer turns every hit into a dimension query (-1). */
        ASSERT(lembed_cache_get_copy(cache, "y", NULL, 0, &d) == 0, "y must be evicted");
        ASSERT(lembed_cache_get_copy(cache, "x", NULL, 0, &d) == -1, "x must survive");
        ASSERT(d == dim, "x must still report its dimension");
        ASSERT(lembed_cache_get_copy(cache, "z", NULL, 0, &d) == -1, "z must be present");
    }

    /* --- borrowed pointer path: still valid, still not freed -------- */
    {
        float* borrowed = NULL;
        int d = 0;
        ASSERT(lembed_cache_get(cache, "x", &borrowed, &d) == 1, "lembed_cache_get hit");
        ASSERT(borrowed != NULL, "borrowed pointer must be set on a hit");
        ASSERT(equals(borrowed, dim, 50), "borrowed pointer must hold the value");
        /* The cache still owns the buffer: clearing it frees the memory once.
         * Reading `borrowed` afterwards would be a use-after-free, which is
         * exactly what the Python wrapper used to do. */
        lembed_cache_clear(cache);
    }

    /* --- overwrite keeps a single entry and returns the new value --- */
    {
        float s4[4];
        fill(s4, dim, 7);
        lembed_cache_put(cache, "w", s4, dim);
        lembed_cache_put(cache, "w", src, dim);
        ASSERT(lembed_cache_size(cache) == 1, "overwrite must not grow the cache");
        int d = 0;
        float buf[4];
        ASSERT(lembed_cache_get_copy(cache, "w", buf, dim, &d) == 1, "read w");
        ASSERT(equals(buf, dim, 1), "overwrite must replace the value");
    }

    /* --- empty vector round trip ------------------------------------ */
    {
        int d = -1;
        float buf[4];
        lembed_cache_put(cache, "empty", src, 0);
        ASSERT(lembed_cache_get_copy(cache, "empty", buf, 0, &d) == 1,
               "a zero-dimension entry is a hit");
        ASSERT(d == 0, "a zero-dimension entry reports a dimension of 0");
    }

    /* --- invalid arguments are rejected ----------------------------- */
    {
        int d = 0;
        float buf[4];
        ASSERT(lembed_cache_get_copy(NULL, "a", buf, dim, &d) == 0,
               "a NULL cache must be rejected");
        ASSERT(lembed_cache_get_copy(cache, NULL, buf, dim, &d) == 0,
               "a NULL key must be rejected");
        ASSERT(lembed_cache_get_copy(cache, "a", buf, dim, NULL) == 0,
               "a NULL out_dim must be rejected");
    }

    /* --- cache creation guards -------------------------------------- */
    {
        lembed_cache_config_t bad = lembed_cache_config_default();
        bad.capacity = 0;
        ASSERT(lembed_cache_create(&bad) == NULL, "capacity 0 must be rejected");
        ASSERT(lembed_cache_create(NULL) == NULL, "a NULL config must be rejected");
    }

    lembed_cache_free(cache);
    /* double free of the handle must be impossible: free(NULL) is a no-op */
    ASSERT(lembed_cache_size(NULL) == 0, "lembed_cache_size(NULL) must be 0");
    ASSERT(lembed_cache_capacity(NULL) == 0, "lembed_cache_capacity(NULL) must be 0");

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures > 0 ? 1 : 0;
}
