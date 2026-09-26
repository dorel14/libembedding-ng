/*
 * libembedding - detail/embedding_cache_impl.hpp
 * LRU cache implementation for embeddings
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_EMBEDDING_CACHE_IMPL_HPP
#define LIBEMBEDDING_DETAIL_EMBEDDING_CACHE_IMPL_HPP

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <list>
#include <mutex>
#include <algorithm>
#include <vector>

namespace lembed { namespace detail {

struct CacheEntry {
    std::string key;
    float* vec;
    int dim;
    time_t expires_at;
};

class LRUCache {
public:
    LRUCache(size_t capacity, int ttl_seconds = 0)
        : capacity_(capacity), ttl_seconds_(ttl_seconds) {}

    ~LRUCache() { clear(); }

    // Returns a pointer owned by the cache (borrowed, see embedding_cache.h).
    // The mutex is released before the caller reads the buffer: a concurrent
    // put/eviction on the same key can free it while the caller is copying.
    // Prefer get_copy() whenever the caller does not need a stable pointer.
    bool get(const std::string& key, float** out_vec, int* dim) {
        std::lock_guard<std::mutex> lock(mtx_);
        Entry* e = touch(key);
        if (!e) return false;
        *out_vec = e->vec;
        *dim = e->dim;
        return true;
    }

    // Copies the value into caller-owned storage while holding the lock: no
    // cache-owned pointer escapes, so a concurrent put/eviction cannot free the
    // buffer in the middle of the copy (the race get() leaves to its caller).
    // *dim always receives the stored dimension on a hit, so a caller that does
    // not know the dimension up front can query it with capacity 0.
    // Returns 1 when the value was copied, 0 on miss, -1 when the entry exists
    // but `capacity` (or `out`) cannot hold it.
    int get_copy(const std::string& key, float* out, int capacity, int* dim) {
        std::lock_guard<std::mutex> lock(mtx_);
        Entry* e = touch(key);
        if (!e) return 0;
        if (dim) *dim = e->dim;
        if (!out || capacity < e->dim) return -1;
        memcpy(out, e->vec, (size_t)e->dim * sizeof(float));
        return 1;
    }

    bool get_copy(const std::string& key, std::vector<float>& out) {
        float* vec = nullptr;
        int dim = 0;
        if (!get(key, &vec, &dim) || dim <= 0) return false;
        out.assign(vec, vec + dim);
        return true;
    }

    void put(const std::string& key, const float* vec, int dim) {
        if (capacity_ == 0) return;

        std::lock_guard<std::mutex> lock(mtx_);
        float* copy = (float*)malloc((size_t)dim * sizeof(float));
        if (!copy) return;
        memcpy(copy, vec, (size_t)dim * sizeof(float));

        time_t expires_at = 0;
        if (ttl_seconds_ > 0) {
            expires_at = time(nullptr) + ttl_seconds_;
        }

        auto it = map_.find(key);
        if (it != map_.end()) {
            free(it->second->vec);
            it->second->vec = copy;
            it->second->dim = dim;
            it->second->expires_at = expires_at;
            list_.splice(list_.begin(), list_, it->second->it);
            it->second->it = list_.begin();
            return;
        }

        if (map_.size() >= capacity_) {
            auto last = list_.end();
            --last;
            remove_by_iterator(*last);
        }

        list_.push_front(key);
        Entry* e = new Entry{list_.begin(), key, copy, dim, expires_at};
        map_[key] = e;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& kv : map_) {
            free(kv.second->vec);
            delete kv.second;
        }
        map_.clear();
        list_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return map_.size();
    }

    size_t capacity() const { return capacity_; }

private:
    struct Entry {
        std::list<std::string>::iterator it;
        std::string key;
        float* vec;
        int dim;
        time_t expires_at;
    };

    size_t capacity_;
    int ttl_seconds_;
    mutable std::mutex mtx_;
    std::list<std::string> list_;
    std::unordered_map<std::string, Entry*> map_;

    // Looks up `key`, drops it when its TTL has expired and promotes it to the
    // front of the LRU list. Returns nullptr on miss. Caller must hold mtx_.
    Entry* touch(const std::string& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;

        if (ttl_seconds_ > 0 && time(nullptr) > it->second->expires_at) {
            remove(it);
            return nullptr;
        }

        list_.splice(list_.begin(), list_, it->second->it);
        it->second->it = list_.begin();
        return it->second;
    }

    void remove(std::unordered_map<std::string, Entry*>::iterator it) {
        free(it->second->vec);
        list_.erase(it->second->it);
        delete it->second;
        map_.erase(it);
    }

    void remove_by_iterator(const std::string& key) {
        auto it = map_.find(key);
        if (it != map_.end()) remove(it);
    }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_EMBEDDING_CACHE_IMPL_HPP */

