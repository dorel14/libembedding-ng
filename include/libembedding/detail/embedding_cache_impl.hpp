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

    bool get(const std::string& key, float** out_vec, int* dim) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;

        if (ttl_seconds_ > 0 && time(nullptr) > it->second->expires_at) {
            remove(it);
            return false;
        }

        *out_vec = it->second->vec;
        *dim = it->second->dim;
        list_.splice(list_.begin(), list_, it->second->it);
        it->second->it = list_.begin();
        return true;
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

