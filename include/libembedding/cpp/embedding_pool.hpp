/*
 * libembedding - cpp/embedding_pool.hpp
 * Session pool for parallel embedding processing
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_CPP_EMBEDDING_POOL_HPP
#define LIBEMBEDDING_CPP_EMBEDDING_POOL_HPP

#include <libembedding/text_embedding.h>
#include <libembedding/cpp/provider.hpp>

#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lembed {

struct PoolOptions {
    std::string                  model_path;
    int                          num_workers = 0;  /* 0 = auto (based on CPU cores) */
    int                          threads_per_worker = 1;
    int                          batch_size = 256;
    bool                         offline = false;
    std::string                  cache_dir;
    std::string                  provider = "cpu";
    int                          device_id = 0;
    int                          max_length = 0;
    bool                         show_download_progress = true;
    lembed_pooling_t             pooling = LEMBED_POOLING_MEAN;
    int                          dim = 0;
};

class EmbeddingPool {
public:
    EmbeddingPool(const PoolOptions& opts) {
        /* Auto-detect number of workers */
        int nw = opts.num_workers;
        if (nw <= 0) {
            int hw = (int)std::thread::hardware_concurrency();
            /* Use number of logical cores, capped at 8 */
            nw = hw > 0 ? std::min(hw, 8) : 2;
        }
        num_workers_ = nw;

        /* Create workers sequentially (ORT session creation is not thread-safe) */
        workers_.reserve(num_workers_);
        for (int i = 0; i < num_workers_; i++) {
            workers_.emplace_back(std::make_unique<Worker>(opts));
        }
    }

    ~EmbeddingPool() { close(); }

    EmbeddingPool(const EmbeddingPool&) = delete;
    EmbeddingPool& operator=(const EmbeddingPool&) = delete;

    /* Embed a batch of texts using all workers */
    std::vector<std::vector<float>> embed(const std::vector<std::string>& texts,
                                          int batch_size = 0) {
        if (texts.empty()) return {};

        int bs = (batch_size <= 0) ? workers_[0]->batch_size : batch_size;
        int total = (int)texts.size();

        /* Split texts across workers */
        int per_worker = (total + num_workers_ - 1) / num_workers_;

        /* Launch async tasks */
        std::vector<std::future<std::vector<std::vector<float>>>> futures;
        futures.reserve(num_workers_);

        for (int w = 0; w < num_workers_; w++) {
            int start = w * per_worker;
            int end = std::min(start + per_worker, total);
            if (start >= end) break;

            futures.push_back(std::async(std::launch::async, [this, w, &texts, start, end, bs]() {
                return workers_[w]->embed_subset(texts, start, end, bs);
            }));
        }

        /* Collect results */
        std::vector<std::vector<float>> result;
        result.reserve(total);

        for (auto& f : futures) {
            auto partial = f.get();
            for (auto& emb : partial) {
                result.push_back(std::move(emb));
            }
        }

        return result;
    }

    /* Embed texts as a stream, yielding one embedding at a time */
    template <typename Callback>
    void embed_stream(const std::vector<std::string>& texts,
                      int batch_size,
                      Callback callback) {
        if (texts.empty()) return;

        int bs = (batch_size <= 0) ? workers_[0]->batch_size : batch_size;
        int total = (int)texts.size();
        int per_worker = (total + num_workers_ - 1) / num_workers_;

        /* Use ordered queue to maintain text order */
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<std::vector<std::vector<float>>> partial_results(num_workers_);
        std::vector<bool> done(num_workers_, false);
        int next_to_yield = 0;

        std::vector<std::thread> threads;
        threads.reserve(num_workers_);

        for (int w = 0; w < num_workers_; w++) {
            int start = w * per_worker;
            int end = std::min(start + per_worker, total);
            if (start >= end) {
                done[w] = true;
                continue;
            }

            threads.emplace_back([this, w, &texts, start, end, bs,
                                  &mtx, &cv, &partial_results, &done, &next_to_yield,
                                  &callback, total]() {
                auto subset = workers_[w]->embed_subset(texts, start, end, bs);

                std::unique_lock<std::mutex> lock(mtx);
                partial_results[w] = std::move(subset);
                done[w] = true;

                /* Yield results in order */
                while (next_to_yield < num_workers_ && done[next_to_yield]) {
                    for (auto& emb : partial_results[next_to_yield]) {
                        callback(emb);
                    }
                    next_to_yield++;
                }
                cv.notify_all();
            });
        }

        for (auto& t : threads) t.join();
    }

    int dimension() const {
        return workers_.empty() ? 0 : workers_[0]->dim;
    }

    int num_workers() const { return num_workers_; }

    void close() {
        workers_.clear();
    }

private:
    struct Worker {
        lembed_text_embedding_t* ctx = nullptr;
        int dim = 0;
        int batch_size = 256;

        Worker(const PoolOptions& opts) {
            lembed_text_options_t c_opts = lembed_text_options_default();
            c_opts.provider = parse_provider(opts.provider);
            c_opts.device_id = opts.device_id;
            c_opts.max_length = opts.max_length;
            c_opts.num_threads = opts.threads_per_worker;
            c_opts.batch_size = opts.batch_size;
            c_opts.offline = opts.offline ? 1 : 0;
            c_opts.show_download_progress = opts.show_download_progress ? 1 : 0;
            c_opts.dim = opts.dim;
            c_opts.pooling = opts.pooling;

            int idx = lembed_find_text_model_by_code(opts.model_path.c_str());
            if (idx >= 0) {
                c_opts.model = (lembed_text_model_t)idx;
                lembed_status_t s = lembed_text_embedding_create(&c_opts, &ctx);
                if (s != LEMBED_OK) {
                    throw std::runtime_error(std::string("libembedding: ") + lembed_last_error());
                }
            } else {
                lembed_status_t s = lembed_text_embedding_create_from_path(
                    opts.model_path.c_str(), &c_opts, &ctx);
                if (s != LEMBED_OK) {
                    throw std::runtime_error(std::string("libembedding: ") + lembed_last_error());
                }
            }

            dim = lembed_text_embedding_dim(ctx);
            batch_size = ctx->batch_size;
        }

        ~Worker() {
            if (ctx) lembed_text_embedding_free(ctx);
        }

        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;

        std::vector<std::vector<float>> embed_subset(const std::vector<std::string>& texts,
                                                      int start, int end, int bs) {
            std::vector<std::vector<float>> result;

            for (int i = start; i < end; i += bs) {
                int n = std::min(bs, end - i);

                std::vector<const char*> c_texts;
                c_texts.reserve(n);
                for (int j = i; j < i + n; j++) {
                    c_texts.push_back(texts[j].c_str());
                }

                lembed_embeddings_t emb = {0};
                lembed_status_t s = lembed_text_embedding_embed(ctx, c_texts.data(), n, n, &emb);
                if (s == LEMBED_OK) {
                    for (int k = 0; k < emb.num_embeddings; k++) {
                        result.emplace_back(
                            emb.data + (size_t)k * emb.dim,
                            emb.data + (size_t)(k + 1) * emb.dim);
                    }
                    lembed_embeddings_free(&emb);
                }
            }

            return result;
        }
    };

    int num_workers_ = 0;
    std::vector<std::unique_ptr<Worker>> workers_;
};

} /* namespace lembed */

#endif /* LIBEMBEDDING_CPP_EMBEDDING_POOL_HPP */




