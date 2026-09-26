/*
 * libembedding - detail/llama_scheduler_impl.hpp
 * Dynamic batching scheduler for llama.cpp session pool
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_LLAMA_SCHEDULER_IMPL_HPP
#define LIBEMBEDDING_DETAIL_LLAMA_SCHEDULER_IMPL_HPP

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <algorithm>
#include <chrono>
#include <future>

namespace lembed { namespace detail {

/* =========================================================================
 * Request types
 * ========================================================================= */

struct LlamaEmbedRequest {
    std::string text;
    std::vector<float> result;
    std::exception_ptr error;
    std::chrono::steady_clock::time_point enqueue_time;
};

/* =========================================================================
 * Dynamic batching scheduler
 * ========================================================================= */

class LlamaScheduler {
public:
    using Callback = std::function<void(const std::vector<std::string>&, std::vector<std::vector<float>>&)>;

    LlamaScheduler(Callback process_batch, int max_batch_size = 32, int max_wait_ms = 5)
        : process_batch_(std::move(process_batch))
        , max_batch_size_(max_batch_size)
        , max_wait_ms_(max_wait_ms)
        , stop_(false)
    {
        worker_ = std::thread([this]() { worker_loop(); });
    }

    ~LlamaScheduler() {
        stop();
    }

    void stop() {
        if (stop_.exchange(true)) return;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.notify_all();
        }
        if (worker_.joinable()) worker_.join();
    }

    std::vector<float> submit(const std::string& text) {
        std::promise<std::vector<float>> p;
        std::future<std::vector<float>> f = p.get_future();

        {
            std::unique_lock<std::mutex> lk(mtx_);
            requests_.push({std::move(p), std::move(f), text, std::chrono::steady_clock::now()});
            cv_.notify_one();
        }

        return f.get();
    }

    void set_max_batch_size(int n) { max_batch_size_ = n; }
    void set_max_wait_ms(int ms) { max_wait_ms_ = ms; }

private:
    struct QueuedRequest {
        std::promise<std::vector<float>> promise;
        std::future<std::vector<float>> future;
        std::string text;
        std::chrono::steady_clock::time_point enqueue_time;
    };

    Callback process_batch_;
    int max_batch_size_;
    int max_wait_ms_;
    std::atomic<bool> stop_;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<QueuedRequest> requests_;
    std::thread worker_;

    void worker_loop() {
        while (!stop_) {
            std::vector<QueuedRequest> batch;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, std::chrono::milliseconds(max_wait_ms_),
                              [this]() { return stop_ || !requests_.empty(); });
                if (stop_) break;

                if (requests_.empty()) continue;

                while (!requests_.empty() && (int)batch.size() < max_batch_size_) {
                    batch.push_back(std::move(requests_.front()));
                    requests_.pop();
                }
            }

            if (batch.empty()) continue;

            std::vector<std::string> texts;
            texts.reserve(batch.size());
            for (auto& r : batch) texts.push_back(std::move(r.text));

            std::vector<std::vector<float>> embeddings;
            try {
                process_batch_(texts, embeddings);
            } catch (...) {
                embeddings.assign(batch.size(), {});
            }

            for (size_t i = 0; i < batch.size(); ++i) {
                if (i < embeddings.size() && !embeddings[i].empty()) {
                    batch[i].promise.set_value(std::move(embeddings[i]));
                } else if (std::current_exception()) {
                    batch[i].promise.set_exception(std::current_exception());
                } else {
                    batch[i].promise.set_exception(
                        std::make_exception_ptr(std::runtime_error("Empty embedding")));
                }
            }
        }
    }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_LLAMA_SCHEDULER_IMPL_HPP */
