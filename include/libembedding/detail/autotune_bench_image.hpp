/*
 * libembedding - detail/autotune_bench_image.hpp
 * Image embedding auto-tuner
 *
 * Tunes threads x batch_size for an image model on a synthetic batch of
 * 224x224 images (the input size of the registry's vision models, so the
 * resize step costs what it costs in production).
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_IMAGE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_IMAGE_HPP

#ifndef LIBEMBEDDING_NO_IMAGE
#include "libembedding/autotuner.h"
#include "libembedding/image_embedding.h"
#include "libembedding/model_registry.h"
#include "autotune_cache.hpp"
#include "autotune_result.hpp"
#include "autotune_search.hpp"
#include "autotune_session.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* =========================================================================
 * Workload
 * ========================================================================= */

/* 24-bit BMP: decodable by stb_image without zlib, deterministic content. */
inline std::vector<unsigned char> create_benchmark_image(int width, int height) {
    const int row = ((width * 3 + 3) / 4) * 4;
    const int size = 54 + row * height;
    std::vector<unsigned char> b((size_t)size, 0);
    auto put32 = [&](int off, int v) {
        for (int i = 0; i < 4; i++) b[(size_t)(off + i)] = (unsigned char)((v >> (8 * i)) & 0xFF);
    };
    b[0] = 'B';
    b[1] = 'M';
    put32(2, size);
    put32(10, 54);    /* pixel data offset */
    put32(14, 40);    /* BITMAPINFOHEADER */
    put32(18, width);
    put32(22, height);
    b[26] = 1;        /* planes */
    b[28] = 24;       /* bits per pixel */
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            unsigned char* p = &b[(size_t)(54 + y * row + x * 3)];
            p[0] = (unsigned char)((x * 255) / width);
            p[1] = (unsigned char)((y * 255) / height);
            p[2] = 128;
        }
    return b;
}

inline int image_tune_batch_images(lembed_autotune_mode_t mode) {
    return (mode == LEMBED_AUTOTUNE_QUICK) ? 16 : 32;
}

/* =========================================================================
 * Configuration space
 * ========================================================================= */

struct image_tune_config {
    int threads;
    int batch;
};

inline std::string describe_image_config(const image_tune_config& c) {
    return "threads=" + std::to_string(c.threads) + " batch=" + std::to_string(c.batch);
}

inline std::vector<image_tune_config> image_tune_grid(lembed_autotune_mode_t mode) {
    int physical = std::max(1, cpu_physical_cores());
    int logical = std::max(physical, cpu_logical_cores());
    std::vector<int> threads = {physical, logical};
    if (mode == LEMBED_AUTOTUNE_FULL) {
        threads.push_back(1);
        threads.push_back(std::max(1, physical / 2));
    }
    std::sort(threads.begin(), threads.end());
    threads.erase(std::unique(threads.begin(), threads.end()), threads.end());

    std::vector<int> batches = (mode == LEMBED_AUTOTUNE_QUICK) ? std::vector<int>{1, 8, 16}
                                                               : std::vector<int>{1, 4, 8, 16, 32};
    std::vector<image_tune_config> grid;
    for (int t : threads)
        for (int b : batches) grid.push_back({t, b});
    return grid;
}

/* =========================================================================
 * Measuring one configuration
 * ========================================================================= */

/* The batch size is a per-call argument: one session per thread count. */
using image_session_cache = autotune_session_cache<int, lembed_image_embedding_t>;

inline autotune_measurement bench_image_config(lembed_image_model_t model,
                                               const std::vector<unsigned char>& img,
                                               int n_images,
                                               const image_tune_config& c,
                                               const autotune_protocol& proto,
                                               image_session_cache& sessions) {
    autotune_measurement failed;

    lembed_image_embedding_t* ctx = sessions.get(c.threads, [&]() -> lembed_image_embedding_t* {
        lembed_image_options_t opts = lembed_image_options_default();
        opts.model = model;
        opts.num_threads = c.threads;
        opts.offline = 1;
        opts.show_download_progress = 0;
        lembed_image_embedding_t* created = nullptr;
        return lembed_image_embedding_create(&opts, &created) == LEMBED_OK ? created : nullptr;
    });
    if (!ctx) return failed;

    std::vector<const unsigned char*> images((size_t)n_images, img.data());
    std::vector<int> sizes((size_t)n_images, (int)img.size());
    return autotune_measure_calls(proto, n_images, sessions.rss_before_mb(), [&]() {
        lembed_embeddings_t r = {0};
        bool ok = lembed_image_embedding_embed_bytes(ctx, images.data(), sizes.data(), n_images,
                                                     c.batch, &r) == LEMBED_OK;
        lembed_embeddings_free(&r);
        return ok;
    });
}

/* =========================================================================
 * Cache
 * ========================================================================= */

inline const char* image_cache_subdir() { return "image"; }

/* Registry index of an image model named by its repo or its canonical name. */
inline int resolve_image_model(const char* model) {
    if (!model) return -1;
    int idx = lembed_find_image_model_by_code(model);
    if (idx >= 0) return idx;
    const int count = (int)(sizeof(lembed__image_models) / sizeof(lembed__image_models[0]));
    for (int i = 0; i < count; i++)
        if (lembed__image_models[i].model_name &&
            std::strcmp(lembed__image_models[i].model_name, model) == 0)
            return i;
    return -1;
}

inline std::string canonical_image_id(const char* model) {
    if (!model) return std::string();
    int idx = resolve_image_model(model);
    if (idx < 0) return std::string(model);
    return std::string(lembed__image_models[idx].model_code);
}

/* =========================================================================
 * Tuner
 * ========================================================================= */

inline lembed_objective_t image_default_objective() { return LEMBED_OBJECTIVE_THROUGHPUT; }

/* max_latency_ms bounds the p95 of one call over the synthetic batch. */
inline lembed_status_t image_autotune_run(const char* model_name, lembed_autotune_mode_t mode,
                                          lembed_objective_t objective, double max_latency_ms,
                                          lembed_unified_tuning_result_t* result) {
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = make_unified_result(LEMBED_TASK_IMAGE);

    int idx = resolve_image_model(model_name);
    if (idx < 0)
        return autotune_fail(LEMBED_ERROR_MODEL_NOT_FOUND,
                             std::string("image_autotune: unknown image model '") + model_name +
                                 "'");

    const int n_images = image_tune_batch_images(mode);
    char variant[64];
    std::snprintf(variant, sizeof(variant), "synthetic%d_p95_%.3f", n_images, max_latency_ms);
    autotune_cache_identity id;
    id.model = canonical_image_id(model_name);
    id.variant = variant;
    id.corpus = "bmp224";
    id.objective = (int)objective;
    id.mode = (int)mode;

    if (read_unified_cache(image_cache_subdir(), id, *result)) return LEMBED_OK;

    autotune_search_options sopts;
    sopts.objective = objective;
    sopts.constraints.max_p95_ms = max_latency_ms;
    sopts.protocol = autotune_protocol_for(mode);
    sopts.label = "image_autotune";

    lembed_image_model_t model = (lembed_image_model_t)idx;
    std::vector<unsigned char> img = create_benchmark_image(224, 224);
    image_session_cache sessions(lembed_image_embedding_free);
    auto search = autotune_grid_search(
        image_tune_grid(mode),
        [&](const image_tune_config& c, const autotune_protocol& p) {
            return bench_image_config(model, img, n_images, c, p, sessions);
        },
        describe_image_config, sopts);
    sessions.reset();
    if (!search.found()) return autotune_fail_no_config("image_autotune", id.model);

    result->threads = search.best_config().threads;
    result->batch_size = search.best_config().batch;
    set_result_metrics(*result, search.best_measurement());

    write_unified_cache(image_cache_subdir(), id, *result);
    return LEMBED_OK;
}

inline lembed_image_tuning_result_t to_image_result(const lembed_unified_tuning_result_t& u) {
    lembed_image_tuning_result_t r = {0};
    r.threads = u.threads;
    r.batch_size = u.batch_size;
    r.throughput_docs_sec = u.throughput_docs_sec;
    r.latency_ms = u.latency_ms;
    r.memory_mb = u.memory_mb;
    return r;
}

extern "C" lembed_status_t lembed_image_autotune(const char* model_name,
                                                 lembed_autotune_mode_t mode,
                                                 lembed_image_tuning_result_t* result) {
    if (!result) return LEMBED_ERROR_INVALID_ARGUMENT;
    *result = lembed_image_tuning_result_t{0};
    return autotune_guard("image_autotune", [&]() {
        lembed_unified_tuning_result_t u;
        lembed_status_t s = image_autotune_run(model_name, mode, image_default_objective(), 0.0, &u);
        if (s == LEMBED_OK) *result = to_image_result(u);
        return s;
    });
}

/* NULL clears every image entry; otherwise the model's entries only. */
inline void lembed_image_autotune_clear_cache(const char* model_name) {
    autotune_guard("image_autotune_clear_cache", [&]() {
        if (!model_name) clear_autotune_cache(nullptr, image_cache_subdir());
        else clear_autotune_cache(canonical_image_id(model_name).c_str(), image_cache_subdir());
        return LEMBED_OK;
    });
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_NO_IMAGE */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_IMAGE_HPP */
