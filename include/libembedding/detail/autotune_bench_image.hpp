/*
 * libembedding - detail/autotune_bench_image.hpp
 * Image embedding auto-tuner
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_IMAGE_HPP
#define LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_IMAGE_HPP

#ifndef LIBEMBEDDING_NO_IMAGE
#include "libembedding/autotuner.h"
#include "autotune_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Create a simple synthetic BMP image for benchmarking (no zlib needed) */
inline void create_benchmark_image(std::vector<unsigned char>& out, int width = 32, int height = 32) {
    /* BMP file header + DIB header + pixel data */
    int rowSize = ((width * 3 + 3) / 4) * 4;  /* rows padded to 4 bytes */
    int imageSize = rowSize * height;
    int fileSize = 54 + imageSize;

    out.clear();

    /* BMP file header (14 bytes) */
    out.push_back('B'); out.push_back('M');  /* signature */
    out.push_back(fileSize & 0xFF); out.push_back((fileSize >> 8) & 0xFF);
    out.push_back((fileSize >> 16) & 0xFF); out.push_back((fileSize >> 24) & 0xFF);
    out.push_back(0); out.push_back(0); out.push_back(0); out.push_back(0);  /* reserved */
    out.push_back(54); out.push_back(0); out.push_back(0); out.push_back(0);  /* data offset */

    /* DIB header (40 bytes) - BITMAPINFOHEADER */
    out.push_back(40); out.push_back(0); out.push_back(0); out.push_back(0);  /* header size */
    out.push_back(width & 0xFF); out.push_back((width >> 8) & 0xFF);
    out.push_back((width >> 16) & 0xFF); out.push_back((width >> 24) & 0xFF);
    out.push_back(height & 0xFF); out.push_back((height >> 8) & 0xFF);
    out.push_back((height >> 16) & 0xFF); out.push_back((height >> 24) & 0xFF);
    out.push_back(1); out.push_back(0);  /* color planes */
    out.push_back(24); out.push_back(0);  /* bits per pixel */
    out.push_back(0); out.push_back(0); out.push_back(0); out.push_back(0);  /* compression */
    for (int i = 0; i < 20; i++) out.push_back(0);  /* rest of header */

    /* Pixel data (BGR format) */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            out.push_back((x * 255) / width);   /* B */
            out.push_back((y * 255) / height);  /* G */
            out.push_back(128);                 /* R */
        }
        /* Pad row to 4-byte boundary */
        for (int p = 0; p < rowSize - width * 3; p++) out.push_back(0);
    }
}

/* Benchmark a single image configuration */
inline lembed_image_tuning_result_t bench_image_config(
    const char* model_name,
    int threads,
    int batch_size,
    int n_images,
    int warmup_iters,
    int bench_iters)
{
    lembed_image_tuning_result_t res = {0};
    res.threads = threads;
    res.batch_size = batch_size;

    lembed_image_options_t opts = lembed_image_options_default();
    opts.num_threads = threads;
    opts.batch_size = batch_size;
    opts.show_download_progress = 0;

    int model_idx = lembed_find_image_model_by_code(model_name);
    if (model_idx < 0) model_idx = 0;
    opts.model = static_cast<lembed_image_model_t>(model_idx);

    lembed_image_embedding_t* ctx = nullptr;
    lembed_status_t s = lembed_image_embedding_create(&opts, &ctx);
    if (s != LEMBED_OK) {
        res.latency_ms = 999999;
        return res;
    }

    /* Create benchmark images */
    std::vector<unsigned char> img_data;
    create_benchmark_image(img_data, 224, 224);

    std::vector<const unsigned char*> images;
    std::vector<int> sizes;
    for (int i = 0; i < n_images; i++) {
        images.push_back(img_data.data());
        sizes.push_back((int)img_data.size());
    }

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        lembed_embeddings_t result = {0};
        lembed_image_embedding_embed_bytes(ctx, images.data(), sizes.data(), n_images, batch_size, &result);
        lembed_embeddings_free(&result);
    }

    /* Benchmark */
    std::vector<double> times;
    for (int i = 0; i < bench_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        lembed_embeddings_t result = {0};
        lembed_image_embedding_embed_bytes(ctx, images.data(), sizes.data(), n_images, batch_size, &result);
        lembed_embeddings_free(&result);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    lembed_image_embedding_free(ctx);

    std::sort(times.begin(), times.end());
    double p50 = times[times.size() / 2];

    res.latency_ms = p50;
    res.throughput_docs_sec = (p50 > 0) ? (1000.0 / p50) * n_images : 0;

    return res;
}

/* Main image auto-tune function */
extern "C" lembed_status_t lembed_image_autotune(
    const char* model_name,
    lembed_autotune_mode_t mode,
    lembed_image_tuning_result_t* result)
{
    if (!model_name || !result) return LEMBED_ERROR_INVALID_ARGUMENT;

    int n_images = 16;
    int warmup = 2;
    int bench_iters = (mode == LEMBED_AUTOTUNE_QUICK) ? 5 : 15;

    int threads_options[] = {1, 2, 4, 8};
    int batch_options[] = {1, 4, 8, 16};

    lembed_image_tuning_result_t best = {0};
    best.latency_ms = 999999;

    int total = sizeof(threads_options) / sizeof(int) * sizeof(batch_options) / sizeof(int);
    int current = 0;

    fprintf(stderr, "image_autotune: testing %d configurations (mode=%s)...\n",
            total, mode == LEMBED_AUTOTUNE_QUICK ? "QUICK" : "FULL");

    for (int t : threads_options) {
        for (int b : batch_options) {
            current++;

            auto r = bench_image_config(model_name, t, b, n_images, warmup, bench_iters);

            if (r.latency_ms < best.latency_ms) {
                best = r;
            }
        }
    }

    fprintf(stderr, "image_autotune: best config: threads=%d batch=%d (P50=%.1fms)\n",
            best.threads, best.batch_size, best.latency_ms);

    *result = best;
    return LEMBED_OK;
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_NO_IMAGE */

#endif /* LIBEMBEDDING_DETAIL_AUTOTUNE_BENCH_IMAGE_HPP */




