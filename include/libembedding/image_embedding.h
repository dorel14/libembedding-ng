/*
 * libembedding - image_embedding.h
 * Image embedding C API
 *
 * Auteur: David Orel
 * Version: 1.4.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_IMAGE_EMBEDDING_H
#define LIBEMBEDDING_IMAGE_EMBEDDING_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

lembed_image_options_t lembed_image_options_default(void);

lembed_status_t lembed_image_embedding_create(
    const lembed_image_options_t* options,
    lembed_image_embedding_t** out);

lembed_status_t lembed_image_embedding_create_from_path(
    const char* dir_path,
    const lembed_image_options_t* options,
    lembed_image_embedding_t** out);

/* Embed images from file paths */
lembed_status_t lembed_image_embedding_embed_files(
    lembed_image_embedding_t* ctx,
    const char* const* file_paths,
    int num_images,
    int batch_size,
    lembed_embeddings_t* result);

/* Embed images from memory buffers */
lembed_status_t lembed_image_embedding_embed_bytes(
    lembed_image_embedding_t* ctx,
    const unsigned char* const* image_data,
    const int* image_sizes,
    int num_images,
    int batch_size,
    lembed_embeddings_t* result);

int lembed_image_embedding_dim(const lembed_image_embedding_t* ctx);

/* Introspection */
const lembed_model_desc_t* lembed_image_embedding_desc(const lembed_image_embedding_t* ctx);
const char* lembed_image_embedding_model_name(const lembed_image_embedding_t* ctx);
int lembed_image_embedding_max_length(const lembed_image_embedding_t* ctx);

/* Runtime statistics */
void lembed_image_embedding_stats(const lembed_image_embedding_t* ctx, lembed_stats_t* out);

void lembed_image_embedding_free(lembed_image_embedding_t* ctx);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION
#ifndef LIBEMBEDDING_IMAGE_EMBEDDING_IMPL
#define LIBEMBEDDING_IMAGE_EMBEDDING_IMPL

#ifndef LIBEMBEDDING_NO_IMAGE

#include "model_registry.h"
#include "downloader.h"
#include "detail/model_loader_impl.hpp"
#include "detail/onnx_session_impl.hpp"
#include "detail/image_preprocess.hpp"
#include "detail/normalize.hpp"
#include "detail/batch.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

struct lembed_image_embedding {
    lembed::detail::OnnxSession session;
    int dim;
    int target_size; /* image preprocessing target size */

    /* Runtime metadata for introspection */
    std::string model_name_str;
    int num_threads;
    int batch_size;
    lembed_execution_provider_t provider;
    int device_id;
    lembed_model_desc_t desc;

    /* Stats counters */
    uint64_t texts_embedded = 0;
    uint64_t batches_run = 0;
    double   total_latency_ms = 0.0;
    int      stats_calls = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

lembed_status_t lembed_image_embedding_create(
        const lembed_image_options_t* options,
        lembed_image_embedding_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_image_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        char* model_dir_cstr = nullptr;
        s = lembed_ensure_image_model(options->model, options->cache_dir,
                                      options->show_download_progress,
                                      options->offline, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        auto* ctx = new lembed_image_embedding();
        ctx->dim = info.dim;
        ctx->target_size = 224; /* Standard for CLIP, ResNet, etc. */
        ctx->model_name_str = info.model_name;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->session.load_from_file(onnx_path.c_str(),
                                    options->num_threads,
                                    (int)options->provider);

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = ctx->dim;
        ctx->desc.max_length = 0;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_image_embedding_create_from_path(
        const char* dir_path,
        const lembed_image_options_t* options,
        lembed_image_embedding_t** out) {
    if (!dir_path || !dir_path[0] || !options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        /* Read model.onnx */
        std::string onnx_data = lembed::detail::read_file_to_string(
            std::string(dir_path) + "/model.onnx");

        /* Read config.json for dimension (optional) */
        int dim = 0, max_len = 0;
        std::string config_path = std::string(dir_path) + "/config.json";
        if (lembed::detail::file_exists(config_path)) {
            std::string config_data = lembed::detail::read_file_to_string(config_path);
            lembed::detail::parse_config_json(config_data, &dim, &max_len);
        }
        if (dim == 0) dim = options->dim;

        if (dim == 0) {
            lembed::detail::set_error(
                "Cannot determine model dimension: no config.json found and "
                "options.dim not set");
            return LEMBED_ERROR_INVALID_ARGUMENT;
        }

        auto* ctx = new lembed_image_embedding();
        ctx->dim = dim;
        ctx->target_size = 224;
        ctx->model_name_str = dir_path;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        ctx->session.load_from_memory(
            (const void*)onnx_data.data(), onnx_data.size(),
            options->num_threads, (int)options->provider);

        ctx->desc.name = ctx->model_name_str.c_str();
        ctx->desc.dimension = ctx->dim;
        ctx->desc.max_length = 0;
        ctx->desc.pooling = LEMBED_POOLING_MEAN;
        ctx->desc.num_threads = ctx->num_threads;
        ctx->desc.batch_size = ctx->batch_size;
        ctx->desc.provider = ctx->provider;
        ctx->desc.device_id = ctx->device_id;

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_IO;
    }
}

lembed_status_t lembed_image_embedding_embed_files(
        lembed_image_embedding_t* ctx,
        const char* const* file_paths,
        int num_images,
        int batch_size,
        lembed_embeddings_t* result) {
    if (!ctx || !file_paths || num_images <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        int dim = ctx->dim;
        result->dim = dim;
        result->num_embeddings = num_images;
        result->data = (float*)malloc((size_t)num_images * dim * sizeof(float));
        if (!result->data) return LEMBED_ERROR_OUT_OF_MEMORY;

        auto t_start = std::chrono::high_resolution_clock::now();
        int out_offset = 0;
        int num_batches = lembed::detail::batch_count(num_images, batch_size);

        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_images, batch_size);
            int bsz = range.end - range.start;
            int ts = ctx->target_size;

            /* Preprocess images directly into batch buffer (zero-copy) */
            std::vector<float> batch_data(bsz * 3 * ts * ts);
            for (int i = 0; i < bsz; i++) {
                lembed::detail::preprocess_image_to_buffer(
                    file_paths[range.start + i],
                    batch_data.data() + i * 3 * ts * ts,
                    ts);
            }

            /* Run ONNX with float input */
            int64_t shape[4] = { bsz, 3, ts, ts };

            /* Find the pixel_values input name (varies by model) */
            const char* input_name = "pixel_values";
            if (!ctx->session.has_input("pixel_values")) {
                if (ctx->session.has_input("input")) input_name = "input";
                else input_name = ctx->session.input_names()[0].c_str();
            }

            auto outputs = ctx->session.run_float(
                input_name, batch_data.data(), shape, 4);

            auto& output = outputs[0];
            int out_dim = (output.shape.size() >= 2)
                ? (int)output.shape[output.shape.size() - 1] : dim;

            /* L2 normalize */
            lembed::detail::l2_normalize(output.data.data(), bsz, out_dim);

            std::memcpy(result->data + out_offset * dim,
                       output.data.data(), (size_t)bsz * dim * sizeof(float));
            out_offset += bsz;
        }

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (result->data) { free(result->data); result->data = NULL; }
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

lembed_status_t lembed_image_embedding_embed_bytes(
        lembed_image_embedding_t* ctx,
        const unsigned char* const* image_data,
        const int* image_sizes,
        int num_images,
        int batch_size,
        lembed_embeddings_t* result) {
    if (!ctx || !image_data || !image_sizes || num_images <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    if (batch_size <= 0) batch_size = ctx->batch_size;

    try {
        int dim = ctx->dim;
        result->dim = dim;
        result->num_embeddings = num_images;
        result->data = (float*)malloc((size_t)num_images * dim * sizeof(float));
        if (!result->data) return LEMBED_ERROR_OUT_OF_MEMORY;

        auto t_start = std::chrono::high_resolution_clock::now();
        int out_offset = 0;
        int num_batches = lembed::detail::batch_count(num_images, batch_size);

        for (int bi = 0; bi < num_batches; bi++) {
            auto range = lembed::detail::get_batch(bi, num_images, batch_size);
            int bsz = range.end - range.start;
            int ts = ctx->target_size;

            std::vector<float> batch_data(bsz * 3 * ts * ts);
            for (int i = 0; i < bsz; i++) {
                int idx = range.start + i;
                lembed::detail::preprocess_image_bytes_to_buffer(
                    image_data[idx], image_sizes[idx],
                    batch_data.data() + i * 3 * ts * ts,
                    ts);
            }

            int64_t shape[4] = { bsz, 3, ts, ts };
            const char* input_name = "pixel_values";
            if (!ctx->session.has_input("pixel_values")) {
                if (ctx->session.has_input("input")) input_name = "input";
                else input_name = ctx->session.input_names()[0].c_str();
            }

            auto outputs = ctx->session.run_float(
                input_name, batch_data.data(), shape, 4);

            auto& output = outputs[0];
            lembed::detail::l2_normalize(output.data.data(), bsz, dim);

            std::memcpy(result->data + out_offset * dim,
                       output.data.data(), (size_t)bsz * dim * sizeof(float));
            out_offset += bsz;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ctx->texts_embedded += num_images;
        ctx->batches_run += num_batches;
        ctx->total_latency_ms += elapsed_ms;
        ctx->stats_calls++;

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (result->data) { free(result->data); result->data = NULL; }
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

int lembed_image_embedding_dim(const lembed_image_embedding_t* ctx) {
    return ctx ? ctx->dim : 0;
}

const lembed_model_desc_t* lembed_image_embedding_desc(const lembed_image_embedding_t* ctx) {
    return ctx ? &ctx->desc : nullptr;
}

const char* lembed_image_embedding_model_name(const lembed_image_embedding_t* ctx) {
    return ctx ? ctx->model_name_str.c_str() : nullptr;
}

int lembed_image_embedding_max_length(const lembed_image_embedding_t* ctx) {
    return 0; /* N/A for image models */
}

void lembed_image_embedding_stats(const lembed_image_embedding_t* ctx, lembed_stats_t* out) {
    if (!out) return;
    if (!ctx) { memset(out, 0, sizeof(*out)); return; }
    out->texts_embedded = ctx->texts_embedded;
    out->batches_run = ctx->batches_run;
    out->avg_latency_ms = ctx->stats_calls > 0
        ? ctx->total_latency_ms / (double)ctx->stats_calls
        : 0.0;
}

void lembed_image_embedding_free(lembed_image_embedding_t* ctx) {
    delete ctx;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_NO_IMAGE */
#endif /* LIBEMBEDDING_IMAGE_EMBEDDING_IMPL */
#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_IMAGE_EMBEDDING_H */




