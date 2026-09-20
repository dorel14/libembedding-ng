/*
 * libembedding - detail/text_embedding_onnx_impl.hpp
 * ONNX backend implementation for text embeddings
 * Include-only header (guarded by LIBEMBEDDING_IMPLEMENTATION)
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_TEXT_EMBEDDING_ONNX_IMPL_HPP
#define LIBEMBEDDING_TEXT_EMBEDDING_ONNX_IMPL_HPP

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "This header must be included only when LIBEMBEDDING_IMPLEMENTATION is defined"
#endif

#include "../text_embedding.h"
#include "onnx_session_impl.hpp"
#include "tokenizer_impl.hpp"
#include "pooling.hpp"
#include "normalize.hpp"
#include "batch.hpp"
#include "embedding_cache_impl.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include <cstdlib>
#include <cstring>
#ifdef __cplusplus
}
extern "C++" {
#endif
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <filesystem>
#ifdef __cplusplus
}
extern "C" {
#endif

/* =========================================================================
 * ONNX backend: creation
 * ========================================================================= */

lembed_status_t lembed_text_embedding_create(
        const lembed_text_options_t* options,
        lembed_text_embedding_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    /* Validate backend enum range */
    if (options->backend < LEMBED_BACKEND_ONNX || options->backend > LEMBED_BACKEND_AUTO) {
        lembed::detail::set_error("Invalid backend value in text embedding options");
        return LEMBED_ERROR_INVALID_ARGUMENT;
    }

    try {
        lembed_model_info_t info;
        lembed_status_t s = lembed_get_text_model_info(options->model, &info);
        if (s != LEMBED_OK) return s;

        char* model_dir_cstr = nullptr;
        s = lembed_ensure_text_model(options->model, options->cache_dir,
                                     options->show_download_progress,
                                     options->offline, &model_dir_cstr);
        if (s != LEMBED_OK) return s;
        std::string model_dir(model_dir_cstr);
        lembed_free_string(model_dir_cstr);

        auto* ctx = new lembed_text_embedding();
        ctx->backend_type = LEMBED_BACKEND_ONNX;
        ctx->pooling = (lembed_pooling_t)info.pooling;
        ctx->quantization = (lembed_quantization_t)info.quantization;
        ctx->dim = info.dim;
        ctx->max_length = (options->max_length > 0) ? options->max_length : info.max_tokens;
        ctx->model_name_str = info.model_name;
        ctx->batch_strategy = (lembed_batch_strategy_t)options->batch_strategy;
        ctx->num_threads = options->num_threads;
        ctx->batch_size = (options->batch_size > 0) ? options->batch_size
                                                     : LEMBED_DEFAULT_BATCH_SIZE;
        ctx->provider = options->provider;
        ctx->device_id = options->device_id;

        if (options->model == LEMBED_TEXT_EMBEDDING_GEMMA_300M) {
            ctx->output_key = "sentence_embedding";
        }

        std::string onnx_path = model_dir + "/" + info.model_file;
        ctx->onnx.session.load_from_file(onnx_path.c_str(),
                                          options->num_threads,
                                          (int)options->provider,
                                          (int)ctx->quantization);

        std::string tok_path = model_dir + "/tokenizer.json";
        ctx->onnx.tokenizer.load_from_file(tok_path, ctx->max_length);

        lembed__text_update_desc(ctx);
        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

/* =========================================================================
 * Create (v2 - with explicit quantization override)
 * ========================================================================= */

lembed_status_t lembed_text_embedding_create_v2(
        const lembed_text_options_v2_t* options,
        lembed_text_embedding_t** out) {
    if (!options || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    lembed_status_t s = lembed_text_embedding_create(&options->base, out);
    if (s != LEMBED_OK || !*out) return s;

    (*out)->quantization = (lembed_quantization_t)options->quantization;
    return LEMBED_OK;
}

lembed_status_t lembed_text_embedding_create_custom(
        const lembed_user_defined_model_t* model,
        lembed_execution_provider_t provider,
        int num_threads,
        lembed_text_embedding_t** out) {
    if (!model || !out || !model->onnx_data || !model->tokenizer_json)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        auto* ctx = new lembed_text_embedding();
        ctx->backend_type = LEMBED_BACKEND_ONNX;
        ctx->pooling = model->pooling;
        ctx->quantization = LEMBED_QUANTIZATION_NONE;
        ctx->dim = model->dim;
        ctx->max_length = (model->max_length > 0) ? model->max_length : LEMBED_DEFAULT_MAX_LENGTH;
        ctx->model_name_str = "custom-model";
        ctx->num_threads = num_threads;
        ctx->batch_size = LEMBED_DEFAULT_BATCH_SIZE;
        ctx->batch_strategy = LEMBED_BATCH_LENGTH_BUCKET;
        ctx->provider = provider;
        ctx->device_id = 0;

        ctx->onnx.session.load_from_memory(model->onnx_data, model->onnx_data_size,
                                            num_threads, (int)provider);
        std::string tok_blob((const char*)model->tokenizer_json, model->tokenizer_json_size);
        ctx->onnx.tokenizer.load_from_blob(tok_blob, ctx->max_length);

        lembed__text_update_desc(ctx);
        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_ONNX_RUNTIME;
    }
}

/* =========================================================================
 * ONNX embed with batch strategy support
 * ========================================================================= */

static lembed_status_t lembed__onnx_embed(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        float* result_data) {
    int dim = ctx->dim;

    /* Prepare indices based on batch strategy */
    std::vector<int> indices(num_texts);
    for (int i = 0; i < num_texts; i++) indices[i] = i;

    if (ctx->batch_strategy == LEMBED_BATCH_LENGTH_BUCKET && num_texts > 1) {
        /* Pre-tokenize to get actual token counts, then sort by length */
        std::vector<int> token_counts(num_texts);
        for (int i = 0; i < num_texts; i++) {
            token_counts[i] = (int)ctx->onnx.tokenizer.encode(texts[i]).size();
        }
        std::sort(indices.begin(), indices.end(),
            [&token_counts](int a, int b) {
                return token_counts[a] > token_counts[b];
            });
    }

    int effective_batch = (ctx->batch_strategy == LEMBED_BATCH_SEQUENTIAL) ? 1 : batch_size;
    int num_batches = lembed::detail::batch_count(num_texts, effective_batch);

    for (int bi = 0; bi < num_batches; bi++) {
        auto range = lembed::detail::get_batch(bi, num_texts, effective_batch);
        int bsz = range.end - range.start;

        /* Tokenize batch (in sorted order if bucketing) */
        std::vector<std::string> batch_texts;
        batch_texts.reserve(bsz);
        for (int i = range.start; i < range.end; i++) {
            batch_texts.push_back(texts[indices[i]]);
        }

        auto enc = ctx->onnx.tokenizer.encode_batch(batch_texts);
        int seq_len = enc.seq_length;
        size_t flat_size = (size_t)bsz * seq_len;
        ctx->ids_buf_.resize(flat_size);
        ctx->mask_buf_.resize(flat_size);
        ctx->type_buf_.assign(flat_size, 0);

        for (int i = 0; i < bsz; i++) {
            for (int j = 0; j < seq_len; j++) {
                ctx->ids_buf_[i * seq_len + j] = enc.input_ids[i][j];
                ctx->mask_buf_[i * seq_len + j] = enc.attention_mask[i][j];
                if ((int)enc.token_type_ids[i].size() > j)
                    ctx->type_buf_[i * seq_len + j] = enc.token_type_ids[i][j];
            }
        }

        int oi = ctx->output_key.empty()
            ? ctx->onnx.session.select_output()
            : ctx->onnx.session.select_output(ctx->output_key.c_str());

        auto outputs = ctx->onnx.session.run(
            ctx->ids_buf_.data(), ctx->mask_buf_.data(), ctx->type_buf_.data(),
            bsz, seq_len, oi);
        auto& output = outputs[0];

        int ndim = (int)output.shape.size();
        int out_batch = (int)output.shape[0];
        int out_seq = (ndim >= 3) ? (int)output.shape[1] : 0;
        int out_dim = (ndim >= 3) ? (int)output.shape[2] :
                    (ndim == 2) ? (int)output.shape[1] : dim;

        ctx->pooled_buf_.resize((size_t)bsz * out_dim);
        if (ctx->pooling == LEMBED_POOLING_CLS) {
            lembed::detail::pool_cls(output.data.data(),
                out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
        } else {
            lembed::detail::pool_mean(output.data.data(), ctx->mask_buf_.data(),
                out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
        }

        lembed::detail::l2_normalize(ctx->pooled_buf_.data(), bsz, out_dim);

        /* Copy to output (reorder back to original positions) */
        for (int i = 0; i < bsz; i++) {
            int orig_idx = indices[range.start + i];
            std::memcpy(result_data + orig_idx * dim,
                       ctx->pooled_buf_.data() + i * dim,
                       dim * sizeof(float));
        }
    }

    return LEMBED_OK;
}

/* =========================================================================
 * ONNX stream implementation (extracted for clarity)
 * ========================================================================= */

static lembed_status_t lembed__onnx_embed_stream(
        lembed_text_embedding_t* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        void (*callback)(const float* embedding, int dim, void* userdata),
        void* userdata) {
    int dim = ctx->dim;
    int num_batches = lembed::detail::batch_count(num_texts, batch_size);

    for (int bi = 0; bi < num_batches; bi++) {
        auto range = lembed::detail::get_batch(bi, num_texts, batch_size);
        int bsz = range.end - range.start;

        std::vector<std::string> batch_texts;
        batch_texts.reserve(bsz);
        for (int i = range.start; i < range.end; i++)
            batch_texts.push_back(texts[i]);

        auto enc = ctx->onnx.tokenizer.encode_batch(batch_texts);
        int seq_len = enc.seq_length;
        size_t flat_size = (size_t)bsz * seq_len;
        ctx->ids_buf_.resize(flat_size);
        ctx->mask_buf_.resize(flat_size);
        ctx->type_buf_.assign(flat_size, 0);

        for (int i = 0; i < bsz; i++) {
            for (int j = 0; j < seq_len; j++) {
                ctx->ids_buf_[i * seq_len + j] = enc.input_ids[i][j];
                ctx->mask_buf_[i * seq_len + j] = enc.attention_mask[i][j];
                if ((int)enc.token_type_ids[i].size() > j)
                    ctx->type_buf_[i * seq_len + j] = enc.token_type_ids[i][j];
            }
        }

        int oi = ctx->output_key.empty()
            ? ctx->onnx.session.select_output()
            : ctx->onnx.session.select_output(ctx->output_key.c_str());

        auto outputs = ctx->onnx.session.run(
            ctx->ids_buf_.data(), ctx->mask_buf_.data(), ctx->type_buf_.data(),
            bsz, seq_len, oi);

        auto& output = outputs[0];
        int ndim = (int)output.shape.size();
        int out_batch = (int)output.shape[0];
        int out_seq = (ndim >= 3) ? (int)output.shape[1] : 0;
        int out_dim = (ndim >= 3) ? (int)output.shape[2] :
                    (ndim == 2) ? (int)output.shape[1] : dim;

        ctx->pooled_buf_.resize((size_t)bsz * out_dim);
        if (ctx->pooling == LEMBED_POOLING_CLS) {
            lembed::detail::pool_cls(output.data.data(),
                out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
        } else {
            lembed::detail::pool_mean(output.data.data(), ctx->mask_buf_.data(),
                out_batch, out_seq, out_dim, ndim, ctx->pooled_buf_.data());
        }

        lembed::detail::l2_normalize(ctx->pooled_buf_.data(), bsz, out_dim);

        for (int i = 0; i < bsz; i++) {
            callback(ctx->pooled_buf_.data() + (size_t)i * out_dim, dim, userdata);
        }
    }

    return LEMBED_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_TEXT_EMBEDDING_ONNX_IMPL_HPP */
