/*
 * libembedding - detail/sparse_text_embedding_gguf_impl.hpp
 * The C entry points of the sparse GGUF runtime.
 *
 * Included at the end of sparse_text_embedding.h's implementation block, after
 * `struct lembed_sparse_embedding` exists and after the ONNX entry points have
 * been declared.
 *
 * Two families of entry points, and the difference matters:
 *
 *   - lembed_sparse_text_embedding_create() and ..._create_from_path() are the
 *     ONNX registry. They take a directory or a registry index and know nothing
 *     about GGUF.
 *   - ..._create_from_gguf_path() and ..._create_from_gguf_model() take a .gguf
 *     file. They are named for the *format*, not for the runtime: nothing here
 *     goes through llama.cpp, so a "_llama_" suffix would be a lie that costs a
 *     rename later.
 *
 * A `.gguf` path handed to ..._create_from_path() is routed here, so that a
 * caller who only knows a file path does not have to know which backend it is.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_GGUF_IMPL
#define LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_GGUF_IMPL

#ifndef LIBEMBEDDING_IMPLEMENTATION
#error "sparse_text_embedding_gguf_impl.hpp requires LIBEMBEDDING_IMPLEMENTATION"
#endif

#include "gguf/gguf_sparse_session.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

/* ---- Embedding, dispatched from the shared C entry point ------------------ */

lembed_status_t lembed_sparse_text_embedding_embed_gguf(
        struct lembed_sparse_embedding* ctx,
        const char* const* texts,
        int num_texts,
        int batch_size,
        const lembed_sparse_options_t* sparse_opts,
        lembed_sparse_embeddings_t* result) {
    if (!ctx || !ctx->gguf || !texts || num_texts <= 0 || !result)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        std::vector<std::string> batch_texts;
        batch_texts.reserve((size_t)num_texts);
        for (int i = 0; i < num_texts; i++) {
            batch_texts.emplace_back(texts[i] ? texts[i] : "");
        }

        /* An explicit sparse_opts always wins, top_k == 0 included: 0 means
         * "keep every term", not "unset". The context values are the fallback
         * for a NULL sparse_opts, exactly as on the ONNX path. */
        const int top_k = sparse_opts ? sparse_opts->top_k : ctx->top_k;
        const float min_weight = sparse_opts ? sparse_opts->min_weight
                                             : ctx->min_weight;
        const int storage_format = sparse_opts ? sparse_opts->storage_format
                                               : ctx->storage_format;

        const auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<lembed::detail::SparseResult> sparse =
            ctx->gguf->embed(batch_texts, batch_size);

        /* count is 0 until the array exists: a caller that frees the result on a
         * non-OK status must not walk num_texts slots of a null pointer. */
        result->count = 0;
        result->items = (lembed_sparse_embedding_t*)calloc(
            num_texts, sizeof(lembed_sparse_embedding_t));
        if (!result->items) return LEMBED_ERROR_OUT_OF_MEMORY;

        int out_offset = 0;
        const int num_batches = lembed::detail::batch_count(num_texts, batch_size);
        for (size_t b = 0; b < sparse.size() && out_offset < num_texts; b++) {
            lembed::detail::SparseResult& sr = sparse[b];
            lembed::detail::sparse_prune_and_sort(sr, top_k, min_weight,
                                                  storage_format);
            lembed::detail::sparse_result_store(result, out_offset, sr);
            out_offset++;
        }
        /* The session returns one result per text, so out_offset == num_texts in
         * practice. A shorter count would mean the graph gave up on a document:
         * the remaining slots stay null rather than pointing at nothing. */
        result->count = out_offset;

        const auto t_end = std::chrono::high_resolution_clock::now();
        ctx->texts_embedded += (uint64_t)out_offset;
        ctx->batches_run += (uint64_t)num_batches;
        ctx->total_latency_ms += std::chrono::duration<double, std::milli>(
                                     t_end - t_start).count();
        ctx->stats_calls++;

        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        lembed_sparse_embeddings_free(result);
        return LEMBED_ERROR_GGUF;
    }
}

/* ---- Context creation ---------------------------------------------------- */

namespace {

/* Builds the context around an already-loaded session, filling in everything
 * the introspection API reports. `model_name` is what the caller will read back
 * from lembed_sparse_text_embedding_model_name(), and `resolved_max_length` is
 * what the session ended up with -- the request capped at the model's own
 * context length, which is not necessarily what was asked for. */
void finish_gguf_ctx(lembed_sparse_embedding_ctx_t* ctx,
                     const std::string& model_name,
                     const lembed_sparse_options_t& options,
                     int resolved_max_length) {
    /* Same sentinel as the two ONNX constructors: a model loaded from a path has
     * no registry index. Nothing reads ctx->model for such a context, and giving
     * it a real one would be a lie. */
    ctx->model = LEMBED_SPARSE_MODEL_COUNT;
    ctx->max_length = resolved_max_length;
    ctx->model_name_str = model_name;
    ctx->num_threads = options.num_threads;
    ctx->batch_size = (options.batch_size > 0) ? options.batch_size
                                               : LEMBED_DEFAULT_BATCH_SIZE;
    /* The GGUF runtime is a CPU ggml graph; there is no provider to honour, and
     * reporting LEMBED_PROVIDER_LLAMACPP would be wrong -- llama.cpp is not
     * involved. */
    ctx->provider = LEMBED_PROVIDER_CPU;
    ctx->device_id = options.device_id;

    ctx->desc.name = ctx->model_name_str.c_str();
    /* 0, as on both ONNX paths: a sparse vector has no fixed dimension, the
     * vocabulary is not a length. Reporting the vocabulary size here would make
     * the same field mean two different things depending on the backend. */
    ctx->desc.dimension = 0;
    ctx->desc.max_length = ctx->max_length;
    ctx->desc.pooling = LEMBED_POOLING_MEAN; /* max-pool, as for every SPLADE model */
    ctx->desc.num_threads = ctx->num_threads;
    ctx->desc.batch_size = ctx->batch_size;
    ctx->desc.provider = ctx->provider;
    ctx->desc.device_id = ctx->device_id;
    ctx->top_k = options.top_k;
    ctx->min_weight = options.min_weight;
    ctx->storage_format = options.storage_format;
}

} /* anonymous namespace */

#ifdef __cplusplus
extern "C" {
#endif

lembed_status_t lembed_sparse_text_embedding_create_from_gguf_path(
        const char* path,
        const lembed_sparse_options_t* options,
        lembed_sparse_embedding_ctx_t** out) {
    if (!path || !path[0] || !out) return LEMBED_ERROR_INVALID_ARGUMENT;

    lembed_sparse_options_t opts =
        options ? *options : lembed_sparse_options_default();

    try {
        auto session = std::unique_ptr<lembed::gguf::SparseSession>(
            new lembed::gguf::SparseSession());
        std::string err;
        if (!session->load_from_file(path, opts.num_threads, opts.max_length, err)) {
            /* Rule 2 of the GGUF convention: say what was missing. A refusal
             * with a reason is debuggable; a refusal without one is not. */
            lembed::detail::set_error(err.c_str());
            return LEMBED_ERROR_UNSUPPORTED;
        }

        auto* ctx = new lembed_sparse_embedding();
        ctx->gguf = std::move(session);
        finish_gguf_ctx(ctx, path, opts, ctx->gguf->max_length());

        *out = ctx;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_GGUF;
    }
}

lembed_status_t lembed_sparse_text_embedding_create_from_gguf_model(
        const char* repo,
        const char* filename,
        const lembed_sparse_options_t* options,
        lembed_sparse_embedding_ctx_t** out) {
    if (!repo || !repo[0] || !filename || !filename[0] || !out)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    lembed_sparse_options_t opts =
        options ? *options : lembed_sparse_options_default();

    char* cached_path = nullptr;
    const lembed_status_t s = lembed_ensure_gguf_model(
        repo, filename, opts.cache_dir, opts.show_download_progress, opts.offline,
        &cached_path);
    if (s != LEMBED_OK) return s;

    const std::string path(cached_path);
    lembed_free_string(cached_path);

    return lembed_sparse_text_embedding_create_from_gguf_path(path.c_str(),
                                                              &opts, out);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_SPARSE_TEXT_EMBEDDING_GGUF_IMPL */