/*
 * Quality Benchmark: ONNX vs llama.cpp
 *
 * Measures cosine similarity between embeddings from both backends
 * for the same model (MiniLM-L6-v2).
 *
 * This validates that the llama.cpp backend produces embeddings
 * that are numerically close to the ONNX backend.
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

/* Cosine similarity between two vectors */
static float cosine_sim(const float* a, const float* b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-8f) ? dot / denom : 0.0f;
}

/* L2 norm */
static float l2_norm(const float* v, int dim) {
    float s = 0;
    for (int i = 0; i < dim; i++) s += v[i] * v[i];
    return sqrtf(s);
}

int main() {
    fprintf(stderr, "=== Quality Benchmark: ONNX vs llama.cpp ===\n\n");

    /* Models to compare (same model, different backends) */
    const char* onnx_dir = "C:\\Users\\david\\.cache\\libembedding\\models--Qdrant-all-MiniLM-L6-v2-onnx";
    const char* gguf_path = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";

    /* Test texts */
    const char* texts[] = {
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "The transformer architecture has become the foundation for modern NLP.",
        "Deep learning models require large amounts of training data.",
        "Climate change affects global weather patterns significantly.",
        "Quantum computing promises to revolutionize cryptography.",
        "The history of ancient Rome spans over a thousand years.",
        "Artificial intelligence has made significant progress in recent years.",
        "Fast embeddings are essential for real-time applications.",
    };
    int n_texts = sizeof(texts) / sizeof(texts[0]);

    /* Create ONNX embedder */
    lembed_text_options_t onnx_opts = lembed_text_options_default();
    onnx_opts.num_threads = 4;
    onnx_opts.batch_size = 16;
    onnx_opts.show_download_progress = 0;

    lembed_text_embedding_t* onnx_emb = nullptr;
    if (lembed_text_embedding_create_from_path(onnx_dir, &onnx_opts, &onnx_emb) != LEMBED_OK) {
        fprintf(stderr, "Failed to create ONNX embedder: %s\n", lembed_last_error());
        return 1;
    }
    int dim = lembed_text_embedding_dim(onnx_emb);
    fprintf(stderr, "ONNX embedder: dim=%d\n", dim);

    /* Create llama.cpp embedder */
    lembed_text_options_t llama_opts = lembed_text_options_default();
    llama_opts.num_threads = 1;
    llama_opts.show_download_progress = 0;

    lembed_text_embedding_t* llama_emb = nullptr;
    if (lembed_text_embedding_create_from_gguf_path(gguf_path, &llama_opts, &llama_emb) != LEMBED_OK) {
        fprintf(stderr, "Failed to create llama.cpp embedder: %s\n", lembed_last_error());
        lembed_text_embedding_free(onnx_emb);
        return 1;
    }
    int dim_llama = lembed_text_embedding_dim(llama_emb);
    fprintf(stderr, "llama.cpp embedder: dim=%d\n\n", dim_llama);

    if (dim != dim_llama) {
        fprintf(stderr, "ERROR: dimension mismatch (ONNX=%d, llama=%d)\n", dim, dim_llama);
        lembed_text_embedding_free(onnx_emb);
        lembed_text_embedding_free(llama_emb);
        return 1;
    }

    /* Embed with ONNX */
    std::vector<const char*> text_ptrs;
    for (int i = 0; i < n_texts; i++) text_ptrs.push_back(texts[i]);

    lembed_embeddings_t onnx_result = {0};
    double t0 = now_ms();
    lembed_text_embedding_embed(onnx_emb, text_ptrs.data(), n_texts, 16, &onnx_result);
    double t1 = now_ms();
    float onnx_time_ms = (float)(t1 - t0);

    /* Embed with llama.cpp */
    lembed_embeddings_t llama_result = {0};
    double t2 = now_ms();
    lembed_text_embedding_embed(llama_emb, text_ptrs.data(), n_texts, 1, &llama_result);
    double t3 = now_ms();
    float llama_time_ms = (float)(t3 - t2);

    /* Compare embeddings */
    fprintf(stderr, "--- Per-text Cosine Similarity ---\n");
    fprintf(stderr, "%-50s %10s %10s %10s\n", "Text", "Cosine", "ONNX norm", "llama norm");
    fprintf(stderr, "--------------------------------------------------------------------------\n");

    float min_cos = 1.0f, max_cos = -1.0f, sum_cos = 0;
    int nan_count = 0;

    for (int i = 0; i < n_texts; i++) {
        float* onnx_vec = onnx_result.data + (size_t)i * dim;
        float* llama_vec = llama_result.data + (size_t)i * dim;

        float cos = cosine_sim(onnx_vec, llama_vec, dim);
        float onnx_n = l2_norm(onnx_vec, dim);
        float llama_n = l2_norm(llama_vec, dim);

        if (std::isnan(cos) || std::isinf(cos)) {
            nan_count++;
            fprintf(stderr, "%-50.49s %10s %10.4f %10.4f\n", texts[i], "NaN", onnx_n, llama_n);
        } else {
            min_cos = std::min(min_cos, cos);
            max_cos = std::max(max_cos, cos);
            sum_cos += cos;
            fprintf(stderr, "%-50.49s %10.6f %10.4f %10.4f\n", texts[i], cos, onnx_n, llama_n);
        }
    }

    float mean_cos = sum_cos / n_texts;

    fprintf(stderr, "\n--- Summary ---\n");
    fprintf(stderr, "Mean cosine similarity: %.6f\n", mean_cos);
    fprintf(stderr, "Min cosine similarity:  %.6f\n", min_cos);
    fprintf(stderr, "Max cosine similarity:  %.6f\n", max_cos);
    fprintf(stderr, "NaN/Inf count:          %d/%d\n", nan_count, n_texts);
    fprintf(stderr, "ONNX time:              %.2f ms (%.1f docs/s)\n", onnx_time_ms, (float)n_texts / (onnx_time_ms / 1000.0f));
    fprintf(stderr, "llama.cpp time:         %.2f ms (%.1f docs/s)\n", llama_time_ms, (float)n_texts / (llama_time_ms / 1000.0f));

    /* Quality assessment */
    fprintf(stderr, "\n--- Quality Assessment ---\n");
    if (mean_cos > 0.99f) {
        fprintf(stderr, "EXCELLENT: embeddings are nearly identical\n");
    } else if (mean_cos > 0.95f) {
        fprintf(stderr, "GOOD: embeddings are very similar (minor quantization differences)\n");
    } else if (mean_cos > 0.90f) {
        fprintf(stderr, "ACCEPTABLE: embeddings are similar (quantization visible)\n");
    } else if (mean_cos > 0.80f) {
        fprintf(stderr, "WARNING: embeddings differ significantly\n");
    } else {
        fprintf(stderr, "CRITICAL: embeddings are very different (bug?)\n");
    }

    /* Check for NaN/Inf in embeddings */
    int onnx_nan = 0, llama_nan = 0;
    for (int i = 0; i < n_texts * dim; i++) {
        if (std::isnan(onnx_result.data[i]) || std::isinf(onnx_result.data[i])) onnx_nan++;
        if (std::isnan(llama_result.data[i]) || std::isinf(llama_result.data[i])) llama_nan++;
    }
    fprintf(stderr, "ONNX NaN/Inf values: %d/%d\n", onnx_nan, n_texts * dim);
    fprintf(stderr, "llama NaN/Inf values: %d/%d\n", llama_nan, n_texts * dim);

    lembed_embeddings_free(&onnx_result);
    lembed_embeddings_free(&llama_result);
    lembed_text_embedding_free(onnx_emb);
    lembed_text_embedding_free(llama_emb);

    return (mean_cos > 0.90f && nan_count == 0) ? 0 : 1;
}
