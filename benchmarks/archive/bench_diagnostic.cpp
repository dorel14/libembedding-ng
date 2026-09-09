/*
 * ONNX Performance Diagnostic
 *
 * Investigate why ONNX dropped from 230-700 docs/s to 15.8 docs/s
 * Tests:
 * 1. Batch size sweep (1, 8, 16, 32, 64, 128)
 * 2. Latency vs throughput consistency
 * 3. Length bucketing vs naive mixed
 * 4. Separate profiling: tokenize / inference / pooling
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

/* Test corpus with various lengths */
static const char* corpus_short[] = {
    "Hello world.",
    "Machine learning.",
    "AI is great.",
    "Fast embeddings.",
    "Benchmark test.",
    "Performance.",
    "Optimization.",
    "Batch size.",
};

static const char* corpus_medium[] = {
    "Machine learning algorithms can identify patterns in large datasets automatically without explicit programming instructions.",
    "The transformer architecture has become the foundation for most modern natural language processing systems.",
    "Deep learning is part of a broader family of machine learning methods based on artificial networks.",
    "Natural language processing enables computers to understand human language.",
};

static const char* corpus_long[] = {
    "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy.",
    "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think. Since then the field has gone through periods of optimism and disappointment known as AI winters.",
};

/* Generate mixed corpus with controlled length distribution */
static std::vector<std::string> make_mixed_corpus(int n_per_bucket) {
    std::vector<std::string> corpus;
    for (int i = 0; i < n_per_bucket; i++) {
        for (auto& t : corpus_short) corpus.push_back(t);
        for (auto& t : corpus_medium) corpus.push_back(t);
        for (auto& t : corpus_long) corpus.push_back(t);
    }
    return corpus;
}

/* Sort corpus by length (for bucketing) */
static std::vector<std::string> sort_by_length(std::vector<std::string> corpus) {
    std::sort(corpus.begin(), corpus.end(),
              [](const std::string& a, const std::string& b) {
                  return a.size() < b.size();
              });
    return corpus;
}

int main() {
    fprintf(stderr, "=== ONNX Performance Diagnostic ===\n\n");

    const char* onnx_dir = "C:\\Users\\david\\.cache\\libembedding\\models--Qdrant-all-MiniLM-L6-v2-onnx";
    const char* gguf_path = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";

    /* ===== Test 1: Batch size sweep (ONNX) ===== */
    fprintf(stderr, "--- Test 1: ONNX Batch Size Sweep ---\n");
    fprintf(stderr, "%-10s %-10s %-10s %-10s %-10s\n", "Batch", "Corpus", "Docs/s", "P50(ms)", "Theoretical");
    fprintf(stderr, "-----------------------------------------------------------\n");

    int batch_sizes[] = {1, 8, 16, 32, 64, 128};

    for (int bsz : batch_sizes) {
        /* Short corpus */
        {
            lembed_backend_config_t cfg = {"onnx", 4, bsz, 1};
            lembed_benchmark_result_t r = {0};
            if (lembed_benchmark_run(onnx_dir, "onnx", LEMBED_CORPUS_SHORT, &cfg, &r) == LEMBED_OK) {
                float theoretical = 1000.0f / r.metrics.latency_p50_ms;
                fprintf(stderr, "%-10d %-10s %-10.1f %-10.2f %-10.0f\n",
                        bsz, "Short", r.metrics.throughput_docs_sec,
                        r.metrics.latency_p50_ms, theoretical);
            }
        }
        /* Medium corpus */
        {
            lembed_backend_config_t cfg = {"onnx", 4, bsz, 1};
            lembed_benchmark_result_t r = {0};
            if (lembed_benchmark_run(onnx_dir, "onnx", LEMBED_CORPUS_MEDIUM, &cfg, &r) == LEMBED_OK) {
                float theoretical = 1000.0f / r.metrics.latency_p50_ms;
                fprintf(stderr, "%-10d %-10s %-10.1f %-10.2f %-10.0f\n",
                        bsz, "Medium", r.metrics.throughput_docs_sec,
                        r.metrics.latency_p50_ms, theoretical);
            }
        }
    }

    /* ===== Test 2: Latency vs Throughput consistency ===== */
    fprintf(stderr, "\n--- Test 2: Latency vs Throughput Consistency ---\n");
    fprintf(stderr, "%-10s %-10s %-10s %-10s %-10s\n", "Backend", "Batch", "Docs/s", "P50(ms)", "1000/P50");
    fprintf(stderr, "-----------------------------------------------------------\n");

    /* ONNX batch=1 */
    {
        lembed_backend_config_t cfg = {"onnx", 4, 1, 1};
        lembed_benchmark_result_t r = {0};
        if (lembed_benchmark_run(onnx_dir, "onnx", LEMBED_CORPUS_SHORT, &cfg, &r) == LEMBED_OK) {
            fprintf(stderr, "%-10s %-10d %-10.1f %-10.2f %-10.0f\n",
                    "ONNX", 1, r.metrics.throughput_docs_sec,
                    r.metrics.latency_p50_ms, 1000.0f / r.metrics.latency_p50_ms);
        }
    }
    /* ONNX batch=64 */
    {
        lembed_backend_config_t cfg = {"onnx", 4, 64, 1};
        lembed_benchmark_result_t r = {0};
        if (lembed_benchmark_run(onnx_dir, "onnx", LEMBED_CORPUS_SHORT, &cfg, &r) == LEMBED_OK) {
            fprintf(stderr, "%-10s %-10d %-10.1f %-10.2f %-10.0f\n",
                    "ONNX", 64, r.metrics.throughput_docs_sec,
                    r.metrics.latency_p50_ms, 1000.0f / r.metrics.latency_p50_ms);
        }
    }
    /* llama.cpp batch=1 */
    {
        lembed_backend_config_t cfg = {"llama.cpp", 1, 1, 0};
        lembed_benchmark_result_t r = {0};
        if (lembed_benchmark_run(gguf_path, "llama.cpp", LEMBED_CORPUS_SHORT, &cfg, &r) == LEMBED_OK) {
            fprintf(stderr, "%-10s %-10d %-10.1f %-10.2f %-10.0f\n",
                    "llama", 1, r.metrics.throughput_docs_sec,
                    r.metrics.latency_p50_ms, 1000.0f / r.metrics.latency_p50_ms);
        }
    }

    /* ===== Test 3: Mixed corpus analysis ===== */
    fprintf(stderr, "\n--- Test 3: Mixed Corpus Analysis ---\n");
    fprintf(stderr, "Testing with custom mixed corpus (short+medium+long mixed)...\n");

    /* Create a custom mixed corpus benchmark */
    auto mixed = make_mixed_corpus(4);  /* 48 texts total */
    auto sorted = sort_by_length(mixed);

    /* ONNX with batch=1 on mixed (no padding issue) */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.num_threads = 4;
        opts.show_download_progress = 0;
        lembed_text_embedding_t* emb = nullptr;
        lembed_status_t s = lembed_text_embedding_create_from_path(onnx_dir, &opts, &emb);
        if (s == LEMBED_OK) {
            int dim = lembed_text_embedding_dim(emb);
            /* Convert mixed corpus to const char** */
            std::vector<const char*> texts;
            for (auto& t : mixed) texts.push_back(t.c_str());
            int total = (int)texts.size();

            /* batch=1: each text individually */
            double t0 = now_ms();
            for (int i = 0; i < total; i++) {
                lembed_embeddings_t r = {0};
                lembed_text_embedding_embed(emb, &texts[i], 1, 1, &r);
                lembed_embeddings_free(&r);
            }
            double t1 = now_ms();
            float docs_per_sec = (float)total / ((float)(t1 - t0) / 1000.0f);
            fprintf(stderr, "ONNX batch=1 mixed: %.1f docs/s (%.2f ms/text)\n",
                    docs_per_sec, (float)(t1-t0)/total);
            lembed_text_embedding_free(emb);
        }
    }

    /* ONNX with batch=64 on mixed (padding issue) */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.num_threads = 4;
        opts.show_download_progress = 0;
        lembed_text_embedding_t* emb = nullptr;
        lembed_status_t s = lembed_text_embedding_create_from_path(onnx_dir, &opts, &emb);
        if (s == LEMBED_OK) {
            int dim = lembed_text_embedding_dim(emb);
            std::vector<const char*> texts;
            for (auto& t : mixed) texts.push_back(t.c_str());
            int total = (int)texts.size();

            /* batch=64: all at once */
            double t0 = now_ms();
            lembed_embeddings_t r = {0};
            lembed_text_embedding_embed(emb, texts.data(), total, 64, &r);
            lembed_embeddings_free(&r);
            double t1 = now_ms();
            float docs_per_sec = (float)total / ((float)(t1 - t0) / 1000.0f);
            fprintf(stderr, "ONNX batch=64 mixed: %.1f docs/s (%.2f ms total)\n",
                    docs_per_sec, (float)(t1-t0));
            lembed_text_embedding_free(emb);
        }
    }

    /* ONNX with sorted corpus (length bucketing) */
    {
        lembed_text_options_t opts = lembed_text_options_default();
        opts.num_threads = 4;
        opts.show_download_progress = 0;
        lembed_text_embedding_t* emb = nullptr;
        lembed_status_t s = lembed_text_embedding_create_from_path(onnx_dir, &opts, &emb);
        if (s == LEMBED_OK) {
            int dim = lembed_text_embedding_dim(emb);
            std::vector<const char*> texts;
            for (auto& t : sorted) texts.push_back(t.c_str());
            int total = (int)texts.size();

            /* batch=8 on sorted (similar lengths together) */
            double t0 = now_ms();
            for (int off = 0; off < total; off += 8) {
                int n = std::min(8, total - off);
                lembed_embeddings_t r = {0};
                lembed_text_embedding_embed(emb, texts.data() + off, n, 8, &r);
                lembed_embeddings_free(&r);
            }
            double t1 = now_ms();
            float docs_per_sec = (float)total / ((float)(t1 - t0) / 1000.0f);
            fprintf(stderr, "ONNX batch=8 sorted: %.1f docs/s (%.2f ms total)\n",
                    docs_per_sec, (float)(t1-t0));
            lembed_text_embedding_free(emb);
        }
    }

    fprintf(stderr, "\n=== Diagnostic Complete ===\n");
    return 0;
}
