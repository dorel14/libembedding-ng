/*
 * Batch Strategy Comparison
 * Tests SEQUENTIAL vs FIXED_BATCH vs LENGTH_BUCKET on MiniLM and BGE-small
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

/* Generate a corpus with controlled length distribution */
static std::vector<std::string> make_corpus(int n_short, int n_medium, int n_long) {
    const char* shorts[] = {
        "Hello world.", "Machine learning.", "AI is great.", "Fast embeddings.",
        "Benchmark test.", "Performance.", "Optimization.", "Batch size.",
        "Tokenization.", "Inference speed.", "CPU utilization.", "Memory usage.",
    };
    const char* mediums[] = {
        "Machine learning algorithms can identify patterns in large datasets automatically without explicit programming instructions.",
        "The transformer architecture has become the foundation for most modern natural language processing systems including BERT and GPT.",
        "Deep learning is part of a broader family of machine learning methods based on artificial networks with representation learning.",
        "Natural language processing enables computers to understand and analyze human language in various forms including text and speech.",
        "Climate change affects global weather patterns and sea levels significantly across all continents and ocean regions worldwide.",
        "Quantum computing promises to revolutionize cryptography drug discovery and materials science through parallel processing capabilities.",
    };
    const char* longs[] = {
        "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy.",
        "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think. Since then the field has gone through periods of optimism and disappointment known as AI winters.",
        "Les algorithmes d'apprentissage automatique peuvent identifier des motifs dans de grands ensembles de données automatiquement sans programmation explicite. Ces avancées ont permis le développement de systèmes capables de comprendre et de traduire le langage humain.",
        "Die künstliche Intelligenz hat bedeutende Fortschritte gemacht insbesondere in den Bereichen maschinelles Lernen und Verarbeitung natürlicher Sprache. Diese Fortschritte haben die Entwicklung von Systemen ermöglicht.",
    };

    std::vector<std::string> corpus;
    for (int i = 0; i < n_short; i++) corpus.push_back(shorts[i % 12]);
    for (int i = 0; i < n_medium; i++) corpus.push_back(mediums[i % 6]);
    for (int i = 0; i < n_long; i++) corpus.push_back(longs[i % 4]);
    return corpus;
}

static double now_ms() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

struct BenchResult {
    float docs_per_sec;
    float total_ms;
    int dim;
};

static BenchResult bench_onnx(const char* onnx_dir, const std::vector<std::string>& corpus,
                               lembed_batch_strategy_t strategy, int batch_size) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 4;
    opts.batch_size = batch_size;
    opts.batch_strategy = strategy;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    BenchResult r = {0, 0, 0};
    if (lembed_text_embedding_create_from_path(onnx_dir, &opts, &emb) != LEMBED_OK) return r;

    int total = (int)corpus.size();
    std::vector<const char*> texts;
    for (auto& t : corpus) texts.push_back(t.c_str());

    lembed_embeddings_t result = {0};
    double t0 = now_ms();
    lembed_status_t s = lembed_text_embedding_embed(emb, texts.data(), total, batch_size, &result);
    double t1 = now_ms();

    if (s == LEMBED_OK) {
        r.docs_per_sec = (float)total / ((float)(t1 - t0) / 1000.0f);
        r.total_ms = (float)(t1 - t0);
        r.dim = result.dim;
        lembed_embeddings_free(&result);
    }
    lembed_text_embedding_free(emb);
    return r;
}

int main() {
    fprintf(stderr, "=== Batch Strategy Comparison ===\n\n");

    const char* onnx_minilm = "C:\\Users\\david\\.cache\\libembedding\\models--Qdrant-all-MiniLM-L6-v2-onnx";
    const char* onnx_bge = "C:\\Users\\david\\.cache\\libembedding\\models--Xenova-bge-small-en-v1.5\\onnx";

    /* Create corpus: 48 short + 24 medium + 16 long = 88 texts */
    auto corpus = make_corpus(48, 24, 16);
    int total = (int)corpus.size();
    fprintf(stderr, "Corpus: %d texts (%d short, %d medium, %d long)\n\n", total, 48, 24, 16);

    const char* strategy_names[] = {"SEQUENTIAL", "FIXED_BATCH", "LENGTH_BUCKET"};
    lembed_batch_strategy_t strategies[] = {
        LEMBED_BATCH_SEQUENTIAL,
        LEMBED_BATCH_FIXED,
        LEMBED_BATCH_LENGTH_BUCKET
    };

    /* ===== MiniLM ===== */
    fprintf(stderr, "--- MiniLM-L6 (ONNX) ---\n");
    fprintf(stderr, "%-15s %-10s %-10s %-10s\n", "Strategy", "Batch", "Docs/s", "Time(ms)");
    fprintf(stderr, "---------------------------------------------------\n");

    int batch_sizes[] = {1, 8, 16, 32, 64};
    for (int si = 0; si < 3; si++) {
        for (int bsz : batch_sizes) {
            /* Skip batch_size > 1 for SEQUENTIAL */
            if (strategies[si] == LEMBED_BATCH_SEQUENTIAL && bsz > 1) continue;
            auto r = bench_onnx(onnx_minilm, corpus, strategies[si], bsz);
            fprintf(stderr, "%-15s %-10d %-10.1f %-10.1f\n",
                    strategy_names[si], bsz, r.docs_per_sec, r.total_ms);
        }
    }

    /* ===== BGE-small ===== */
    fprintf(stderr, "\n--- BGE-small (ONNX) ---\n");
    fprintf(stderr, "%-15s %-10s %-10s %-10s\n", "Strategy", "Batch", "Docs/s", "Time(ms)");
    fprintf(stderr, "---------------------------------------------------\n");

    for (int si = 0; si < 3; si++) {
        for (int bsz : batch_sizes) {
            if (strategies[si] == LEMBED_BATCH_SEQUENTIAL && bsz > 1) continue;
            auto r = bench_onnx(onnx_bge, corpus, strategies[si], bsz);
            fprintf(stderr, "%-15s %-10d %-10.1f %-10.1f\n",
                    strategy_names[si], bsz, r.docs_per_sec, r.total_ms);
        }
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
