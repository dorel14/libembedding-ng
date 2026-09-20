/*
 * Bucketing Benchmark for llama.cpp
 * Measures LENGTH_BUCKET vs naive batching
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <libembedding/detail/llama_session_impl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static const char* corpus[] = {
    "Hello world.",
    "Bonjour le monde.",
    "Hallo Welt.",
    "Machine learning transforms data into insights.",
    "L'intelligence artificielle transforme les données.",
    "Künstliche Intelligenz verändert die Welt.",
    "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with the interactions between computers and human language in particular how to program computers to process and analyze large amounts of natural language data.",
    "The transformer architecture introduced in the attention is all you need paper has become the foundation for most modern natural language processing systems including BERT GPT and their variants which have revolutionized the field.",
    "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy. Large language models trained on vast amounts of text data have demonstrated capabilities that were previously thought to be decades away including reasoning summarization and creative writing that rivals human output in many domains.",
    "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think. Since then the field has gone through periods of optimism and disappointment known as AI winters. Today we are in a period of rapid advancement driven by increases in computational power the availability of large datasets and improvements in algorithms particularly deep learning.",
};

int main(int argc, char** argv) {
    const char* model_path = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";
    if (argc > 1) model_path = argv[1];

    int sessions = (argc > 2) ? atoi(argv[2]) : 6;
    int batch_size = (argc > 3) ? atoi(argv[3]) : 8;

    fprintf(stderr, "=== Bucketing Benchmark: sessions=%d, batch_size=%d ===\n", sessions, batch_size);
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Corpus: %d texts\n\n", (int)(sizeof(corpus) / sizeof(corpus[0])));

    std::vector<std::string> texts;
    for (size_t i = 0; i < sizeof(corpus) / sizeof(corpus[0]); i++) {
        texts.emplace_back(corpus[i]);
    }

    try {
        lembed::detail::LlamaSessionPool pool;
        pool.load_from_file(model_path, sessions, 1, 0, 0, 0, false);

        auto naive = pool.embed_batch(texts);
        auto bucketed = pool.embed_batch_bucketed(texts, batch_size);

        fprintf(stderr, "Naive: %zu embeddings\n", naive.size());
        fprintf(stderr, "Bucketed: %zu embeddings\n", bucketed.size());

        double t0 = now_ms();
        for (int iter = 0; iter < 50; iter++) {
            pool.embed_batch(texts);
        }
        double t1 = now_ms();
        double naive_ms = (t1 - t0) / 50.0;

        double t2 = now_ms();
        for (int iter = 0; iter < 50; iter++) {
            pool.embed_batch_bucketed(texts, batch_size);
        }
        double t3 = now_ms();
        double bucketed_ms = (t3 - t2) / 50.0;

        fprintf(stderr, "\nResults (avg over 50 runs):\n");
        fprintf(stderr, "  Naive:     %.2f ms/batch (%.1f docs/s)\n", naive_ms, 1000.0 / naive_ms * texts.size());
        fprintf(stderr, "  Bucketed:  %.2f ms/batch (%.1f docs/s)\n", bucketed_ms, 1000.0 / bucketed_ms * texts.size());
        if (naive_ms > 0) {
            fprintf(stderr, "  Gain:      %.1f%%\n", (naive_ms - bucketed_ms) / naive_ms * 100.0);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
