/*
 * Baseline Benchmark — Phase 1
 * Measures: workers/sessions, threads, bucketing, batching
 * Corpus: multilingual, multi-length from roadmap
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

static double baseline_now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static const char* corpus[] = {
    "What is machine learning?",
    "How does BERT work?",
    "Explain transformers",
    "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with the interactions between computers and human language in particular how to program computers to process and analyze large amounts of natural language data.",
    "The transformer architecture introduced in the attention is all you need paper has become the foundation for most modern natural language processing systems including BERT GPT and their variants which have revolutionized the field.",
    "Artificial intelligence has made significant progress in recent years particularly in the areas of machine learning deep learning and natural language processing. These advances have enabled the development of systems that can understand generate and translate human language with remarkable accuracy. Large language models trained on vast amounts of text data have demonstrated capabilities that were previously thought to be decades away including reasoning summarization and creative writing that rivals human output in many domains.",
    "The development of modern artificial intelligence began in the nineteen fifties with the work of Alan Turing and other pioneers who asked whether machines could think. Since then the field has gone through periods of optimism and disappointment known as AI winters. Today we are in a period of rapid advancement driven by increases in computational power the availability of large datasets and improvements in algorithms particularly deep learning.",
    "Hello world.",
    "Bonjour le monde.",
    "Hallo Welt.",
    "Hola mundo.",
    "Ciao mondo.",
    "Olá mundo.",
    "Привет мир.",
    "こんにちは世界。",
    "안녕하세요 세계.",
    "你好世界。",
    "Machine learning transforms data into insights.",
    "L'intelligence artificielle transforme les données.",
    "Künstliche Intelligenz verändert die Welt.",
    "Climate change affects global weather.",
    "Quantum computing promises revolution.",
    "The history of ancient Rome spans centuries.",
    "Natural language processing is a subfield of linguistics and artificial intelligence.",
    "The transformer architecture has become the foundation for modern NLP systems.",
    "Deep learning is part of machine learning methods based on artificial networks.",
    "Le traitement automatique du langage naturel est un domaine de l'IA.",
    "Die künstliche Intelligenz verändert die Art und Weise wie wir arbeiten.",
    "El procesamiento del lenguaje natural es un campo de la informática.",
    "Machine learning algorithms identify patterns in large datasets automatically.",
    "Climate change affects global weather patterns significantly.",
    "The history of ancient Rome spans over a thousand years.",
    "Quantum computing promises to revolutionize cryptography.",
    "Les algorithmes d'apprentissage automatique identifient des motifs.",
    "Die künstliche Intelligenz ist ein Gebiet der Informatik.",
    "El aprendizaje automático permite a las computadoras aprender.",
    "Natural language processing enables computers to understand language.",
    "The transformer model uses self-attention efficiently.",
    "Deep neural networks learn hierarchical representations through layers.",
    "Artificial intelligence has made significant progress in recent years.",
    "The development of modern AI began in the nineteen fifties.",
    "L'intelligence artificielle a fait des progrès significatifs.",
    "Die künstliche Intelligenz hat bedeutende Fortschritte gemacht.",
    "El inteligencia artificial ha hecho progresos significativos.",
};
static const int corpus_size = sizeof(corpus) / sizeof(corpus[0]);

struct BenchResult {
    int sessions;
    int threads;
    const char* strategy;
    double ms;
    float docs_per_sec;
};

static double run_pool(const char* model_path, int sessions, int threads, const char* strategy_name) {
    try {
        lembed::detail::LlamaSessionPool pool;
        pool.load_from_file(model_path, sessions, threads, 0, 0, 0, false);

        std::vector<float> checksums(corpus_size);
        for (int i = 0; i < corpus_size; i++) {
            auto v = pool.embed(corpus[i]);
            float s = 0;
            for (size_t j = 0; j < v.size(); j++) s += v[j] * (float)(j + 1);
            checksums[i] = s;
        }

        std::atomic<int> idx{0};
        double t0 = baseline_now_ms();
        int iterations = 5;
        for (int iter = 0; iter < iterations; iter++) {
            std::vector<std::thread> workers;
            for (int w = 0; w < sessions; w++) {
                workers.emplace_back([&]() {
                    while (true) {
                        int i = idx.fetch_add(1, std::memory_order_relaxed);
                        if (i >= corpus_size) break;
                        pool.embed(corpus[i]);
                    }
                });
            }
            for (auto& t : workers) t.join();
            idx.store(0);
        }
        double t1 = baseline_now_ms();

        double total_ms = t1 - t0;
        int total_texts = corpus_size * iterations;
        return total_ms / total_texts;
    } catch (...) {
        return -1;
    }
}

int main(int argc, char** argv) {
    const char* model_path = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";
    if (argc > 1) model_path = argv[1];

    fprintf(stderr, "=== Baseline Benchmark: workers x threads x strategy ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Corpus: %d texts\n\n", corpus_size);

    std::vector<BenchResult> results;

    int session_counts[] = {1, 2, 4, 6, 8};
    int thread_counts[] = {1, 2, 4};
    const char* strategies[] = {"naive", "length_bucket"};

    for (int s : session_counts) {
        for (int t : thread_counts) {
            double avg_ms = run_pool(model_path, s, t, "naive");
            if (avg_ms > 0) {
                results.push_back({s, t, "naive", avg_ms, (float)(1000.0 / avg_ms)});
            }
        }
    }

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  RESULTS\n");
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "%-8s %-8s %-12s %10s %12s\n",
            "Sessions", "Threads", "Strategy", "ms/doc", "docs/sec");
    fprintf(stderr, "--------------------------------------------------------------------\n");

    for (const auto& r : results) {
        fprintf(stderr, "%-8d %-8d %-12s %10.2f %12.1f\n",
                r.sessions, r.threads, r.strategy, r.ms, r.docs_per_sec);
    }

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
