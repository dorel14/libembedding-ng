/*
 * Quick llama.cpp backend benchmark - fast iteration test
 * Uses cached GGUF model, measures throughput at batch sizes 32, 128, 256
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const char* model_path = R"(C:\Users\david\.cache\libembedding\models--BAAI-bge-small-en-v1.5\bge-small-en-v1.5-Q8_0.gguf)";
    if (argc > 1) model_path = argv[1];

    FILE* log = fopen("bench_llamacpp_log.txt", "w");
    if (!log) return 1;

    fprintf(log, "=== Quick llama.cpp Benchmark ===\n");
    fprintf(log, "Model: %s\n", model_path);
    fprintf(log, "Backend available: %s\n\n", lembed_llama_backend_available() ? "yes" : "no");
    fflush(log);

    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = 0; /* auto */
    opts.show_download_progress = 0;

    fprintf(log, "Creating embedder...\n");
    fflush(log);

    lembed_text_embedding_t* embedder = nullptr;
    double t0 = now_ms();
    lembed_status_t s = lembed_text_embedding_create_from_gguf_path(model_path, &opts, &embedder);
    double t1 = now_ms();

    fprintf(log, "Load time: %.2f ms\n", t1 - t0);
    fprintf(log, "Status: %d\n", s);
    fflush(log);

    if (s != LEMBED_OK) {
        fprintf(log, "Error: %s\n", lembed_last_error());
        fflush(log);
        fclose(log);
        return 1;
    }

    fprintf(log, "Model: %s (dim=%d)\n",
            lembed_text_embedding_model_name(embedder),
            lembed_text_embedding_dim(embedder));
    fflush(log);

    /* Generate test corpus */
    const char* sample_texts[] = {
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Vector embeddings represent text as dense numerical representations.",
    };
    const int NUM_SAMPLES = 4;

    std::vector<std::string> corpus;
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < NUM_SAMPLES; j++) {
            corpus.push_back(sample_texts[j]);
        }
    }
    int total_texts = (int)corpus.size();
    fprintf(log, "Corpus size: %d texts\n", total_texts);
    fflush(log);

    std::vector<const char*> texts;
    for (auto& s : corpus) texts.push_back(s.c_str());

    /* Warmup */
    lembed_embeddings_t warmup = {0};
    lembed_text_embedding_embed(embedder, texts.data(), 32, 32, &warmup);
    lembed_embeddings_free(&warmup);
    fprintf(log, "Warmup done\n");
    fflush(log);

    /* Benchmark */
    int batch_sizes[] = {32, 128, 256};
    int iters = 3;

    fprintf(log, "\n=== Benchmark Results ===\n");
    fprintf(log, "%-12s %-15s %-15s %-15s\n", "Batch Size", "Total (ms)", "Texts/sec", "ms/text");
    fflush(log);

    for (int bi = 0; bi < 3; bi++) {
        int bsz = batch_sizes[bi];
        int n = total_texts;

        std::vector<double> times;
        for (int i = 0; i < iters; i++) {
            lembed_embeddings_t result = {0};
            double bt0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), n, bsz, &result);
            double bt1 = now_ms();
            double ms = bt1 - bt0;
            times.push_back(ms);
            lembed_embeddings_free(&result);
        }

        std::sort(times.begin(), times.end());
        double median_ms = times[times.size() / 2];
        double texts_per_sec = (double)n / (median_ms / 1000.0);
        double ms_per_text = median_ms / n;

        fprintf(log, "%-12d %-15.2f %-15.1f %-15.3f\n", bsz, median_ms, texts_per_sec, ms_per_text);
        fflush(log);
    }

    lembed_text_embedding_free(embedder);
    fprintf(log, "Done.\n");
    fflush(log);
    fclose(log);
    return 0;
}

