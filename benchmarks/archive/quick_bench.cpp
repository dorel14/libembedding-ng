/* Quick performance test using cached model - with debug output */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

int main() {
    FILE* log = fopen("bench_log.txt", "w");
    if (!log) return 1;

    fprintf(log, "Starting...\n");
    fflush(log);

    /* Use cached bge-small-en-v1.5 model */
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 0; /* auto */
    opts.offline = 1;
    opts.show_download_progress = 0;

    fprintf(log, "Creating embedder (model=%d)...\n", (int)opts.model);
    fflush(log);

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    fprintf(log, "Status: %d, embedder: %p\n", s, (void*)embedder);
    fflush(log);

    if (s != LEMBED_OK) {
        fprintf(log, "Error: %s\n", lembed_last_error());
        fflush(log);
        fclose(log);
        return 1;
    }

    fprintf(log, "Model loaded OK. dim=%d\n", lembed_text_embedding_dim(embedder));
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

    /* Prepare C-string pointers */
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
    fprintf(log, "%-12s %-15s %-15s\n", "Batch Size", "Total (ms)", "Texts/sec");
    fflush(log);

    for (int bi = 0; bi < 3; bi++) {
        int bsz = batch_sizes[bi];
        int n = total_texts;

        std::vector<double> times;
        for (int i = 0; i < iters; i++) {
            lembed_embeddings_t result = {0};
            auto t0 = std::chrono::high_resolution_clock::now();
            lembed_text_embedding_embed(embedder, texts.data(), n, bsz, &result);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            times.push_back(ms);
            lembed_embeddings_free(&result);
        }

        std::sort(times.begin(), times.end());
        double median_ms = times[times.size() / 2];
        double texts_per_sec = (double)n / (median_ms / 1000.0);

        fprintf(log, "%-12d %-15.2f %-15.1f\n", bsz, median_ms, texts_per_sec);
        fflush(log);
    }

    lembed_text_embedding_free(embedder);
    fprintf(log, "Done.\n");
    fflush(log);
    fclose(log);
    return 0;
}
