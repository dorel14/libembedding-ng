#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [n_threads] [iters]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int nt = argc > 2 ? std::atoi(argv[2]) : 2;
    int iters = argc > 3 ? std::atoi(argv[3]) : 5;
    std::printf("model=%s n_threads=%d iters=%d\n", path, nt, iters);

    lembed_sparse_options_t opts = lembed_sparse_options_default();
    opts.num_threads = nt;
    opts.batch_size = 8;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    lembed_status_t s = lembed_sparse_text_embedding_create_from_gguf_path(path, &opts, &ctx);
    if (s != LEMBED_OK) {
        std::printf("FAIL: create returned %d\n", (int)s);
        return 1;
    }

    const char* texts[] = {
        "a quick brown fox jumps over a lazy dog 0",
        "the mitochondria is the powerhouse of the cell 1",
        "machine learning models need lots of data 2",
        "the cat sat on the mat by the door 3",
        "weather today is sunny and warm 4",
        "python is a programming language 5",
        "the sky is blue and the grass is green 6",
        "spelling errors are easy to make 7"
    };
    const int n = 8;

    // warmup
    for (int i = 0; i < 3; ++i) {
        lembed_sparse_embeddings_t out = {0};
        lembed_sparse_text_embedding_embed(ctx, texts, n, 8, nullptr, &out);
        lembed_sparse_embeddings_free(&out);
    }

    double best = 1e18, sum = 0;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        lembed_sparse_embeddings_t out = {0};
        lembed_sparse_text_embedding_embed(ctx, texts, n, 8, nullptr, &out);
        lembed_sparse_embeddings_free(&out);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
        sum += ms;
    }
    double mean = sum / iters;
    double docs_per_s = (mean > 0) ? (double)n / (mean / 1000.0) : 0;
    std::printf("mean=%.2fms best=%.2fms throughput=%.1f docs/s\n",
                mean, best, docs_per_s);

    lembed_sparse_text_embedding_free(ctx);
    return 0;
}