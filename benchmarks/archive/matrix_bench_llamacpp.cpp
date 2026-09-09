/*
 * llama.cpp matrix benchmark: token length x threads
 * Tests performance by text length and thread count
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}

static double stddev(const std::vector<double>& v, double m) {
    double s = 0; for (double x : v) s += (x-m)*(x-m); return std::sqrt(s/v.size());
}

/* Generate text of approximately N tokens (words) */
static std::string make_text(int n_tokens) {
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "is", "subset", "of", "artificial", "intelligence",
        "natural", "language", "processing", "enables", "computers", "to", "understand",
        "human", "language", "vector", "embeddings", "represent", "text", "as", "dense",
        "numerical", "representations", "deep", "neural", "networks", "learn", "patterns",
        "from", "data", "transformer", "models", "use", "attention", "mechanisms",
        "to", "capture", "contextual", "relationships", "between", "words", "in", "sentences"
    };
    int nwords = sizeof(words) / sizeof(words[0]);
    std::string result;
    for (int i = 0; i < n_tokens; i++) {
        if (i > 0) result += " ";
        result += words[i % nwords];
    }
    return result;
}

struct Result {
    double mean_tps, std_tps, mean_ms;
};

Result benchmark_threads(const char* model_path, int nthreads,
                         const std::vector<std::string>& texts, int niter = 5) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.num_threads = nthreads;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    if (lembed_text_embedding_create_from_gguf_path(model_path, &opts, &emb) != LEMBED_OK) {
        fprintf(stderr, "Error: %s\n", lembed_last_error());
        return {0, 0, 0};
    }

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        std::vector<const char*> ct;
        int n = std::min(4, (int)texts.size());
        for (int j = 0; j < n; j++) ct.push_back(texts[j].c_str());
        lembed_embeddings_t r = {0};
        lembed_text_embedding_embed(emb, ct.data(), n, n, &r);
        lembed_embeddings_free(&r);
    }

    /* Benchmark */
    std::vector<double> tps, ms;
    int n = texts.size();
    for (int i = 0; i < niter; i++) {
        double t0 = now_ms();
        for (int off = 0; off < n; off += 64) {
            int cnt = std::min(64, n - off);
            std::vector<const char*> ct;
            for (int j = off; j < off+cnt; j++) ct.push_back(texts[j].c_str());
            lembed_embeddings_t r = {0};
            lembed_text_embedding_embed(emb, ct.data(), cnt, cnt, &r);
            lembed_embeddings_free(&r);
        }
        double t1 = now_ms();
        ms.push_back(t1 - t0);
        tps.push_back((double)n / ((t1-t0)/1000.0));
    }

    lembed_text_embedding_free(emb);

    Result r;
    r.mean_tps = mean(tps);
    r.std_tps = stddev(tps, r.mean_tps);
    r.mean_ms = mean(ms);
    return r;
}

int main(int argc, char** argv) {
    const char* model_path = R"(C:\Users\david\.cache\libembedding\models--BAAI-bge-small-en-v1.5\bge-small-en-v1.5-Q8_0.gguf)";
    if (argc > 1) model_path = argv[1];

    printf("=== llama.cpp Matrix: Longueur x Threads ===\n");
    printf("Model: %s\n", model_path);
    printf("batch=64, 5 iterations per point\n\n");

    /* Text lengths to test (approx tokens/words) */
    int token_counts[] = {16, 64, 128};
    int nlengths = sizeof(token_counts) / sizeof(token_counts[0]);

    /* Thread counts to test */
    int thread_counts[] = {1, 2, 4, 8};
    int nthreads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    /* Header */
    printf("%-10s", "Threads");
    for (int li = 0; li < nlengths; li++)
        printf(" %10d tok", token_counts[li]);
    printf("\n");
    printf("----------");
    for (int li = 0; li < nlengths; li++)
        printf(" -------------");
    printf("\n");

    /* Matrix */
    for (int ti = 0; ti < nthreads; ti++) {
        int nt = thread_counts[ti];
        printf("%-10d", nt);
        for (int li = 0; li < nlengths; li++) {
            int ntok = token_counts[li];
            int ntexts = 32;

            std::vector<std::string> texts;
            for (int i = 0; i < ntexts; i++)
                texts.push_back(make_text(ntok));

            Result r = benchmark_threads(model_path, nt, texts, 5);
            printf(" %10.0f", r.mean_tps);
        }
        printf("\n");
    }

    printf("\n=== Termine ===\n");
    return 0;
}

