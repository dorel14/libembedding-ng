/*
 * Matrice benchmark : longueur tokens x modele
 * Teste les performances selon la longueur du texte
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

/* Genere un texte d'environ N tokens (mots) */
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

Result benchmark_model(lembed_text_model_t model, int nthreads,
                       const std::vector<std::string>& texts, int niter = 5) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = model;
    opts.num_threads = nthreads;
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* emb = nullptr;
    if (lembed_text_embedding_create(&opts, &emb) != LEMBED_OK) {
        fprintf(stderr, "Erreur: %s\n", lembed_last_error());
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

Result benchmark_parallel(lembed_text_model_t model, int nsess, int nthreads,
                          const std::vector<std::string>& texts, int niter = 5) {
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = model;
    opts.num_threads = nthreads;
    opts.offline = 1;
    opts.show_download_progress = 0;

    std::vector<lembed_text_embedding_t*> emb(nsess);
    for (int i = 0; i < nsess; i++) {
        if (lembed_text_embedding_create(&opts, &emb[i]) != LEMBED_OK) {
            fprintf(stderr, "Erreur: %s\n", lembed_last_error());
            return {0, 0, 0};
        }
    }

    int n = texts.size();
    int per = n / nsess;

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < nsess; j++) {
            lembed_embeddings_t w = {0};
            std::vector<const char*> ct;
            for (int k = 0; k < per && (j*per+k) < n; k++)
                ct.push_back(texts[j*per+k].c_str());
            if (!ct.empty())
                lembed_text_embedding_embed(emb[j], ct.data(), (int)ct.size(), (int)ct.size(), &w);
            lembed_embeddings_free(&w);
        }
    }

    /* Benchmark */
    std::vector<double> tps, ms;
    for (int i = 0; i < niter; i++) {
        double t0 = now_ms();
        std::vector<std::thread> th;
        for (int j = 0; j < nsess; j++) {
            th.emplace_back([&, j]() {
                int off = j * per;
                int cnt = (j == nsess-1) ? n - off : per;
                std::vector<const char*> ct;
                for (int k = 0; k < cnt; k++) ct.push_back(texts[off+k].c_str());
                lembed_embeddings_t r = {0};
                lembed_text_embedding_embed(emb[j], ct.data(), cnt, cnt, &r);
                lembed_embeddings_free(&r);
            });
        }
        for (auto& t : th) t.join();
        double t1 = now_ms();
        ms.push_back(t1 - t0);
        tps.push_back((double)n / ((t1-t0)/1000.0));
    }

    for (int i = 0; i < nsess; i++) lembed_text_embedding_free(emb[i]);

    Result r;
    r.mean_tps = mean(tps);
    r.std_tps = stddev(tps, r.mean_tps);
    r.mean_ms = mean(ms);
    return r;
}

int main() {
    printf("=== Matrice Longueur x Modele ===\n");
    printf("batch=64, 5 iterations par point\n\n");

    /* Longueurs de texte a tester (en tokens/mots approx) */
    int token_counts[] = {16, 64, 128};
    int nlengths = sizeof(token_counts) / sizeof(token_counts[0]);

    /* Modeles */
    struct Test {
        const char* name;
        lembed_text_model_t model;
    };
    Test tests[] = {
        {"MiniLM-L6-v2-Q", LEMBED_TEXT_ALL_MINILM_L6_V2_Q},
        {"BGE-small-en  ", LEMBED_TEXT_BGE_SMALL_EN_V15},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    /* Header */
    printf("%-18s", "Modele");
    for (int li = 0; li < nlengths; li++)
        printf(" %6d tok", token_counts[li]);
    printf("\n");
    printf("------------------");
    for (int li = 0; li < nlengths; li++)
        printf(" ----------");
    printf("\n");

    /* Matrice 8 sessions x 1 thread */
    printf("\n--- Configuration: 8 sessions x 1 thread ---\n");
    for (int ti = 0; ti < ntests; ti++) {
        printf("%-18s", tests[ti].name);
        for (int li = 0; li < nlengths; li++) {
            int ntok = token_counts[li];
            int ntexts = 32;

            /* Genere les textes */
            std::vector<std::string> texts;
            for (int i = 0; i < ntexts; i++)
                texts.push_back(make_text(ntok));

            Result r = benchmark_parallel(tests[ti].model, 8, 1, texts, 3);
            printf(" %8.0f", r.mean_tps);
        }
        printf("\n");
    }

    printf("\n=== Termine ===\n");
    return 0;
}
