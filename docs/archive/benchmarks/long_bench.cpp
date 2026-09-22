/*
 * Benchmark textes longs (512 tokens)
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}

std::string make_text(int n_tokens) {
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

void embed_task(lembed_text_embedding_t* emb, const std::vector<std::string>& texts) {
    std::vector<const char*> ct;
    for (auto& t : texts) ct.push_back(t.c_str());
    lembed_embeddings_t r = {0};
    lembed_text_embedding_embed(emb, ct.data(), (int)ct.size(), (int)ct.size(), &r);
    lembed_embeddings_free(&r);
}

int main() {
    printf("=== Benchmark Textes Longs (512 tokens) ===\n\n");

    int ntok = 128;
    int ntexts = 32;

    std::vector<std::string> texts;
    for (int i = 0; i < ntexts; i++)
        texts.push_back(make_text(ntok));

    printf("%d textes de %d tokens chacun\n\n", ntexts, ntok);

    struct Test { const char* name; lembed_text_model_t model; };
    Test tests[] = {
        {"MiniLM-L6-v2-Q", LEMBED_TEXT_ALL_MINILM_L6_V2_Q},
        {"BGE-small-en  ", LEMBED_TEXT_BGE_SMALL_EN_V15},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    printf("%-20s %-12s %-12s\n", "Modele", "1x4", "8x1");
    printf("------------------------------------\n");

    for (int ti = 0; ti < ntests; ti++) {
        printf("%-20s", tests[ti].name);

        /* 1x4 */
        {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = tests[ti].model;
            opts.num_threads = 4;
            opts.offline = 1;
            opts.show_download_progress = 0;
            lembed_text_embedding_t* emb = nullptr;
            lembed_text_embedding_create(&opts, &emb);

            for (int i = 0; i < 3; i++) embed_task(emb, texts);

            std::vector<double> tps;
            for (int iter = 0; iter < 2; iter++) {
                double t0 = now_ms();
                embed_task(emb, texts);
                double t1 = now_ms();
                tps.push_back((double)ntexts / ((t1-t0)/1000.0));
            }
            printf(" %-12.0f", mean(tps));
            lembed_text_embedding_free(emb);
        }

        /* 8x1 */
        {
            lembed_text_options_t opts = lembed_text_options_default();
            opts.model = tests[ti].model;
            opts.num_threads = 1;
            opts.offline = 1;
            opts.show_download_progress = 0;

            const int NSess = 8;
            std::vector<lembed_text_embedding_t*> emb(NSess);
            for (int i = 0; i < NSess; i++)
                lembed_text_embedding_create(&opts, &emb[i]);

            int per = ntexts / NSess;
            std::vector<std::vector<std::string>> subsets(NSess);
            for (int j = 0; j < NSess; j++)
                for (int k = 0; k < per; k++)
                    subsets[j].push_back(texts[j*per+k]);

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < NSess; j++)
                    embed_task(emb[j], subsets[j]);

            std::vector<double> tps;
            for (int iter = 0; iter < 2; iter++) {
                double t0 = now_ms();
                std::vector<std::thread> th;
                for (int j = 0; j < NSess; j++)
                    th.emplace_back(embed_task, emb[j], std::cref(subsets[j]));
                for (auto& t : th) t.join();
                double t1 = now_ms();
                tps.push_back((double)ntexts / ((t1-t0)/1000.0));
            }
            printf(" %-12.0f", mean(tps));

            for (int i = 0; i < NSess; i++) lembed_text_embedding_free(emb[i]);
        }

        printf(" docs/s\n");
    }

    printf("\n=== Termine ===\n");
    return 0;
}
