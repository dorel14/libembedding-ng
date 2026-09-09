/*
 * Mesure separate tokenisation vs inference
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <libembedding/detail/tokenizer_impl.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

int main() {
    printf("=== Tokenisation vs Inference ===\n\n");

    /* Load tokenizer only */
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_file("C:/Users/david/.cache/libembedding/models--Xenova-bge-small-en-v1.5/tokenizer.json", 512);

    printf("Tokenizer charge (WordPiece BERT)\n");

    /* Test texts */
    const char* short_text = "Hello world";
    const char* medium_text = "The quick brown fox jumps over the lazy dog. This is a test.";
    const char* long_text = "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.";

    /* Warmup */
    {
        std::vector<std::string> warmup_texts = {medium_text};
        int max_len = 512;
        std::vector<int64_t> ids(max_len);
        std::vector<int64_t> mask(max_len);
        std::vector<int64_t> types(max_len);
        for (int i = 0; i < 100; i++) {
            int seq_len = 0;
            tok.encode_batch_flat(warmup_texts, ids.data(), mask.data(), types.data(), &seq_len, 1);
        }
    }

    /* Profile single text tokenization using batch_flat */
    printf("\n--- Tokenisation seule (encode_batch_flat, bsz=1) ---\n");
    struct Test { const char* name; const char* text; };
    Test tests[] = {
        {"Court (2 mots)", short_text},
        {"Moyen (12 mots)", medium_text},
        {"Long (28 mots)", long_text},
    };
    for (int t = 0; t < 3; t++) {
        std::vector<std::string> texts = {tests[t].text};
        int max_len = 512;
        std::vector<int64_t> ids(max_len);
        std::vector<int64_t> mask(max_len);
        std::vector<int64_t> types(max_len);

        std::vector<double> times;
        for (int i = 0; i < 1000; i++) {
            double t0 = now_ms();
            int seq_len = 0;
            tok.encode_batch_flat(texts, ids.data(), mask.data(), types.data(), &seq_len, 1);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        printf("%-20s %.3f ms (mediane sur 1000)\n", tests[t].name, med);
    }

    /* Profile batch tokenization */
    printf("\n--- Tokenisation batch (encode_batch_flat, 1 thread) ---\n");
    const int batch_sizes[] = {1, 4, 16, 64, 128};
    for (int bi = 0; bi < 5; bi++) {
        int bsz = batch_sizes[bi];
        std::vector<std::string> texts;
        for (int i = 0; i < bsz; i++) texts.push_back(medium_text);

        int max_len = 512;
        std::vector<int64_t> ids(bsz * max_len);
        std::vector<int64_t> mask(bsz * max_len);
        std::vector<int64_t> types(bsz * max_len);

        std::vector<double> times;
        for (int i = 0; i < 200; i++) {
            double t0 = now_ms();
            int seq_len = 0;
            tok.encode_batch_flat(texts, ids.data(), mask.data(), types.data(), &seq_len, 1);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        printf("batch=%-5d  %.2f ms total  %.3f ms/texte  seq_len=%d\n",
               bsz, med, med / bsz, 12);
    }

    /* Profile batch tokenization parallel */
    printf("\n--- Tokenisation batch parallele (encode_batch_flat, 4 threads) ---\n");
    for (int bi = 0; bi < 5; bi++) {
        int bsz = batch_sizes[bi];
        std::vector<std::string> texts;
        for (int i = 0; i < bsz; i++) texts.push_back(medium_text);

        int max_len = 512;
        std::vector<int64_t> ids(bsz * max_len);
        std::vector<int64_t> mask(bsz * max_len);
        std::vector<int64_t> types(bsz * max_len);

        std::vector<double> times;
        for (int i = 0; i < 200; i++) {
            double t0 = now_ms();
            int seq_len = 0;
            tok.encode_batch_flat(texts, ids.data(), mask.data(), types.data(), &seq_len, 4);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        printf("batch=%-5d  %.2f ms total  %.3f ms/texte\n", bsz, med, med / bsz);
    }

    /* Compare with full pipeline */
    printf("\n--- Pipeline complet (tokenisation + inference) ---\n");
    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 4;
    opts.offline = 1;
    opts.show_download_progress = 0;

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);
    if (s != LEMBED_OK) {
        fprintf(stderr, "Erreur: %s\n", lembed_last_error());
        return 1;
    }

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        lembed_embeddings_t w = {0};
        lembed_text_embedding_embed(embedder, &medium_text, 1, 1, &w);
        lembed_embeddings_free(&w);
    }

    for (int bi = 0; bi < 5; bi++) {
        int bsz = batch_sizes[bi];
        std::vector<const char*> texts;
        for (int i = 0; i < bsz; i++) texts.push_back(medium_text);

        std::vector<double> times;
        for (int i = 0; i < 50; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, texts.data(), bsz, bsz, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }
        std::sort(times.begin(), times.end());
        double med = times[times.size() / 2];
        printf("batch=%-5d  %.2f ms total  %.3f ms/texte\n", bsz, med, med / bsz);
    }

    lembed_text_embedding_free(embedder);
    printf("\n=== Termine ===\n");
    return 0;
}
