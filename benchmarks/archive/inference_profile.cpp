/*
 * Mesure l'inference ONNX pure :
 * - Run() seul
 * - Pooling seul
 * - Copies memoire
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <libembedding/detail/onnx_session_impl.hpp>
#include <libembedding/detail/pooling.hpp>
#include <libembedding/detail/normalize.hpp>
#include <libembedding/detail/tokenizer_impl.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main() {
    printf("=== Profilage inference ONNX pure ===\n\n");

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

    printf("Modele: %s (dim=%d)\n", lembed_text_embedding_model_name(embedder),
           lembed_text_embedding_dim(embedder));

    /* Tokenize once to get input data */
    int bsz = 64;
    int seq_len = 32;
    size_t total = (size_t)bsz * seq_len;

    std::vector<int64_t> ids(total);
    std::vector<int64_t> mask(total);
    std::vector<int64_t> types(total);

    /* Create some tokenized data */
    lembed::detail::TokenizerWrapper tok;
    tok.load_from_file("C:/Users/david/.cache/libembedding/models--Xenova-bge-small-en-v1.5/tokenizer.json", 512);
    std::vector<std::string> texts;
    for (int i = 0; i < bsz; i++) {
        texts.push_back("The quick brown fox jumps over the lazy dog.");
    }
    tok.encode_batch_flat(texts, ids.data(), mask.data(), types.data(), &seq_len, 1);

    printf("Batch: %d textes, seq_len=%d\n", bsz, seq_len);

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        lembed_embeddings_t w = {0};
        const char* dummy = "test";
        lembed_text_embedding_embed(embedder, &dummy, 1, 1, &w);
        lembed_embeddings_free(&w);
    }

    /* 1. Full pipeline */
    printf("\n--- 1. Pipeline complet ---\n");
    {
        std::vector<const char*> text_ptrs;
        for (int i = 0; i < bsz; i++) text_ptrs.push_back("The quick brown fox jumps over the lazy dog.");

        std::vector<double> times;
        for (int i = 0; i < 20; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, text_ptrs.data(), bsz, bsz, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }
        printf("%.2f ms total = %.3f ms/texte\n", median(times), median(times)/bsz);
    }

    /* 2. Inference only (via internal session) - we need to access private members */
    /* Instead, let's measure the difference between full pipeline and just tokenization */

    /* 3. Tokenization time for the same batch */
    printf("\n--- 2. Tokenisation seule ---\n");
    {
        std::vector<double> times;
        for (int i = 0; i < 100; i++) {
            int sl = 0;
            double t0 = now_ms();
            tok.encode_batch_flat(texts, ids.data(), mask.data(), types.data(), &sl, 1);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.3f ms total = %.4f ms/texte\n", median(times), median(times)/bsz);
    }

    /* 4. Difference = inference + pooling + copies */
    printf("\n--- 3. Inference + pooling (par difference) ---\n");
    printf("Si pipeline = X ms et tokenisation = Y ms\n");
    printf("Alors inference+pooling = X - Y ms\n");

    /* 5. Test with different output tensor sizes */
    printf("\n--- 4. Taille du tensor de sortie ---\n");
    printf("batch=%d, seq_len=%d, dim=384\n", bsz, seq_len);
    printf("Taille last_hidden_state: %d x %d x %d = %.1f MB (float32)\n",
           bsz, seq_len, 384, (double)bsz * seq_len * 384 * 4 / (1024*1024));
    printf("Taille pool finale: %d x %d = %.1f MB (float32)\n",
           bsz, 384, (double)bsz * 384 * 4 / (1024*1024));

    /* 6. Measure with batch=1 to see overhead */
    printf("\n--- 5. Mesures batch=1 (overhead fixe) ---\n");
    {
        const char* text = "The quick brown fox jumps over the lazy dog.";
        std::vector<double> times;
        for (int i = 0; i < 50; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, &text, 1, 1, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }
        printf("batch=1: %.2f ms total\n", median(times));
    }

    /* 7. Measure with batch=8 */
    printf("\n--- 6. Mesures batch=8 ---\n");
    {
        std::vector<const char*> text_ptrs;
        for (int i = 0; i < 8; i++) text_ptrs.push_back("The quick brown fox jumps over the lazy dog.");
        std::vector<double> times;
        for (int i = 0; i < 50; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, text_ptrs.data(), 8, 8, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }
        printf("batch=8: %.2f ms total = %.3f ms/texte\n", median(times), median(times)/8);
    }

    /* 8. Measure with batch=32 */
    printf("\n--- 7. Mesures batch=32 ---\n");
    {
        std::vector<const char*> text_ptrs;
        for (int i = 0; i < 32; i++) text_ptrs.push_back("The quick brown fox jumps over the lazy dog.");
        std::vector<double> times;
        for (int i = 0; i < 50; i++) {
            lembed_embeddings_t result = {0};
            double t0 = now_ms();
            lembed_text_embedding_embed(embedder, text_ptrs.data(), 32, 32, &result);
            double t1 = now_ms();
            times.push_back(t1 - t0);
            lembed_embeddings_free(&result);
        }
        printf("batch=32: %.2f ms total = %.3f ms/texte\n", median(times), median(times)/32);
    }

    lembed_text_embedding_free(embedder);
    printf("\n=== Termine ===\n");
    return 0;
}
