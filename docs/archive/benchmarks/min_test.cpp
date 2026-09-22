/* Minimal test to verify model loading works */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>

int main() {
    fprintf(stderr, "Starting...\n");
    fflush(stderr);

    lembed_text_options_t opts = lembed_text_options_default();
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 1;
    opts.offline = 1;
    opts.show_download_progress = 0;

    fprintf(stderr, "Creating embedder...\n");
    fflush(stderr);

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    fprintf(stderr, "Status: %d\n", s);
    fflush(stderr);

    if (s != LEMBED_OK) {
        fprintf(stderr, "Model load failed: %s\n", lembed_last_error());
        return 1;
    }

    fprintf(stderr, "Model loaded: %s (dim=%d)\n",
           lembed_text_embedding_model_name(embedder),
           lembed_text_embedding_dim(embedder));
    fflush(stderr);

    const char* text = "Hello world";
    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(embedder, &text, 1, 1, &result);
    fprintf(stderr, "Embed status: %d\n", s);
    fflush(stderr);

    if (s == LEMBED_OK) {
        fprintf(stderr, "Result: %d texts, dim=%d\n", result.num_embeddings, result.dim);
        fprintf(stderr, "First 5 values: %.4f %.4f %.4f %.4f %.4f\n",
               result.data[0], result.data[1], result.data[2], result.data[3], result.data[4]);
        fflush(stderr);
        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(embedder);
    fprintf(stderr, "Done.\n");
    fflush(stderr);
    return 0;
}
