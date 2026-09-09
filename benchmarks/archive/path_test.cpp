/* Test with create_from_path */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>

int main() {
    fprintf(stderr, "Step 1\n");

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.num_threads = 1;
    opts.pooling = LEMBED_POOLING_CLS;
    opts.dim = 384;

    fprintf(stderr, "Step 2: Creating embedder from path...\n");

    const char* model_path = "C:/Users/david/.cache/libembedding/models--Xenova-bge-small-en-v1.5";
    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_path(model_path, &opts, &embedder);

    fprintf(stderr, "Step 3: Status=%d\n", s);

    if (s != LEMBED_OK) {
        fprintf(stderr, "Error: %s\n", lembed_last_error());
        return 1;
    }

    fprintf(stderr, "Step 4: Success! dim=%d\n", lembed_text_embedding_dim(embedder));

    const char* text = "Hello world";
    lembed_embeddings_t result = {0};
    s = lembed_text_embedding_embed(embedder, &text, 1, 1, &result);
    fprintf(stderr, "Embed status=%d\n", s);

    if (s == LEMBED_OK) {
        fprintf(stderr, "First 5 values: %.4f %.4f %.4f %.4f %.4f\n",
               result.data[0], result.data[1], result.data[2], result.data[3], result.data[4]);
        lembed_embeddings_free(&result);
    }

    lembed_text_embedding_free(embedder);
    fprintf(stderr, "Done\n");
    return 0;
}
