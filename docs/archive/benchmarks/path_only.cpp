/* Minimal path test */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>

int main() {
    fprintf(stderr, "Starting...\n");

    const char* model_path = "C:/Users/david/.cache/libembedding/models--Xenova-bge-small-en-v1.5";
    fprintf(stderr, "Path: %s\n", model_path);

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.num_threads = 1;
    opts.pooling = LEMBED_POOLING_MEAN;
    opts.dim = 384;

    fprintf(stderr, "Creating embedder...\n");

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_path(model_path, &opts, &embedder);

    fprintf(stderr, "Status: %d, embedder: %p\n", s, (void*)embedder);

    if (s != LEMBED_OK) {
        fprintf(stderr, "Error: %s\n", lembed_last_error());
        return 1;
    }

    fprintf(stderr, "Success! dim=%d\n", lembed_text_embedding_dim(embedder));

    lembed_text_embedding_free(embedder);
    fprintf(stderr, "Done\n");
    return 0;
}
