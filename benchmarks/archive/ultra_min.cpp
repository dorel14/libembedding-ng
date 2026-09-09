/* Ultra minimal test */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#define LIBEMBEDDING_NO_IMAGE
#include <libembedding/libembedding.h>
#include <cstdio>

int main() {
    fprintf(stderr, "STEP 1: Starting\n");
    fflush(stderr);

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 1;
    opts.offline = 1;

    fprintf(stderr, "STEP 2: Creating embedder\n");
    fflush(stderr);

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    fprintf(stderr, "STEP 3: Status=%d, embedder=%p\n", s, (void*)embedder);
    fflush(stderr);

    if (s != LEMBED_OK) {
        fprintf(stderr, "ERROR: %s\n", lembed_last_error());
        fflush(stderr);
        return 1;
    }

    fprintf(stderr, "STEP 4: Model loaded OK\n");
    fflush(stderr);

    lembed_text_embedding_free(embedder);
    fprintf(stderr, "STEP 5: Done\n");
    fflush(stderr);
    return 0;
}
