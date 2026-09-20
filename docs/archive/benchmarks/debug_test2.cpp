/* Debug test with stderr */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>

int main() {
    fprintf(stderr, "Step 1: Creating options\n");

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 1;
    opts.offline = 1;
    opts.show_download_progress = 0;

    fprintf(stderr, "Step 2: Options created\n");

    fprintf(stderr, "Step 3: Creating embedder...\n");

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    fprintf(stderr, "Step 4: Status=%d\n", s);

    if (s != LEMBED_OK) {
        fprintf(stderr, "Error: %s\n", lembed_last_error());
        return 1;
    }

    fprintf(stderr, "Step 5: Success!\n");

    lembed_text_embedding_free(embedder);
    fprintf(stderr, "Step 6: Done\n");
    return 0;
}
