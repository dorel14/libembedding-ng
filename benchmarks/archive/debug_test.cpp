/* Debug test */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>

int main() {
    printf("Step 1: Creating options\n");
    fflush(stdout);

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 1;
    opts.offline = 1;
    opts.show_download_progress = 0;

    printf("Step 2: Options created. model=%d, offline=%d\n", (int)opts.model, opts.offline);
    fflush(stdout);

    printf("Step 3: Creating embedder...\n");
    fflush(stdout);

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    printf("Step 4: Status=%d, embedder=%p\n", s, (void*)embedder);
    fflush(stdout);

    if (s != LEMBED_OK) {
        printf("Error: %s\n", lembed_last_error());
        fflush(stdout);
        return 1;
    }

    printf("Step 5: Success! dim=%d\n", lembed_text_embedding_dim(embedder));
    fflush(stdout);

    lembed_text_embedding_free(embedder);
    printf("Step 6: Done\n");
    fflush(stdout);
    return 0;
}
