/* Absolute minimal CRT test */
#include <cstdio>
#include <cstring>
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>

int main() {
    puts("A: before opts");
    fflush(stdout);

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.num_threads = 1;
    opts.dim = 384;

    puts("B: opts created");
    fflush(stdout);

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create_from_path(
        "C:/Users/david/.cache/libembedding/models--Xenova-bge-small-en-v1.5",
        &opts, &embedder);

    printf("C: status=%d ptr=%p\n", s, (void*)embedder);
    fflush(stdout);

    if (s != LEMBED_OK) {
        printf("Error: %s\n", lembed_last_error());
        return 1;
    }

    printf("D: dim=%d\n", lembed_text_embedding_dim(embedder));
    fflush(stdout);

    lembed_text_embedding_free(embedder);
    puts("E: done");
    return 0;
}
