/* Test without model loading */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>

int main() {
    fprintf(stderr, "Step 1: Testing without model loading\n");

    /* Just test the options default */
    lembed_text_options_t opts = lembed_text_options_default();
    fprintf(stderr, "Step 2: model=%d\n", (int)opts.model);

    /* Test model registry */
    lembed_model_info_t info;
    lembed_status_t s = lembed_get_text_model_info(LEMBED_TEXT_BGE_SMALL_EN_V15, &info);
    fprintf(stderr, "Step 3: Status=%d\n", s);
    if (s == LEMBED_OK) {
        fprintf(stderr, "Model: %s, dim=%d\n", info.model_name, info.dim);
    }

    fprintf(stderr, "Done\n");
    return 0;
}
