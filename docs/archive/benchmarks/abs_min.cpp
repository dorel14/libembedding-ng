/* Absolute minimal test */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>

int main() {
    printf("Hello from minimal test!\n");
    fflush(stdout);

    lembed_text_options_t opts = lembed_text_options_default();
    printf("Options created. model=%d\n", (int)opts.model);
    fflush(stdout);

    return 0;
}
