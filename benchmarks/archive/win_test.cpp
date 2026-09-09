/* Test with Windows API */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <windows.h>

void log_msg(const char* msg) {
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD written;
    WriteFile(h, msg, (DWORD)strlen(msg), &written, NULL);
}

int main() {
    log_msg("Step 1\n");

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.num_threads = 1;

    log_msg("Step 2\n");

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    log_msg("Step 3\n");

    if (s != LEMBED_OK) {
        char buf[256];
        sprintf(buf, "Error: %s\n", lembed_last_error());
        log_msg(buf);
        return 1;
    }

    log_msg("Step 4: Success!\n");

    lembed_text_embedding_free(embedder);
    log_msg("Done\n");
    return 0;
}
