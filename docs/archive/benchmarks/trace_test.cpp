/* Debug trace test */
#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <cstdio>
#include <cstring>
#include <windows.h>

void dbg(const char* msg) {
    OutputDebugStringA(msg);
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD written;
    WriteFile(h, msg, (DWORD)strlen(msg), &written, NULL);
}

int main() {
    dbg("Step 1: Starting\n");

    lembed_text_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.model = LEMBED_TEXT_BGE_SMALL_EN_V15;
    opts.num_threads = 1;
    opts.offline = 1;
    opts.show_download_progress = 0;

    dbg("Step 2: Options created\n");

    /* Check if model file exists */
    const char* cache_path = "C:/Users/david/.cache/libembedding/models--Xenova-bge-small-en-v1.5/onnx/model.onnx";
    WIN32_FILE_ATTRIBUTE_DATA attrs;
    BOOL exists = GetFileAttributesExA(cache_path, GetFileExInfoStandard, &attrs);
    char buf[256];
    sprintf(buf, "Step 3: Model file exists=%d\n", exists);
    dbg(buf);

    dbg("Step 4: Creating embedder...\n");

    lembed_text_embedding_t* embedder = nullptr;
    lembed_status_t s = lembed_text_embedding_create(&opts, &embedder);

    sprintf(buf, "Step 5: Status=%d\n", s);
    dbg(buf);

    if (s != LEMBED_OK) {
        sprintf(buf, "Error: %s\n", lembed_last_error());
        dbg(buf);
        return 1;
    }

    dbg("Step 6: Success!\n");

    lembed_text_embedding_free(embedder);
    dbg("Done\n");
    return 0;
}
