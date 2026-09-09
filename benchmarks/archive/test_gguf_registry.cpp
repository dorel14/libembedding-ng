/*
 * Test the GGUF model registry
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>
#include <cstdio>

int main() {
    fprintf(stderr, "=== GGUF Model Registry Test ===\n\n");

    /* List all recommended GGUF models */
    const lembed_gguf_model_info_t* models = nullptr;
    int count = 0;
    lembed_status_t s = lembed_list_gguf_models(&models, &count);

    if (s != LEMBED_OK) {
        fprintf(stderr, "Error: %s\n", lembed_last_error());
        return 1;
    }

    fprintf(stderr, "Recommended GGUF models (%d):\n\n", count);
    fprintf(stderr, "%-20s %-8s %-6s %-8s %-8s %-6s %s\n",
            "Name", "Dim", "Params", "SizeMB", "MTEB", "Sess", "Description");
    fprintf(stderr, "----------------------------------------------------------------------------------------\n");

    for (int i = 0; i < count; i++) {
        fprintf(stderr, "%-20s %-8d %-6d %-8d %-8.1f %-6d %s\n",
                models[i].name, models[i].dim, models[i].params_m,
                models[i].file_size_mb, models[i].quality_mteb,
                models[i].recommended_sessions, models[i].description);
    }

    /* Test find */
    fprintf(stderr, "\n--- Find test ---\n");
    auto* found = lembed_find_gguf_model("snowflake-xs");
    if (found) {
        fprintf(stderr, "Found 'snowflake-xs': %s (MTEB %.1f, %d MB)\n",
                found->name, found->quality_mteb, found->file_size_mb);
    }

    /* Test default */
    auto* def = lembed_default_gguf_model();
    if (def) {
        fprintf(stderr, "\nDefault model: %s\n", def->name);
        fprintf(stderr, "  URL: %s\n", def->gguf_url);
        fprintf(stderr, "  Dim: %d, Params: %dM, Size: %dMB\n",
                def->dim, def->params_m, def->file_size_mb);
        fprintf(stderr, "  MTEB: %.1f, Sessions: %d\n",
                def->quality_mteb, def->recommended_sessions);
    }

    return 0;
}
