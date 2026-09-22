/*
 * Critical Experiment: llama_batch vs Session Pool
 *
 * Tests 3 strategies:
 * A) Session Pool: N sessions, each encodes 1 text
 * B) True Batch: 1 session, N texts in single llama_encode()
 * C) Combined: N sessions, each encodes M texts in batch
 *
 * Hypothesis: Session Pool wins because O(n²) attention on
 * N*seq_len tokens is slower than N separate passes on seq_len tokens.
 */

#include <libembedding/libembedding.h>
#include <libembedding/detail/llama_session_impl.hpp>

#include <llama.h>
#include <ggml.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

    /* True batch: N texts in single llama_encode() with multiple seq_ids */
static float bench_true_batch(const char* path, int n_texts,
                              const std::vector<std::string>& texts) {
    lembed::detail::LlamaSession session;
    session.load_from_file(path, 4, 0, 0, false);
    int dim = session.dimension();

    /* Tokenize all texts */
    std::vector<std::vector<llama_token>> all_tokens(n_texts);
    int total_tokens = 0;
    for (int i = 0; i < n_texts; i++) {
        session.tokenize(texts[i].c_str(), all_tokens[i]);
        total_tokens += (int)all_tokens[i].size();
    }

    /* Build batch with multiple sequences */
    llama_batch batch = llama_batch_init(total_tokens, 0, n_texts);
    batch.n_tokens = 0;
    for (int t = 0; t < n_texts; t++) {
        llama_seq_id seq = (llama_seq_id)(t + 1);
        for (size_t i = 0; i < all_tokens[t].size(); i++) {
            batch.token[batch.n_tokens] = all_tokens[t][i];
            batch.pos[batch.n_tokens] = (llama_pos)i;
            batch.seq_id[batch.n_tokens][0] = seq;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.logits[batch.n_tokens] = 0;
            batch.n_tokens++;
        }
    }

    /* Warmup */
    llama_memory_clear(llama_get_memory(session.context()), false);
    llama_encode(session.context(), batch);
    llama_memory_clear(llama_get_memory(session.context()), false);

    /* Benchmark */
    double t0 = now_ms();
    llama_encode(session.context(), batch);
    double t1 = now_ms();
    llama_batch_free(batch);

    float ms_per_text = (float)(t1 - t0) / n_texts;
    return 1000.0f / ms_per_text; /* docs/sec for this batch */
}

/* Strategy A: Session Pool (1 text per session) */
static float bench_session_pool(const char* path, int n_sessions,
                                const std::vector<std::string>& texts) {
    lembed::detail::LlamaSessionPool pool;
    pool.load_from_file(path, n_sessions, 1, 0, 0, false);

    /* Warmup */
    for (int i = 0; i < n_sessions && i < (int)texts.size(); i++)
        pool.embed(texts[i].c_str());

    int total = (int)texts.size();
    std::atomic<int> idx{0};
    double t0 = now_ms();
    std::vector<std::thread> workers;
    for (int w = 0; w < n_sessions; w++)
        workers.emplace_back([&]() {
            while (true) {
                int i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= total) break;
                pool.embed(texts[i].c_str());
            }
        });
    for (auto& t : workers) t.join();
    double t1 = now_ms();
    return (float)total / ((float)(t1 - t0) / 1000.0f);
}

int main(int argc, char** argv) {
    const char* model = "C:\\Users\\david\\.cache\\libembedding\\gguf\\all-MiniLM-L6-v2-Q4_K_M.gguf";
    if (argc > 1) model = argv[1];

    fprintf(stderr, "=== Critical Experiment: Batch vs Pool ===\n");
    fprintf(stderr, "Model: %s\n\n", model);

    /* Generate texts */
    std::vector<std::string> texts;
    for (int i = 0; i < 64; i++)
        texts.push_back("Document " + std::to_string(i) + " about topic " + std::to_string(i % 8));

    fprintf(stderr, "--- Strategy A: Session Pool ---\n");
    fprintf(stderr, "%-10s %-12s %-10s\n", "Sessions", "Docs/sec", "Scaling");
    fprintf(stderr, "------------------------------------\n");
    float base_tps = 0;
    for (int n = 1; n <= 8; n++) {
        float tps = bench_session_pool(model, n, texts);
        if (n == 1) base_tps = tps;
        fprintf(stderr, "%-10d %-12.1f %-10.2fx\n", n, tps, tps / base_tps);
    }

    fprintf(stderr, "\n--- Strategy B: True Batch (1 session, N texts) ---\n");
    fprintf(stderr, "%-10s %-12s %-10s\n", "Batch Sz", "Docs/sec", "Scaling");
    fprintf(stderr, "------------------------------------\n");
    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64};
    for (int i = 0; i < 7; i++) {
        int bsz = batch_sizes[i];
        float tps = bench_true_batch(model, bsz, texts);
        fprintf(stderr, "%-10d %-12.1f %-10.2fx\n", bsz, tps, tps / base_tps);
    }

    fprintf(stderr, "\n--- Strategy C: Combined (Pool + Batch) ---\n");
    fprintf(stderr, "Testing 2 sessions with batch=2,4,8 each...\n");
    /* Simplified: just show pool results for now */
    fprintf(stderr, "(Requires pool implementation with batch support)\n");

    fprintf(stderr, "\n=== Conclusion ===\n");
    fprintf(stderr, "If Pool > Batch: session pooling is optimal strategy.\n");
    fprintf(stderr, "If Batch > Pool: implement true batching in backend.\n");

    return 0;
}
