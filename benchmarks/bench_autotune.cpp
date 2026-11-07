/*
 * libembedding - benchmarks/bench_autotune.cpp
 * Before/after harness for the auto-tuners.
 *
 * Answers the questions that matter for a tuner, for one task per process:
 *   1. how long does a QUICK tuning take (cache cleared first)?
 *   2. how long does a repeated call take (cache hit)?
 *   3. is the chosen configuration actually faster than the library defaults
 *      on an evaluation workload the tuner never saw?
 *   4. is the choice stable (a second cold tuning lands on an equivalent config)?
 *
 * Only the public C API is used, so the same source measures the tuners before
 * and after a refactor. One task per process keeps runtimes, thread pools and
 * allocator state from leaking between tasks.
 *
 * Usage (from the repository root, Release build):
 *   build/bin/Release/bench_autotune.exe --task text   > before_text.json
 *   build/bin/Release/bench_autotune.exe --task rerank > before_rerank.json
 *   build/bin/Release/bench_autotune.exe --task sparse > before_sparse.json
 *   build/bin/Release/bench_autotune.exe --task image  > before_image.json
 * Options: --model <code> overrides the default model of the task,
 *          --runs <n> timed evaluation runs (default 5, median reported).
 * Progress goes to stderr, the JSON report to stdout. Models must already be in
 * the local model cache: nothing is downloaded.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#include <libembedding/libembedding.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

/* ---------------------------------------------------------------------------
 * Timing helpers
 * ------------------------------------------------------------------------- */

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* Runs `fn` once as warmup, then `runs` timed times. Returns the median
 * throughput in items/s, or 0 when any call fails. */
double median_throughput(int items, int runs, const std::function<bool()>& fn) {
    if (!fn()) return 0.0;
    std::vector<double> rates;
    for (int i = 0; i < runs; i++) {
        double t0 = now_ms();
        if (!fn()) return 0.0;
        double dt = now_ms() - t0;
        if (dt > 0.0) rates.push_back(items * 1000.0 / dt);
    }
    return median(rates);
}

/* ---------------------------------------------------------------------------
 * Evaluation workloads. Deliberately different from the tuners' internal
 * corpora (sizes and lengths), so a config cannot win by over-fitting them.
 * ------------------------------------------------------------------------- */

const char* kWords[] = {
    "retrieval", "augmented", "generation", "combines", "search", "with", "language",
    "models", "dense", "vectors", "capture", "meaning", "while", "sparse", "weights",
    "keep", "exact", "terms", "rerankers", "score", "query", "document", "pairs",
    "latency", "matters", "for", "interactive", "systems", "throughput", "for", "batch",
    "indexing", "of", "large", "collections", "on", "commodity", "hardware",
};
const int kNumWords = (int)(sizeof(kWords) / sizeof(kWords[0]));

std::vector<std::string> eval_texts(int n) {
    /* Mixed lengths: 8, 24, 48, 96, 160 words, cycled. */
    static const int lengths[] = {8, 24, 48, 96, 160};
    std::vector<std::string> out;
    out.reserve(n);
    for (int i = 0; i < n; i++) {
        int len = lengths[i % 5];
        std::string s;
        for (int j = 0; j < len; j++) {
            if (j) s += ' ';
            s += kWords[(i * 11 + j * 3) % kNumWords];
        }
        out.push_back(s);
    }
    return out;
}

std::vector<const char*> c_strings(const std::vector<std::string>& v) {
    std::vector<const char*> out;
    out.reserve(v.size());
    for (const auto& s : v) out.push_back(s.c_str());
    return out;
}

/* 24-bit BMP, decodable without zlib. */
std::vector<unsigned char> eval_bmp(int w, int h, int seed) {
    int row = ((w * 3 + 3) / 4) * 4;
    int size = 54 + row * h;
    std::vector<unsigned char> b(size, 0);
    auto put32 = [&](int off, int v) {
        for (int i = 0; i < 4; i++) b[off + i] = (unsigned char)((v >> (8 * i)) & 0xFF);
    };
    b[0] = 'B'; b[1] = 'M';
    put32(2, size); put32(10, 54); put32(14, 40); put32(18, w); put32(22, h);
    b[26] = 1; b[28] = 24;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char* p = &b[54 + y * row + x * 3];
            p[0] = (unsigned char)((x * 7 + seed) & 0xFF);
            p[1] = (unsigned char)((y * 5 + seed * 3) & 0xFF);
            p[2] = (unsigned char)((x ^ y) & 0xFF);
        }
    return b;
}

/* ---------------------------------------------------------------------------
 * Configuration under evaluation (subset of lembed_unified_tuning_result_t).
 * 0 means "library default" for every field.
 * ------------------------------------------------------------------------- */

struct EvalConfig {
    int workers = 0;
    int threads = 0;
    int batch_size = 0;
    int max_tokens = 0;
    int top_k = -1;          /* -1: keep the sparse default */
    float min_weight = -1.0f;
    int storage_format = -1;
};

EvalConfig from_result(const lembed_unified_tuning_result_t& r) {
    EvalConfig c;
    c.workers = r.workers;
    c.threads = r.threads;
    c.batch_size = r.batch_size;
    c.max_tokens = r.max_tokens;
    c.top_k = r.top_k;
    c.min_weight = r.min_weight;
    c.storage_format = r.storage_format;
    return c;
}

/* ---------------------------------------------------------------------------
 * Per-task evaluation: throughput of one configuration on the eval workload.
 * ------------------------------------------------------------------------- */

double eval_text(const char* model, const EvalConfig& c, int runs) {
    int idx = lembed_resolve_text_model(model);
    if (idx < 0) return 0.0;
    int workers = c.workers > 0 ? c.workers : 1;

    lembed_text_options_t o = lembed_text_options_default();
    o.model = (lembed_text_model_t)idx;
    o.offline = 1;
    o.show_download_progress = 0;
    if (c.threads > 0) o.num_threads = c.threads;
    if (c.batch_size > 0) o.batch_size = c.batch_size;

    std::vector<lembed_text_embedding_t*> ctx(workers, nullptr);
    for (int i = 0; i < workers; i++) {
        if (lembed_text_embedding_create(&o, &ctx[i]) != LEMBED_OK) {
            for (int j = 0; j < i; j++) lembed_text_embedding_free(ctx[j]);
            return 0.0;
        }
    }

    const int n = 256;
    std::vector<std::string> texts = eval_texts(n);
    std::vector<const char*> ct = c_strings(texts);

    auto run = [&]() -> bool {
        std::vector<int> ok(workers, 0);
        std::vector<std::thread> th;
        int per = n / workers;
        for (int w = 0; w < workers; w++) {
            int start = w * per;
            int cnt = (w == workers - 1) ? n - start : per;
            th.emplace_back([&, w, start, cnt]() {
                lembed_embeddings_t r = {0};
                ok[w] = lembed_text_embedding_embed(ctx[w], ct.data() + start, cnt, 0, &r) == LEMBED_OK;
                lembed_embeddings_free(&r);
            });
        }
        for (auto& t : th) t.join();
        return std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; });
    };

    double thr = median_throughput(n, runs, run);
    for (auto* p : ctx) lembed_text_embedding_free(p);
    return thr;
}

double eval_rerank(const char* model, const EvalConfig& c, int runs) {
    int idx = lembed_resolve_reranker_model(model);
    if (idx < 0) return 0.0;
    lembed_reranker_options_t o = lembed_reranker_options_default();
    o.model = (lembed_reranker_model_t)idx;
    o.show_download_progress = 0;
    if (c.threads > 0) o.num_threads = c.threads;
    if (c.batch_size > 0) o.batch_size = c.batch_size;
    if (c.max_tokens > 0) o.max_length = c.max_tokens;

    lembed_reranker_t* ctx = nullptr;
    if (lembed_reranker_create(&o, &ctx) != LEMBED_OK) return 0.0;

    const int n = 64;
    std::vector<std::string> docs = eval_texts(n);
    std::vector<const char*> cd = c_strings(docs);
    auto run = [&]() -> bool {
        lembed_rerank_results_t r = {0};
        bool ok = lembed_reranker_rerank(ctx, "how do rerankers score documents?",
                                         cd.data(), n, 0, &r) == LEMBED_OK;
        lembed_rerank_results_free(&r);
        return ok;
    };
    double thr = median_throughput(n, runs, run);
    lembed_reranker_free(ctx);
    return thr;
}

double eval_sparse(const char* model, const EvalConfig& c, int runs) {
    int idx = lembed_resolve_sparse_model(model);
    if (idx < 0) return 0.0;
    lembed_sparse_options_t o = lembed_sparse_options_default();
    o.model = (lembed_sparse_model_t)idx;
    o.show_download_progress = 0;
    if (c.threads > 0) o.num_threads = c.threads;
    if (c.batch_size > 0) o.batch_size = c.batch_size;
    if (c.top_k >= 0) o.top_k = c.top_k;
    if (c.min_weight >= 0.0f) o.min_weight = c.min_weight;
    if (c.storage_format >= 0) o.storage_format = c.storage_format;

    lembed_sparse_embedding_ctx_t* ctx = nullptr;
    if (lembed_sparse_text_embedding_create(&o, &ctx) != LEMBED_OK) return 0.0;

    const int n = 128;
    std::vector<std::string> texts = eval_texts(n);
    std::vector<const char*> ct = c_strings(texts);
    auto run = [&]() -> bool {
        lembed_sparse_embeddings_t r = {0};
        bool ok = lembed_sparse_text_embedding_embed(ctx, ct.data(), n, 0, &o, &r) == LEMBED_OK;
        if (ok) lembed_sparse_embeddings_free(&r);
        return ok;
    };
    double thr = median_throughput(n, runs, run);
    lembed_sparse_text_embedding_free(ctx);
    return thr;
}

double eval_image(const char* model, const EvalConfig& c, int runs) {
    int idx = lembed_find_image_model_by_code(model);
    if (idx < 0) return 0.0;
    lembed_image_options_t o = lembed_image_options_default();
    o.model = (lembed_image_model_t)idx;
    o.show_download_progress = 0;
    if (c.threads > 0) o.num_threads = c.threads;
    if (c.batch_size > 0) o.batch_size = c.batch_size;

    lembed_image_embedding_t* ctx = nullptr;
    if (lembed_image_embedding_create(&o, &ctx) != LEMBED_OK) return 0.0;

    const int n = 32;
    std::vector<std::vector<unsigned char>> imgs;
    std::vector<const unsigned char*> ptrs;
    std::vector<int> sizes;
    for (int i = 0; i < n; i++) imgs.push_back(eval_bmp(256, 192, i));
    for (auto& b : imgs) { ptrs.push_back(b.data()); sizes.push_back((int)b.size()); }
    auto run = [&]() -> bool {
        lembed_embeddings_t r = {0};
        bool ok = lembed_image_embedding_embed_bytes(ctx, ptrs.data(), sizes.data(), n, 0, &r) == LEMBED_OK;
        lembed_embeddings_free(&r);
        return ok;
    };
    double thr = median_throughput(n, runs, run);
    lembed_image_embedding_free(ctx);
    return thr;
}

/* ---------------------------------------------------------------------------
 * Driver
 * ------------------------------------------------------------------------- */

struct Task {
    const char* name;
    lembed_task_t task;
    const char* default_model;
    double (*eval)(const char*, const EvalConfig&, int);
};

const Task kTasks[] = {
    {"text",   LEMBED_TASK_EMBEDDING, "sentence-transformers/all-MiniLM-L6-v2", eval_text},
    {"rerank", LEMBED_TASK_RERANKING, "jinaai/jina-reranker-v1-turbo-en",       eval_rerank},
    {"sparse", LEMBED_TASK_SPARSE,    "Qdrant/Splade_PP_en_v1",                 eval_sparse},
    {"image",  LEMBED_TASK_IMAGE,     "Qdrant/clip-ViT-B-32-vision",            eval_image},
};

void print_config(const char* key, const lembed_unified_tuning_result_t& r, bool comma) {
    std::printf("  \"%s\": {\"workers\": %d, \"threads\": %d, \"batch_size\": %d, "
                "\"max_tokens\": %d, \"top_k\": %d, \"min_weight\": %.3f, "
                "\"storage_format\": %d, \"reported_throughput\": %.2f, "
                "\"reported_p50_ms\": %.3f, \"reported_p95_ms\": %.3f, "
                "\"reported_memory_mb\": %.1f}%s\n",
                key, r.workers, r.threads, r.batch_size, r.max_tokens, r.top_k,
                r.min_weight, r.storage_format, r.throughput_docs_sec, r.latency_ms,
                r.p95_latency_ms, r.memory_mb, comma ? "," : "");
}

}  // namespace

int main(int argc, char** argv) {
    const char* task_name = nullptr;
    const char* model = nullptr;
    int runs = 5;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--task") && i + 1 < argc) task_name = argv[++i];
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
    }
    const Task* t = nullptr;
    for (const auto& k : kTasks)
        if (task_name && !std::strcmp(task_name, k.name)) t = &k;
    if (!t) {
        std::fprintf(stderr, "usage: bench_autotune --task text|rerank|sparse|image "
                             "[--model CODE] [--runs N]\n");
        return 2;
    }
    if (!model) model = t->default_model;

    /* 1. Cold QUICK tuning. */
    lembed_autotune_unified_clear_cache(t->task, model);
    lembed_unified_tuning_result_t first = {};
    double t0 = now_ms();
    lembed_status_t s1 = lembed_autotune_unified(t->task, model, LEMBED_AUTOTUNE_QUICK, &first);
    double cold_ms = now_ms() - t0;
    if (s1 != LEMBED_OK) {
        std::fprintf(stderr, "autotune failed: status=%d (%s)\n", (int)s1, lembed_last_error());
        return 1;
    }

    /* 2. Repeated call: a cache hit when the tuner caches. */
    lembed_unified_tuning_result_t again = {};
    t0 = now_ms();
    lembed_autotune_unified(t->task, model, LEMBED_AUTOTUNE_QUICK, &again);
    double warm_ms = now_ms() - t0;

    /* 4. Second cold tuning, for stability. */
    lembed_autotune_unified_clear_cache(t->task, model);
    lembed_unified_tuning_result_t second = {};
    t0 = now_ms();
    lembed_status_t s2 = lembed_autotune_unified(t->task, model, LEMBED_AUTOTUNE_QUICK, &second);
    double cold2_ms = now_ms() - t0;

    /* 3. Evaluation workload: defaults vs each chosen configuration. */
    std::fprintf(stderr, "bench_autotune: evaluating configurations (%d runs each)...\n", runs);
    double thr_default = t->eval(model, EvalConfig{}, runs);
    double thr_first = t->eval(model, from_result(first), runs);
    double thr_second = (s2 == LEMBED_OK) ? t->eval(model, from_result(second), runs) : 0.0;

    std::printf("{\n");
    std::printf("  \"task\": \"%s\",\n  \"model\": \"%s\",\n  \"mode\": \"QUICK\",\n", t->name, model);
    std::printf("  \"cold_tuning_ms\": %.1f,\n  \"repeat_call_ms\": %.2f,\n"
                "  \"second_cold_tuning_ms\": %.1f,\n", cold_ms, warm_ms, cold2_ms);
    print_config("chosen_first", first, true);
    print_config("chosen_second", second, true);
    std::printf("  \"eval_runs\": %d,\n", runs);
    std::printf("  \"eval_throughput_default\": %.2f,\n", thr_default);
    std::printf("  \"eval_throughput_first\": %.2f,\n", thr_first);
    std::printf("  \"eval_throughput_second\": %.2f,\n", thr_second);
    std::printf("  \"speedup_first_vs_default\": %.3f,\n",
                thr_default > 0 ? thr_first / thr_default : 0.0);
    std::printf("  \"stability_second_vs_first\": %.3f\n",
                thr_first > 0 ? thr_second / thr_first : 0.0);
    std::printf("}\n");
    return 0;
}
