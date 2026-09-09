/*
 * libembedding - bench_reranker_compare.cpp
 * Benchmark: ONNX reranker vs llama.cpp reranker
 *
 * Measures:
 *   - Workers matrix: 1x4, 2x2, 4x1, 8x1 (sessions x threads)
 *   - n_batch sweep: 32, 64, 128, 256
 *   - Top-K: 20, 50, 100
 *   - Latency (ms) per query and per pair
 *   - Throughput (queries/sec)
 *   - Memory footprint (MB)
 *   - Quality metrics (nDCG@10, MRR)
 *
 * SPDX-License-Identifier: MIT
 */

#define LIBEMBEDDING_IMPLEMENTATION
#include <libembedding/libembedding.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

#ifndef LEMBED_RERANKER_MODEL_DEFAULT
#define LEMBED_RERANKER_MODEL_DEFAULT 0
#endif

/* =========================================================================
 * Synthetic benchmark data (MS MARCO-like passage reranking)
 * ========================================================================= */
static const char* QUERIES[] = {
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Explain the theory of relativity",
    "What are the benefits of exercise?",
    "How do vaccines work?",
    "What is quantum computing?",
    "Describe the water cycle",
    "What causes climate change?",
    "How does the internet work?",
    "What is DNA replication?",
    "Explain neural networks",
    "What is the speed of light?",
    "How do airplanes fly?",
    "What is the human genome?",
    "Describe plate tectonics",
    "What is artificial intelligence?",
    "How does GPS work?",
    "What are black holes?",
    "Explain the immune system",
};

static const char* PASSAGES[] = {
    "The capital of France is Paris.",
    "Paris is the largest city in France.",
    "France is a country in Western Europe.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Chlorophyll absorbs light during photosynthesis.",
    "Plants use photosynthesis to produce oxygen.",
    "Machine learning is a subset of artificial intelligence.",
    "Algorithms learn patterns from data in machine learning.",
    "Deep learning uses neural networks for machine learning.",
    "Einstein developed the theory of relativity.",
    "Relativity describes gravity as curvature of spacetime.",
    "Special relativity introduced E=mc^2.",
    "Exercise improves cardiovascular health.",
    "Regular exercise reduces stress and anxiety.",
    "Physical activity strengthens muscles and bones.",
    "Vaccines stimulate the immune system to recognize pathogens.",
    "mRNA vaccines teach cells to produce antigens.",
    "Vaccination has eradicated smallpox.",
    "Quantum computing uses qubits instead of classical bits.",
    "Qubits can exist in superposition states.",
    "Quantum algorithms can solve certain problems exponentially faster.",
    "The water cycle includes evaporation, condensation, and precipitation.",
    "Water vapor rises and cools in the atmosphere.",
    "Rain and snow return water to Earth's surface.",
    "Burning fossil fuels releases greenhouse gases.",
    "CO2 traps heat in the atmosphere.",
    "Deforestation contributes to climate change.",
    "The internet connects billions of devices worldwide.",
    "TCP/IP protocols enable internet communication.",
    "Data travels through routers and switches on the internet.",
    "DNA replication copies genetic material before cell division.",
    "Helicase unwinds the DNA double helix.",
    "DNA polymerase synthesizes new DNA strands.",
    "Neural networks are inspired by biological neurons.",
    "Backpropagation trains neural networks.",
    "Convolutional networks excel at image recognition.",
    "Light travels at approximately 299,792 kilometers per second.",
    "Nothing can travel faster than light in vacuum.",
    "The speed of light is a fundamental constant.",
    "Airplanes generate lift using wing shape and angle of attack.",
    "Bernoulli's principle explains aerodynamic lift.",
    "Jet engines provide thrust for aircraft.",
    "The human genome contains about 3 billion base pairs.",
    "DNA encodes the instructions for life.",
    "The Human Genome Project mapped all human genes.",
    "Plate tectonics explains the movement of Earth's crust.",
    "Continental drift was proposed by Alfred Wegener.",
    "Earthquakes occur at plate boundaries.",
    "AI enables machines to perform tasks requiring intelligence.",
    "Machine learning is a subset of AI.",
    "AI applications include robotics and natural language processing.",
    "GPS satellites orbit Earth and transmit timing signals.",
    "Receivers calculate position using trilateration.",
    "GPS accuracy depends on satellite geometry.",
    "Black holes have extreme gravitational pull.",
    "Event horizons mark the point of no return around black holes.",
    "Hawking radiation is theoretical emission from black holes.",
    "The immune system defends against pathogens.",
    "Antibodies recognize and neutralize antigens.",
    "T cells and B cells are key immune cells.",
};

static const int N_QUERIES = 20;
static const int TOTAL_PASSAGES = 60;

/* =========================================================================
 * Benchmark result structure
 * ========================================================================= */
struct BenchmarkResult {
    std::string backend;
    std::string config;
    int n_batch;
    int top_k;
    double avg_latency_ms;
    double p50_latency_ms;
    double p95_latency_ms;
    double p99_latency_ms;
    double throughput_qps;
    double ms_per_pair;
    size_t memory_mb;
    double ndcg_at_10;
    double mrr;
    int num_queries;
    int num_documents_per_query;
};

/* =========================================================================
 * Worker configuration
 * ========================================================================= */
struct WorkerConfig {
    int sessions;
    int threads_per_session;
    std::string label;
};

static const std::vector<WorkerConfig> WORKER_CONFIGS = {
    {1, 4, "1x4"},
    {2, 2, "2x2"},
    {4, 1, "4x1"},
    {8, 1, "8x1"},
};

static const std::vector<int> N_BATCH_VALUES = {32, 64, 128, 256};
static const std::vector<int> TOP_K_VALUES = {20, 50, 100};

/* =========================================================================
 * Timing helpers
 * ========================================================================= */
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

/* =========================================================================
 * Compute nDCG@10 from relevance judgments and reranked order
 * ========================================================================= */
static double compute_ndcg_at_10(const std::vector<int>& relevance,
                                  const lembed_rerank_result_t* results,
                                  int result_count) {
    if (result_count == 0) return 0.0;

    double dcg = 0.0;
    int limit = std::min(10, result_count);
    for (int i = 0; i < limit; i++) {
        int idx = results[i].index;
        if (idx >= 0 && idx < (int)relevance.size()) {
            double rel = relevance[idx];
            dcg += (rel > 0) ? (1.0 / std::log2(i + 2)) : 0.0;
        }
    }

    std::vector<double> ideal;
    for (int r : relevance) ideal.push_back(r);
    std::sort(ideal.begin(), ideal.end(), std::greater<double>());
    double idcg = 0.0;
    for (int i = 0; i < limit; i++) {
        idcg += (ideal[i] > 0) ? (1.0 / std::log2(i + 2)) : 0.0;
    }

    return (idcg > 0.0) ? (dcg / idcg) : 0.0;
}

/* =========================================================================
 * Compute MRR (Mean Reciprocal Rank)
 * ========================================================================= */
static double compute_mrr(const std::vector<int>& relevance,
                          const lembed_rerank_result_t* results,
                          int result_count) {
    if (result_count == 0) return 0.0;
    for (int i = 0; i < result_count; i++) {
        int idx = results[i].index;
        if (idx >= 0 && idx < (int)relevance.size() && relevance[idx] > 0) {
            return 1.0 / (i + 1);
        }
    }
    return 0.0;
}

/* =========================================================================
 * Run quality benchmark with ground truth
 * ========================================================================= */
static std::pair<double, double> measure_quality(
        lembed_reranker_t* reranker,
        const std::vector<std::string>& queries,
        const std::vector<std::string>& documents,
        const std::vector<std::vector<int>>& relevance,
        int top_k,
        int n_batch) {

    double total_ndcg = 0.0;
    double total_mrr = 0.0;
    int num_queries = (int)queries.size();

    for (int q = 0; q < num_queries; q++) {
        lembed_rerank_results_t results = {0};
        std::vector<const char*> doc_ptrs;
        doc_ptrs.reserve(documents.size());
        for (const auto& d : documents) doc_ptrs.push_back(d.c_str());

        lembed_status_t s = lembed_reranker_rerank(
            reranker, queries[q].c_str(), doc_ptrs.data(),
            (int)doc_ptrs.size(), n_batch, &results);

        if (s == LEMBED_OK && results.items) {
            total_ndcg += compute_ndcg_at_10(relevance[q], results.items, results.count);
            total_mrr += compute_mrr(relevance[q], results.items, results.count);
            lembed_rerank_results_free(&results);
        }
    }

    return {total_ndcg / num_queries, total_mrr / num_queries};
}

/* =========================================================================
 * Extract latency percentiles
 * ========================================================================= */
static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p * v.size());
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

/* =========================================================================
 * Get memory usage (Windows-specific)
 * ========================================================================= */
static size_t get_process_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (size_t)(pmc.WorkingSetSize / (1024 * 1024));
    }
#endif
    return 0;
}

/* =========================================================================
 * Benchmark a single reranker instance (sequential, for latency percentiles)
 * ========================================================================= */
static std::vector<double> benchmark_single(
        lembed_reranker_t* reranker,
        const std::vector<std::string>& queries,
        const std::vector<std::string>& documents,
        int n_batch,
        int num_iterations) {

    std::vector<double> latencies;
    latencies.reserve(queries.size() * num_iterations);

    for (int iter = 0; iter < num_iterations; iter++) {
        for (const auto& query : queries) {
            auto t_start = now_ms();
            lembed_rerank_results_t results = {0};
            std::vector<const char*> doc_ptrs;
            doc_ptrs.reserve(documents.size());
            for (const auto& d : documents) doc_ptrs.push_back(d.c_str());

            lembed_status_t s = lembed_reranker_rerank(
                reranker, query.c_str(), doc_ptrs.data(),
                (int)doc_ptrs.size(), n_batch, &results);

            double t_end = now_ms();
            if (s == LEMBED_OK && results.items) {
                latencies.push_back(t_end - t_start);
                lembed_rerank_results_free(&results);
            }
        }
    }
    return latencies;
}

/* =========================================================================
 * Benchmark worker throughput (parallel sessions)
 * ========================================================================= */
static double benchmark_workers(
        const std::vector<lembed_reranker_t*>& rerankers,
        const std::vector<std::string>& queries,
        const std::vector<std::string>& documents,
        int n_batch,
        int num_iterations) {

    int num_sessions = (int)rerankers.size();
    if (num_sessions == 0) return 0.0;

    int queries_per_session = (int)queries.size() / num_sessions;
    if (queries_per_session == 0) queries_per_session = 1;

    auto t_start = now_ms();

    for (int iter = 0; iter < num_iterations; iter++) {
        std::vector<std::thread> threads;
        std::atomic<int> total_processed(0);

        for (int s = 0; s < num_sessions; s++) {
            threads.emplace_back([&, s]() {
                int start_q = s * queries_per_session;
                int end_q = (s == num_sessions - 1) ? (int)queries.size() : start_q + queries_per_session;

                for (int q = start_q; q < end_q; q++) {
                    lembed_rerank_results_t results = {0};
                    std::vector<const char*> doc_ptrs;
                    doc_ptrs.reserve(documents.size());
                    for (const auto& d : documents) doc_ptrs.push_back(d.c_str());

                    lembed_status_t status = lembed_reranker_rerank(
                        rerankers[s], queries[q].c_str(), doc_ptrs.data(),
                        (int)doc_ptrs.size(), n_batch, &results);

                    if (status == LEMBED_OK && results.items) {
                        total_processed++;
                        lembed_rerank_results_free(&results);
                    }
                }
            });
        }

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    }

    auto t_end = now_ms();
    double total_time_ms = t_end - t_start;
    int total_queries = (int)queries.size() * num_iterations;
    return total_queries / (total_time_ms / 1000.0);
}

/* =========================================================================
 * Generate relevance judgments
 * ========================================================================= */
static std::vector<std::vector<int>> generate_relevance_judgments() {
    std::vector<std::vector<int>> relevance(N_QUERIES, std::vector<int>(TOTAL_PASSAGES, 0));
    for (int q = 0; q < N_QUERIES; q++) {
        int start = q * 3;
        for (int i = 0; i < 3 && (start + i) < TOTAL_PASSAGES; i++) {
            relevance[q][start + i] = 1;
        }
    }
    return relevance;
}

/* =========================================================================
 * Parse comma-separated integers
 * ========================================================================= */
static std::vector<int> parse_int_list(const char* arg, const std::vector<int>& defaults) {
    if (!arg) return defaults;
    std::vector<int> result;
    std::string s(arg);
    size_t start = 0;
    while (true) {
        size_t pos = s.find(',', start);
        if (pos == std::string::npos) {
            result.push_back(atoi(s.substr(start).c_str()));
            break;
        }
        result.push_back(atoi(s.substr(start, pos - start).c_str()));
        start = pos + 1;
    }
    return result;
}

/* =========================================================================
 * Main benchmark
 * ========================================================================= */
int main(int argc, char* argv[]) {
    printf("=============================================================\n");
    printf("libembedding Reranker Benchmark: ONNX vs llama.cpp\n");
    printf("=============================================================\n\n");

    const char* onnx_model = "BAAI/bge-reranker-base";
    std::vector<const char*> gguf_models;
    int num_iterations = 2;
    int warmup_iterations = 1;
    bool quick_mode = false;

    std::vector<int> top_k_values = TOP_K_VALUES;
    std::vector<int> n_batch_values = N_BATCH_VALUES;
    std::vector<WorkerConfig> worker_configs = WORKER_CONFIGS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--onnx") == 0 && i + 1 < argc) {
            onnx_model = argv[++i];
        } else if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) {
            gguf_models.push_back(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            num_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k_values = parse_int_list(argv[++i], top_k_values);
        } else if (strcmp(argv[i], "--n-batch") == 0 && i + 1 < argc) {
            n_batch_values = parse_int_list(argv[++i], n_batch_values);
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            std::string wstr = argv[++i];
            worker_configs.clear();
            size_t start = 0;
            while (true) {
                size_t pos = wstr.find(',', start);
                std::string token = (pos == std::string::npos)
                    ? wstr.substr(start)
                    : wstr.substr(start, pos - start);
                size_t xpos = token.find('x');
                if (xpos != std::string::npos) {
                    WorkerConfig wc;
                    wc.sessions = atoi(token.substr(0, xpos).c_str());
                    wc.threads_per_session = atoi(token.substr(xpos + 1).c_str());
                    wc.label = token;
                    worker_configs.push_back(wc);
                }
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = true;
            top_k_values = {20};
            n_batch_values = {32};
            worker_configs = {{1, 4, "1x4"}};
            num_iterations = 1;
            warmup_iterations = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: bench_reranker_compare [options]\n");
            printf("Options:\n");
            printf("  --onnx MODEL      ONNX model name or path\n");
            printf("  --gguf PATH       GGUF model path (repeatable)\n");
            printf("  --iterations N    Number of iterations (default: 2)\n");
            printf("  --top-k K,K,K     Top-K values (default: 20,50,100)\n");
            printf("  --n-batch B,B,B   n_batch values (default: 32,64,128,256)\n");
            printf("  --workers W,W,W   Worker configs sessionsxthreads (default: 1x4,2x2,4x1,8x1)\n");
            printf("  --quick           Reduced configs\n");
            return 0;
        }
    }

    printf("Generating benchmark data...\n");
    std::vector<std::string> queries;
    for (int i = 0; i < N_QUERIES; i++) queries.push_back(QUERIES[i]);

    std::vector<std::string> documents;
    for (int i = 0; i < TOTAL_PASSAGES; i++) documents.push_back(PASSAGES[i]);

    auto relevance = generate_relevance_judgments();
    printf("  Queries: %d\n", (int)queries.size());
    printf("  Documents per query: %d\n", TOTAL_PASSAGES);
    printf("  Worker configs: %d\n", (int)worker_configs.size());
    printf("  n_batch values: %d\n", (int)n_batch_values.size());
    printf("  Top-K values: %d\n", (int)top_k_values.size());
    printf("  Iterations: %d\n\n", num_iterations);

    std::vector<BenchmarkResult> results;

    /* =========================================================
     * ONNX Backend
     * ========================================================= */
    printf("-------------------------------------------------------------\n");
    printf("Backend 1: ONNX\n");
    printf("-------------------------------------------------------------\n");

    lembed_reranker_t* onnx_reranker = nullptr;
    {
        lembed_reranker_options_t opts = lembed_reranker_options_default();
        opts.model = LEMBED_RERANKER_MODEL_DEFAULT;
        opts.batch_size = 32;
        opts.num_threads = 4;

        printf("Loading ONNX model: %s...\n", onnx_model);

        int idx = lembed_find_reranker_model_by_code(onnx_model);
        if (idx >= 0) {
            opts.model = (lembed_reranker_model_t)idx;
        }

        lembed_status_t s = lembed_reranker_create(&opts, &onnx_reranker);
        if (s != LEMBED_OK) {
            printf("WARNING: Failed to load ONNX model: %s\n", lembed_last_error());
            printf("Skipping ONNX backend.\n\n");
        } else {
            printf("ONNX model loaded successfully.\n\n");
        }
    }

    if (onnx_reranker) {
        size_t mem_before = get_process_memory_mb();

        for (const auto& wc : worker_configs) {
            printf("  [ONNX] Creating %d session(s) for config %s (%d threads each)...\n",
                   wc.sessions, wc.label.c_str(), wc.threads_per_session);
            fflush(stdout);

            std::vector<lembed_reranker_t*> sessions;
            sessions.reserve(wc.sessions);
            for (int s = 0; s < wc.sessions; s++) {
                lembed_reranker_options_t opts = lembed_reranker_options_default();
                opts.model = LEMBED_RERANKER_MODEL_DEFAULT;
                opts.batch_size = 32;
                opts.num_threads = wc.threads_per_session;

                int idx = lembed_find_reranker_model_by_code(onnx_model);
                if (idx >= 0) opts.model = (lembed_reranker_model_t)idx;

                lembed_reranker_t* session = nullptr;
                if (lembed_reranker_create(&opts, &session) == LEMBED_OK) {
                    sessions.push_back(session);
                }
            }

            if (sessions.empty()) {
                printf("FAILED to create sessions\n");
                continue;
            }

            printf("  [ONNX] %d session(s) created.\n", (int)sessions.size());

            for (int n_batch : n_batch_values) {
                for (int top_k : top_k_values) {
                    printf("  [ONNX] config=%s n_batch=%d top_k=%d... ", wc.label.c_str(), n_batch, top_k);
                    fflush(stdout);

                    for (int w = 0; w < warmup_iterations; w++) {
                        for (const auto& query : queries) {
                            lembed_rerank_results_t r = {0};
                            std::vector<const char*> doc_ptrs;
                            for (const auto& d : documents) doc_ptrs.push_back(d.c_str());
                            lembed_reranker_rerank(sessions[0], query.c_str(), doc_ptrs.data(),
                                                   (int)doc_ptrs.size(), n_batch, &r);
                            if (r.items) lembed_rerank_results_free(&r);
                        }
                    }

                    double qps = benchmark_workers(sessions, queries, documents, n_batch, num_iterations);
                    auto latencies = benchmark_single(sessions[0], queries, documents, n_batch, num_iterations);

                    size_t mem_after = get_process_memory_mb();
                    size_t mem_used = (mem_after > mem_before) ? (mem_after - mem_before) : mem_after;

                    double total_time = 0.0;
                    for (double l : latencies) total_time += l;
                    double avg_latency = latencies.empty() ? 0.0 : total_time / latencies.size();
                    double ms_per_pair = avg_latency / documents.size();

                    BenchmarkResult res;
                    res.backend = "ONNX";
                    res.config = wc.label;
                    res.n_batch = n_batch;
                    res.top_k = top_k;
                    res.avg_latency_ms = avg_latency;
                    res.p50_latency_ms = percentile(latencies, 0.50);
                    res.p95_latency_ms = percentile(latencies, 0.95);
                    res.p99_latency_ms = percentile(latencies, 0.99);
                    res.throughput_qps = qps;
                    res.ms_per_pair = ms_per_pair;
                    res.memory_mb = mem_used;
                    res.ndcg_at_10 = 0.0;
                    res.mrr = 0.0;
                    res.num_queries = (int)queries.size();
                    res.num_documents_per_query = TOTAL_PASSAGES;
                    results.push_back(res);

                    printf("QPS=%.1f avg=%.2fms pair=%.4fms mem=%zumb\n",
                           qps, avg_latency, ms_per_pair, mem_used);
                }
            }

            for (auto* sess : sessions) lembed_reranker_free(sess);
        }

        printf("\n  Running quality test...\n");
        auto quality = measure_quality(onnx_reranker, queries, documents, relevance, 20, 32);
        for (auto& r : results) {
            if (r.backend == "ONNX") {
                r.ndcg_at_10 = quality.first;
                r.mrr = quality.second;
            }
        }
        printf("  nDCG@10=%.4f MRR=%.4f\n", quality.first, quality.second);

        lembed_reranker_free(onnx_reranker);
    }

    /* =========================================================
     * llama.cpp Backend
     * ========================================================= */
    if (!gguf_models.empty()) {
        printf("\n-------------------------------------------------------------\n");
        printf("Backend 2: llama.cpp (GGUF)\n");
        printf("-------------------------------------------------------------\n");

        for (const char* gguf_model : gguf_models) {
            printf("Loading GGUF model: %s...\n", gguf_model);
            size_t mem_before = get_process_memory_mb();

            for (const auto& wc : worker_configs) {
                printf("  [llama] Creating %d session(s) for config %s (%d threads each)...\n",
                       wc.sessions, wc.label.c_str(), wc.threads_per_session);
                fflush(stdout);

                std::vector<lembed_reranker_t*> sessions;
                sessions.reserve(wc.sessions);
                for (int s = 0; s < wc.sessions; s++) {
                    lembed_reranker_options_t opts = lembed_reranker_options_default();
                    opts.batch_size = 32;
                    opts.num_threads = wc.threads_per_session;

                    lembed_reranker_t* session = nullptr;
                    if (lembed_reranker_create_from_gguf_path(gguf_model, &opts, &session) == LEMBED_OK) {
                        sessions.push_back(session);
                    }
                }

                if (sessions.empty()) {
                    printf("  [llama] FAILED to create sessions for config %s\n", wc.label.c_str());
                    continue;
                }

                printf("  [llama] %d session(s) created.\n", (int)sessions.size());

                for (int n_batch : n_batch_values) {
                    for (int top_k : top_k_values) {
                        printf("  [llama] model=%s config=%s n_batch=%d top_k=%d... ",
                               gguf_model, wc.label.c_str(), n_batch, top_k);
                        fflush(stdout);

                        for (int w = 0; w < warmup_iterations; w++) {
                            for (const auto& query : queries) {
                                lembed_rerank_results_t r = {0};
                                std::vector<const char*> doc_ptrs;
                                for (const auto& d : documents) doc_ptrs.push_back(d.c_str());
                                lembed_reranker_rerank(sessions[0], query.c_str(), doc_ptrs.data(),
                                                       (int)doc_ptrs.size(), n_batch, &r);
                                if (r.items) lembed_rerank_results_free(&r);
                            }
                        }

                        double qps = benchmark_workers(sessions, queries, documents, n_batch, num_iterations);
                        auto latencies = benchmark_single(sessions[0], queries, documents, n_batch, num_iterations);

                        size_t mem_after = get_process_memory_mb();
                        size_t mem_used = (mem_after > mem_before) ? (mem_after - mem_before) : mem_after;

                        double total_time = 0.0;
                        for (double l : latencies) total_time += l;
                        double avg_latency = latencies.empty() ? 0.0 : total_time / latencies.size();
                        double ms_per_pair = avg_latency / documents.size();

                        BenchmarkResult res;
                        res.backend = "llama.cpp";
                        res.config = wc.label;
                        res.n_batch = n_batch;
                        res.top_k = top_k;
                        res.avg_latency_ms = avg_latency;
                        res.p50_latency_ms = percentile(latencies, 0.50);
                        res.p95_latency_ms = percentile(latencies, 0.95);
                        res.p99_latency_ms = percentile(latencies, 0.99);
                        res.throughput_qps = qps;
                        res.ms_per_pair = ms_per_pair;
                        res.memory_mb = mem_used;
                        res.ndcg_at_10 = 0.0;
                        res.mrr = 0.0;
                        res.num_queries = (int)queries.size();
                        res.num_documents_per_query = TOTAL_PASSAGES;
                        results.push_back(res);

                        printf("QPS=%.1f avg=%.2fms pair=%.4fms mem=%zumb\n",
                               qps, avg_latency, ms_per_pair, mem_used);
                    }
                }

                for (auto* sess : sessions) lembed_reranker_free(sess);
            }

            printf("\n  Running quality test for %s...\n", gguf_model);
            lembed_reranker_t* llama_session = nullptr;
            lembed_reranker_options_t qopts = lembed_reranker_options_default();
            qopts.batch_size = 32;
            qopts.num_threads = 4;
            if (lembed_reranker_create_from_gguf_path(gguf_model, &qopts, &llama_session) == LEMBED_OK) {
                auto quality = measure_quality(llama_session, queries, documents, relevance, 20, 32);
                for (auto& r : results) {
                    if (r.backend == "llama.cpp") {
                        r.ndcg_at_10 = quality.first;
                        r.mrr = quality.second;
                    }
                }
                printf("  nDCG@10=%.4f MRR=%.4f\n", quality.first, quality.second);
                lembed_reranker_free(llama_session);
            }
        }
    } else {
        printf("\n-------------------------------------------------------------\n");
        printf("Backend 2: llama.cpp (skipped - no GGUF path provided)\n");
        printf("-------------------------------------------------------------\n");
        printf("Usage: bench_reranker_compare --gguf path/to/model.gguf\n\n");
    }

    /* =========================================================
     * Summary tables
     * ========================================================= */
    printf("\n=============================================================\n");
    printf("BENCHMARK SUMMARY\n");
    printf("=============================================================\n\n");

    printf("=== Throughput (QPS) by Workers / n_batch ===\n");
    printf("%-12s %-6s %8s %8s %8s %8s %10s\n",
           "Backend", "Config", "n_batch", "Top-K", "QPS", "ms/pair", "Mem(MB)");
    printf("%-12s %-6s %8s %8s %8s %8s %10s\n",
           "-------", "------", "-------", "-----", "-----", "--------", "-------");

    for (const auto& r : results) {
        printf("%-12s %-6s %8d %8d %8.1f %8.4f %10zu\n",
               r.backend.c_str(), r.config.c_str(), r.n_batch, r.top_k,
               r.throughput_qps, r.ms_per_pair, r.memory_mb);
    }

    printf("\n=== Latency Percentiles (single session) ===\n");
    printf("%-12s %-6s %8s %8s %10s %10s %10s %10s\n",
           "Backend", "Config", "n_batch", "Top-K", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)");
    printf("%-12s %-6s %8s %8s %10s %10s %10s %10s\n",
           "-------", "------", "-------", "-----", "--------", "--------", "--------", "--------");

    for (const auto& r : results) {
        printf("%-12s %-6s %8d %8d %10.2f %10.2f %10.2f %10.2f\n",
               r.backend.c_str(), r.config.c_str(), r.n_batch, r.top_k,
               r.avg_latency_ms, r.p50_latency_ms, r.p95_latency_ms, r.p99_latency_ms);
    }

    printf("\n=== Quality ===\n");
    printf("%-12s %8s %8s %10s\n", "Backend", "nDCG@10", "MRR", "Memory(MB)");
    printf("%-12s %8s %8s %10s\n", "-------", "--------", "---", "----------");
    for (const auto& r : results) {
        if (r.ndcg_at_10 > 0.0) {
            printf("%-12s %8.4f %8.4f %10zu\n",
                   r.backend.c_str(), r.ndcg_at_10, r.mrr, r.memory_mb);
        }
    }

    printf("\n=============================================================\n");
    printf("Recommendations\n");
    printf("=============================================================\n\n");

    if (results.size() >= 2) {
        double best_onnx_qps = 0.0;
        double best_llama_qps = 0.0;
        size_t onnx_mem = 0;
        size_t llama_mem = 0;
        double onnx_ndcg = 0.0;
        double llama_ndcg = 0.0;

        for (const auto& r : results) {
            if (r.backend == "ONNX") {
                if (r.throughput_qps > best_onnx_qps) best_onnx_qps = r.throughput_qps;
                if (r.ndcg_at_10 > onnx_ndcg) onnx_ndcg = r.ndcg_at_10;
                if (r.memory_mb > 0 && (onnx_mem == 0 || r.memory_mb < onnx_mem)) onnx_mem = r.memory_mb;
            } else if (r.backend == "llama.cpp") {
                if (r.throughput_qps > best_llama_qps) best_llama_qps = r.throughput_qps;
                if (r.ndcg_at_10 > llama_ndcg) llama_ndcg = r.ndcg_at_10;
                if (r.memory_mb > 0 && (llama_mem == 0 || r.memory_mb < llama_mem)) llama_mem = r.memory_mb;
            }
        }

        printf("Best ONNX throughput: %.1f QPS (memory: %zu MB, nDCG@10: %.4f)\n",
               best_onnx_qps, onnx_mem, onnx_ndcg);
        printf("Best llama.cpp throughput: %.1f QPS (memory: %zu MB, nDCG@10: %.4f)\n",
               best_llama_qps, llama_mem, llama_ndcg);

        if (best_llama_qps > 0 && best_onnx_qps > 0) {
            printf("Speedup: %.2fx\n", best_onnx_qps / best_llama_qps);
        }
        if (llama_mem > 0 && onnx_mem > 0) {
            printf("Memory ratio: %.2fx (llama.cpp uses %.2fx less memory)\n",
                   (float)onnx_mem / (float)llama_mem, (float)onnx_mem / (float)llama_mem);
        }
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
