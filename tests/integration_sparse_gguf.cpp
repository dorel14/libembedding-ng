/*
 * libembedding - sparse GGUF integration profiler
 *
 * Profiles the SPLADE GGUF runtime directly, including ggml graph execution.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_GGUF_PROFILE
#include <libembedding/error.h>
#include <libembedding/detail/gguf/gguf_sparse_session.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <ctime>
#include <sys/resource.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model;
    std::string corpus;
    int batch = 4;
    int threads = 0;
    int warmups = 2;
    int repetitions = 10;
    int top_k = 50;
    int max_length = 0;
    int max_batch_tokens = -1; /* -1 keeps the session default */
    int workers = 0;           /* 0 = session default (threads / 2) */
    bool per_text = false;     /* one embed() call per text: query latency */
};

struct Statistics {
    double minimum = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
    double stdev = 0.0;
};

struct Sample {
    double wall_ms = 0.0;
    double prune_ms = 0.0;
    double process_cpu_ms = 0.0;
    size_t nnz = 0;
    lembed::gguf::SparseProfile profile;
};

struct Stage {
    const char* name;
    double (*value)(const Sample&);
};

double cpu_time_ms() {
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        return 0.0;
    }
    ULARGE_INTEGER k{}, u{};
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;
    u.HighPart = user.dwHighDateTime;
    return (double)(k.QuadPart + u.QuadPart) / 10000.0;
#else
    return 1000.0 * (double)std::clock() / (double)CLOCKS_PER_SEC;
#endif
}

double current_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return (double)counters.WorkingSetSize / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

double peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return (double)counters.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
#elif defined(__APPLE__)
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return (double)usage.ru_maxrss / (1024.0 * 1024.0);
    }
#elif defined(__linux__)
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return (double)usage.ru_maxrss / 1024.0;
    }
#endif
    return 0.0;
}

Statistics statistics(std::vector<double> values) {
    Statistics out;
    if (values.empty()) return out;
    std::sort(values.begin(), values.end());
    out.minimum = values.front();
    out.maximum = values.back();
    out.median = (values.size() % 2 == 0)
        ? (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0
        : values[values.size() / 2];
    out.mean = std::accumulate(values.begin(), values.end(), 0.0) /
               (double)values.size();
    out.p95 = values[(size_t)std::ceil(0.95 * (double)values.size()) - 1];
    double variance = 0.0;
    for (double value : values) {
        variance += (value - out.mean) * (value - out.mean);
    }
    out.stdev = std::sqrt(variance / (double)values.size());
    return out;
}

std::vector<std::string> default_corpus() {
    return {
        "a quick brown fox jumps over a lazy dog",
        "the mitochondria is the powerhouse of the cell",
        "machine learning models need lots of data",
        "the cat sat on the mat by the door",
        "weather today is sunny and warm",
        "python is a programming language",
        "the sky is blue and the grass is green",
        "spelling errors are easy to make",
        "sparse vectors represent documents with weighted terms",
        "gguf stores model weights in a compact binary format",
        "semantic search compares text embeddings",
        "the library supports local inference on windows",
    };
}

bool load_corpus(const std::string& path, std::vector<std::string>& texts,
                 std::string& error) {
    if (path.empty()) {
        texts = default_corpus();
        return true;
    }
    std::ifstream input(path);
    if (!input) {
        error = "cannot open corpus file: " + path;
        return false;
    }
    std::string line;
    while (std::getline(input, line)) {
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const size_t last = line.find_last_not_of(" \t\r\n");
        texts.push_back(line.substr(first, last - first + 1));
    }
    if (input.bad()) {
        error = "error while reading corpus file: " + path;
        return false;
    }
    if (texts.empty()) {
        error = "corpus contains no non-empty lines: " + path;
        return false;
    }
    return true;
}

bool parse_int(const char* text, int minimum, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end != '\0' || parsed < minimum ||
        parsed > 1000000) {
        return false;
    }
    value = (int)parsed;
    return true;
}

void print_usage(const char* executable) {
    std::fprintf(stderr,
        "Usage: %s --model <model.gguf> [options]\n"
        "  --corpus <file.txt>   One document per non-empty line (default: 12 built-in texts)\n"
        "  --batch <n>           Batch size (default: 4)\n"
        "  --threads <n>         Total ggml CPU threads; 0 = every logical CPU (default: 0)\n"
        "  --warmups <n>         Untimed warmup passes (default: 2)\n"
        "  --repetitions <n>     Timed passes (default: 10)\n"
        "  --top-k <n>           Terms retained after pooling; 0 keeps all (default: 50)\n"
        "  --max-length <n>      Token limit; 0 uses model limit (default: 0)\n"
        "  --max-batch-tokens <n> Cap on batch * seq_len; 0 disables it (default: session default)\n"
        "  --workers <n>         Independent graphs sharing --threads; 0 = auto (default: 0)\n"
        "  --mode <m>            corpus: one embed() per pass over the corpus (default);\n"
        "                        per-text: one embed() per text, i.e. query latency\n",
        executable);
}

bool parse_options(int argc, char** argv, Options& options, bool& show_help) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            show_help = true;
            return false;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
        const char* value = argv[++i];
        int parsed = 0;
        if (arg == "--model") options.model = value;
        else if (arg == "--corpus") options.corpus = value;
        else if (arg == "--batch" && parse_int(value, 1, parsed)) options.batch = parsed;
        else if (arg == "--threads" && parse_int(value, 0, parsed)) options.threads = parsed;
        else if (arg == "--warmups" && parse_int(value, 0, parsed)) options.warmups = parsed;
        else if (arg == "--repetitions" && parse_int(value, 1, parsed)) options.repetitions = parsed;
        else if (arg == "--top-k" && parse_int(value, 0, parsed)) options.top_k = parsed;
        else if (arg == "--max-length" && parse_int(value, 0, parsed)) options.max_length = parsed;
        else if (arg == "--max-batch-tokens" && parse_int(value, 0, parsed)) options.max_batch_tokens = parsed;
        else if (arg == "--workers" && parse_int(value, 0, parsed)) options.workers = parsed;
        else if (arg == "--mode" && (std::string(value) == "corpus" ||
                                     std::string(value) == "per-text"))
            options.per_text = std::string(value) == "per-text";
        else {
            std::fprintf(stderr, "Invalid option or value: %s %s\n",
                         arg.c_str(), value);
            print_usage(argv[0]);
            return false;
        }
    }
    if (options.model.empty()) {
        std::fprintf(stderr, "--model is required\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

double profile_tokenization(const Sample& s) { return s.profile.tokenization_ms; }
double profile_pack(const Sample& s) { return s.profile.input_pack_ms; }
double profile_graph_build(const Sample& s) { return s.profile.graph_build_ms; }
double profile_mask(const Sample& s) { return s.profile.mask_build_ms; }
double profile_upload(const Sample& s) { return s.profile.input_upload_ms; }
double profile_graph_exec(const Sample& s) { return s.profile.graph_exec_ms; }
double profile_download(const Sample& s) { return s.profile.logits_download_ms; }
double profile_pooling(const Sample& s) { return s.profile.pooling_ms; }
double profile_pruning(const Sample& s) { return s.prune_ms; }
double profile_wall(const Sample& s) { return s.wall_ms; }
double profile_residual(const Sample& s) {
    return std::max(0.0, s.wall_ms - s.profile.tokenization_ms -
        s.profile.input_pack_ms - s.profile.graph_build_ms -
        s.profile.mask_build_ms - s.profile.input_upload_ms -
        s.profile.graph_exec_ms - s.profile.logits_download_ms -
        s.profile.pooling_ms - s.prune_ms);
}

std::vector<Stage> stages() {
    return {
        {"Tokenization", profile_tokenization},
        {"Input packing", profile_pack},
        {"Graph build + allocation", profile_graph_build},
        {"Attention mask", profile_mask},
        {"Input upload", profile_upload},
        {"GraphExec (ggml)", profile_graph_exec},
        {"Logits download", profile_download},
        {"SPLADE pooling", profile_pooling},
        {"Prune + sort", profile_pruning},
        {"Unattributed wrapper", profile_residual},
        {"End-to-end", profile_wall},
    };
}

void print_stage_table(const std::vector<Sample>& samples) {
    std::printf("\n=== Runtime stages per timed pass (ms) ===\n");
    std::printf("%-28s %10s %10s %10s %10s %10s %10s\n",
                "Stage", "Median", "Mean", "Min", "P95*", "Max", "Stdev");
    for (const Stage& stage : stages()) {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const Sample& sample : samples) values.push_back(stage.value(sample));
        const Statistics s = statistics(std::move(values));
        std::printf("%-28s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                    stage.name, s.median, s.mean, s.minimum, s.p95,
                    s.maximum, s.stdev);
    }
    std::printf("* nearest-rank empirical p95; at low repetition counts this may equal the maximum.\n");
}

size_t model_file_size(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return 0;
    const std::streampos size = input.tellg();
    return size < 0 ? 0 : (size_t)size;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    bool show_help = false;
    if (!parse_options(argc, argv, options, show_help)) {
        return show_help ? 0 : 2;
    }

    std::vector<std::string> texts;
    std::string error;
    if (!load_corpus(options.corpus, texts, error)) {
        std::fprintf(stderr, "ERROR: %s\n", error.c_str());
        return 2;
    }

    lembed::gguf::SparseSession session;
    const auto load_start = Clock::now();
    if (!session.load_from_file(options.model.c_str(), options.threads, options.workers,
                                options.max_length, error)) {
        std::fprintf(stderr, "ERROR loading model: %s\n", error.c_str());
        return 1;
    }
    if (options.max_batch_tokens >= 0) {
        session.set_max_batch_tokens(options.max_batch_tokens);
    }
    const double load_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - load_start).count();

    /* One sample = one embed() call: the whole corpus, or a single text in
     * per-text mode, where the samples are query latencies. */
    const auto run = [&](const std::vector<std::string>& call_texts) {
        session.reset_profile();
        const double cpu_start = cpu_time_ms();
        const auto wall_start = Clock::now();
        std::vector<lembed::detail::SparseResult> embeddings =
            session.embed(call_texts, options.batch);
        const auto prune_start = Clock::now();
        size_t nnz = 0;
        for (auto& embedding : embeddings) {
            lembed::detail::sparse_prune_and_sort(
                embedding, options.top_k, 0.0f, LEMBED_SPARSE_FORMAT_DICT);
            nnz += embedding.indices.size();
        }
        Sample sample;
        sample.wall_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - wall_start).count();
        sample.prune_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - prune_start).count();
        sample.process_cpu_ms = cpu_time_ms() - cpu_start;
        sample.nnz = nnz;
        sample.profile = session.profile();
        return sample;
    };
    const auto run_pass = [&](std::vector<Sample>& out) {
        if (!options.per_text) {
            out.push_back(run(texts));
            return;
        }
        for (const std::string& text : texts) out.push_back(run({text}));
    };
    const size_t texts_per_sample = options.per_text ? 1 : texts.size();

    std::vector<Sample> warmup_samples;
    for (int i = 0; i < options.warmups; ++i) run_pass(warmup_samples);

    std::vector<Sample> samples;
    samples.reserve((size_t)options.repetitions *
                    (options.per_text ? texts.size() : 1));
    for (int i = 0; i < options.repetitions; ++i) run_pass(samples);

    const std::vector<double> wall_values = [&samples]() {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const Sample& sample : samples) values.push_back(sample.wall_ms);
        return values;
    }();
    const Statistics wall = statistics(wall_values);
    const double docs_per_sec = wall.median > 0.0
        ? (double)texts_per_sample * 1000.0 / wall.median : 0.0;
    const double avg_nnz = std::accumulate(
        samples.begin(), samples.end(), 0.0,
        [](double total, const Sample& sample) {
            return total + (double)sample.nnz;
        }) / ((double)samples.size() * (double)texts_per_sample);
    const Sample& last_profile = samples.back();

    std::printf("=== Sparse GGUF runtime profiler ===\n");
    std::printf("Model: %s (%zu bytes, %.2f MiB)\n", options.model.c_str(),
                model_file_size(options.model),
                (double)model_file_size(options.model) / (1024.0 * 1024.0));
    std::printf("Corpus: %zu texts; mode=%s; batch=%d; max_batch_tokens=%d; "
                "workers=%d; threads=%d (total); top_k=%d\n",
                texts.size(), options.per_text ? "per-text" : "corpus",
                options.batch, session.max_batch_tokens(),
                session.workers(), session.total_threads(), options.top_k);
    std::printf("Warmups: %d; repetitions: %d; seq_len(max): %d; vocab: %d\n",
                options.warmups, options.repetitions,
                last_profile.profile.max_seq_len, session.vocab_size());
    std::printf("Logical processors: %u; C++: %ld; build: %s\n",
                std::thread::hardware_concurrency(), (long)__cplusplus,
#ifdef NDEBUG
                "Release/NDEBUG"
#else
                "Debug/assertions enabled"
#endif
    );
    std::printf("Model load: %.2f ms\n", load_ms);

    if (!warmup_samples.empty()) {
        const Sample& first = warmup_samples.front();
        std::printf("Warmup pass 1: %.3f ms total graph build, "
                    "%.3f ms total GraphExec, %llu nodes in last graph, "
                    "last shape=(seq=%d,batch=%d), %llu graph build(s)\n",
                    first.profile.graph_build_ms, first.profile.graph_exec_ms,
                    (unsigned long long)first.profile.graph_nodes,
                    first.profile.graph_seq_len, first.profile.graph_batch,
                    (unsigned long long)first.profile.graph_builds);
    } else if (!samples.empty()) {
        std::printf("Timed pass 1: %.3f ms total graph build, "
                    "%.3f ms total GraphExec, %llu nodes in last graph, "
                    "last shape=(seq=%d,batch=%d), %llu graph build(s)\n",
                    samples.front().profile.graph_build_ms,
                    samples.front().profile.graph_exec_ms,
                    (unsigned long long)samples.front().profile.graph_nodes,
                    samples.front().profile.graph_seq_len,
                    samples.front().profile.graph_batch,
                    (unsigned long long)samples.front().profile.graph_builds);
    }

    print_stage_table(samples);
    std::printf("\n=== End-to-end summary ===\n");
    std::printf("Latency: median=%.3f ms mean=%.3f ms min=%.3f ms "
                "p95=%.3f ms max=%.3f ms stdev=%.3f ms\n",
                wall.median, wall.mean, wall.minimum, wall.p95,
                wall.maximum, wall.stdev);
    std::printf("Throughput: %.2f documents/s; average output nnz: %.2f\n",
                docs_per_sec, avg_nnz);
    std::printf("RSS: current=%.1f MiB peak=%.1f MiB\n",
                current_rss_mb(), peak_rss_mb());
    const double cpu_sum = std::accumulate(
        samples.begin(), samples.end(), 0.0,
        [](double total, const Sample& sample) {
            return total + sample.process_cpu_ms;
        });
    const double wall_sum = std::accumulate(
        samples.begin(), samples.end(), 0.0,
        [](double total, const Sample& sample) {
            return total + sample.wall_ms;
        });
    std::printf("Process CPU / wall: %.1f%% over timed passes\n",
                wall_sum > 0.0 ? 100.0 * cpu_sum / wall_sum : 0.0);
    const uint64_t graph_builds = std::accumulate(
        samples.begin(), samples.end(), (uint64_t)0,
        [](uint64_t total, const Sample& sample) {
            return total + sample.profile.graph_builds;
        });
    const uint64_t graph_hits = std::accumulate(
        samples.begin(), samples.end(), (uint64_t)0,
        [](uint64_t total, const Sample& sample) {
            return total + sample.profile.graph_cache_hits;
        });
    const uint64_t batches = std::accumulate(
        samples.begin(), samples.end(), (uint64_t)0,
        [](uint64_t total, const Sample& sample) {
            return total + sample.profile.batches;
        });
    uint64_t real_tokens = 0, padded_tokens = 0;
    for (const Sample& sample : samples) {
        real_tokens += sample.profile.real_tokens;
        padded_tokens += sample.profile.padded_tokens;
    }
    std::printf("Timed graph batches=%llu; builds=%llu; cache hits=%llu; "
                "real tokens=%llu; padding tokens=%llu\n",
                (unsigned long long)batches, (unsigned long long)graph_builds,
                (unsigned long long)graph_hits, (unsigned long long)real_tokens,
                (unsigned long long)padded_tokens);
    std::printf("\n=== GGML operation inventory (not per-op timing) ===\n");
    session.print_graph_operations(stdout);
    return 0;
}
