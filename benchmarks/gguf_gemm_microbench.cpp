/*
 * libembedding - GGUF GEMM microbenchmark
 *
 * Measures isolated ggml weight by F32 activation matrix products for the
 * weight types a GGUF model can carry (Q8_0, F16, F32). Every weight type and
 * thread count of one shape runs in the same repetition, in alternating
 * order, so thermal drift does not favour one configuration.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
    int64_t m;
    int64_t k;
};

struct WeightType {
    ggml_type type;
    const char* name;
};

struct Statistics {
    double minimum = 0.0;
    double median = 0.0;
    double maximum = 0.0;
    double stdev = 0.0;
};

constexpr int kThreadCounts[] = {1, 2, 4, 8};
constexpr int kThreadSlots = sizeof(kThreadCounts) / sizeof(kThreadCounts[0]);
constexpr int64_t kBatchSizes[] = {16, 32, 64, 80, 128, 256, 512};
constexpr Shape kShapes[] = {
    {768, 768},
    {3072, 768},
    {768, 3072},
    {30522, 768},
};
constexpr int kShapeCount = sizeof(kShapes) / sizeof(kShapes[0]);
constexpr WeightType kWeightTypes[] = {
    {GGML_TYPE_Q8_0, "q8_0"},
    {GGML_TYPE_F16, "f16"},
    {GGML_TYPE_F32, "f32"},
};
constexpr int kTypeCount = sizeof(kWeightTypes) / sizeof(kWeightTypes[0]);
constexpr int kF32Index = 2;
constexpr int kWarmups = 2;
constexpr int kRepetitions = 5;
constexpr int kAll = -1;

struct Options {
    int warmups = kWarmups;
    int repetitions = kRepetitions;
    int shape_index = kAll;
    int type_index = kAll;
    int64_t n = 0;          // 0 = every batch size
    int threads = 0;        // 0 = every thread count
    uint64_t cpumask = 0;   // 0 = default affinity
};

bool thread_selected(const Options& options, int slot) {
    return options.threads == 0 || kThreadCounts[slot] == options.threads;
}

bool type_selected(const Options& options, int index) {
    return options.type_index == kAll || options.type_index == index;
}

uint32_t next_random(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random_value(uint32_t& state) {
    const int32_t signed_value = (int32_t)(next_random(state) & 0xffffu) - 32768;
    return (float)signed_value / 32768.0f;
}

Statistics statistics(std::vector<double> values) {
    Statistics result;
    if (values.empty()) return result;

    std::sort(values.begin(), values.end());
    result.minimum = values.front();
    result.maximum = values.back();
    result.median = values[values.size() / 2];

    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        (double)values.size();
    double variance = 0.0;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    result.stdev = std::sqrt(variance / (double)values.size());
    return result;
}

bool parse_int(const char* text, int minimum, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end != '\0' || parsed < minimum || parsed > 1000) {
        return false;
    }
    value = (int)parsed;
    return true;
}

void print_usage(const char* executable) {
    std::fprintf(stderr,
        "Usage: %s [--warmups <n>] [--repetitions <n>] "
        "[--shape-index <0-3>] [--type <q8_0|f16|f32>] "
        "[--n <16|32|64|80|128|256|512>] [--threads <1|2|4|8>] "
        "[--cpumask <mask>]\n"
        "  --warmups <n>       Untimed executions per configuration (default: %d)\n"
        "  --repetitions <n>   Timed alternating executions (default: %d)\n"
        "  --shape-index <n>   One shape: 0=768x768, 1=3072x768, "
        "2=768x3072, 3=30522x768\n"
        "  --type <name>       One weight type (default: all; the relative\n"
        "                      error column needs f32 in the run)\n"
        "  --n <n>             One activation column count (default: all)\n"
        "  --threads <n>       One worker count (default: all)\n"
        "  --cpumask <mask>    Strict worker affinity mask (hex such as 0x55)\n",
        executable, kWarmups, kRepetitions);
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", option.c_str());
            return false;
        }
        const char* value = argv[++i];

        if (option == "--cpumask") {
            errno = 0;
            char* end = nullptr;
            const unsigned long long mask = std::strtoull(value, &end, 0);
            if (value[0] == '-' || !value[0] || !end || *end != '\0' ||
                errno == ERANGE) {
                std::fprintf(stderr, "Invalid CPU mask: %s\n", value);
                return false;
            }
            options.cpumask = (uint64_t)mask;
            continue;
        }
        if (option == "--type") {
            for (int t = 0; t < kTypeCount; ++t) {
                if (kWeightTypes[t].name == std::string(value)) {
                    options.type_index = t;
                }
            }
            if (options.type_index == kAll) {
                std::fprintf(stderr, "Unsupported weight type: %s\n", value);
                return false;
            }
            continue;
        }

        const int minimum =
            (option == "--warmups" || option == "--shape-index") ? 0 : 1;
        int parsed = 0;
        if (!parse_int(value, minimum, parsed)) {
            std::fprintf(stderr, "Invalid value for %s: %s\n", option.c_str(), value);
            return false;
        }
        if (option == "--warmups") {
            options.warmups = parsed;
        } else if (option == "--repetitions") {
            options.repetitions = parsed;
        } else if (option == "--shape-index") {
            if (parsed >= kShapeCount) {
                std::fprintf(stderr, "Unsupported shape index: %s\n", value);
                return false;
            }
            options.shape_index = parsed;
        } else if (option == "--n") {
            for (int64_t candidate : kBatchSizes) {
                if (candidate == parsed) options.n = candidate;
            }
            if (options.n == 0) {
                std::fprintf(stderr, "Unsupported N value: %s\n", value);
                return false;
            }
        } else if (option == "--threads") {
            for (int candidate : kThreadCounts) {
                if (candidate == parsed) options.threads = candidate;
            }
            if (options.threads == 0) {
                std::fprintf(stderr, "Unsupported thread count: %s\n", value);
                return false;
            }
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", option.c_str());
            return false;
        }
    }
    if (options.cpumask != 0) {
        int active_cpus = 0;
        for (uint64_t mask = options.cpumask; mask != 0; mask &= mask - 1) {
            ++active_cpus;
        }
        const int required_cpus = options.threads == 0
            ? kThreadCounts[kThreadSlots - 1] : options.threads;
        if (active_cpus < required_cpus) {
            std::fprintf(stderr,
                         "CPU mask has %d processors; at least %d are required\n",
                         active_cpus, required_cpus);
            return false;
        }
    }
    return true;
}

// Owns the ggml context and thread pools of one shape run.
struct Resources {
    ggml_context* ctx = nullptr;
    ggml_threadpool_t pools[kThreadSlots] = {};

    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() {
        for (ggml_threadpool_t pool : pools) {
            if (pool) ggml_threadpool_free(pool);
        }
        if (ctx) ggml_free(ctx);
    }
};

// ||y - reference|| / ||reference|| over the M x N output.
double relative_error(const ggml_tensor* output, const ggml_tensor* reference) {
    const float* y = static_cast<const float*>(output->data);
    const float* r = static_cast<const float*>(reference->data);
    const int64_t count = ggml_nelements(reference);
    double difference = 0.0;
    double norm = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        const double d = (double)y[i] - (double)r[i];
        difference += d * d;
        norm += (double)r[i] * (double)r[i];
    }
    return norm > 0.0 ? std::sqrt(difference / norm) : 0.0;
}

bool run_shape(const Shape& shape, int64_t n, const Options& options) {
    const size_t weight_floats = (size_t)shape.m * (size_t)shape.k;
    const size_t activation_floats = (size_t)shape.k * (size_t)n;
    const size_t output_floats = (size_t)shape.m * (size_t)n;

    size_t context_bytes = activation_floats * sizeof(float) +
                           ggml_tensor_overhead() + 16u * 1024u * 1024u;
    for (int t = 0; t < kTypeCount; ++t) {
        if (!type_selected(options, t)) continue;
        context_bytes += ggml_row_size(kWeightTypes[t].type, shape.k) *
                             (size_t)shape.m +
                         output_floats * sizeof(float) +
                         2 * ggml_tensor_overhead() + ggml_graph_overhead();
    }

    Resources resources;
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    resources.ctx = ggml_init(params);
    ggml_context* ctx = resources.ctx;
    if (!ctx) {
        std::fprintf(stderr, "ERROR: ggml_init failed for M=%lld K=%lld N=%lld\n",
                     (long long)shape.m, (long long)shape.k, (long long)n);
        return false;
    }

    // Same seed and draw order for every type: the weights differ only by
    // their storage format.
    std::vector<float> source_weights(weight_floats);
    uint32_t random_state = 0x4c454d42u;
    for (float& value : source_weights) value = random_value(random_state);

    ggml_tensor* activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.k, n);
    if (!activations || !activations->data) {
        std::fprintf(stderr, "ERROR: activation allocation failed\n");
        return false;
    }
    float* activation_data = static_cast<float*>(activations->data);
    for (size_t i = 0; i < activation_floats; ++i) {
        activation_data[i] = random_value(random_state);
    }

    ggml_tensor* outputs[kTypeCount] = {};
    ggml_cgraph* graphs[kTypeCount] = {};
    for (int t = 0; t < kTypeCount; ++t) {
        if (!type_selected(options, t)) continue;
        const ggml_type type = kWeightTypes[t].type;
        ggml_tensor* weights = ggml_new_tensor_2d(ctx, type, shape.k, shape.m);
        if (!weights || !weights->data) {
            std::fprintf(stderr, "ERROR: %s weight allocation failed\n",
                         kWeightTypes[t].name);
            return false;
        }
        const size_t written = ggml_quantize_chunk(
            type, source_weights.data(), weights->data, 0, shape.m, shape.k,
            nullptr);
        if (written != ggml_nbytes(weights)) {
            std::fprintf(stderr, "ERROR: %s conversion wrote %zu bytes, expected %zu\n",
                         kWeightTypes[t].name, written, ggml_nbytes(weights));
            return false;
        }
        outputs[t] = ggml_mul_mat(ctx, weights, activations);
        graphs[t] = ggml_new_graph(ctx);
        if (!outputs[t] || !graphs[t]) {
            std::fprintf(stderr, "ERROR: graph construction failed\n");
            return false;
        }
        ggml_build_forward_expand(graphs[t], outputs[t]);
    }

    for (int slot = 0; slot < kThreadSlots; ++slot) {
        if (!thread_selected(options, slot)) continue;
        ggml_threadpool_params thread_params =
            ggml_threadpool_params_default(kThreadCounts[slot]);
        thread_params.paused = true;
        if (options.cpumask != 0) {
            for (int cpu = 0; cpu < 64; ++cpu) {
                thread_params.cpumask[cpu] =
                    (options.cpumask & (UINT64_C(1) << cpu)) != 0;
            }
            thread_params.strict_cpu = true;
        }
        resources.pools[slot] = ggml_threadpool_new(&thread_params);
        if (!resources.pools[slot]) {
            std::fprintf(stderr, "ERROR: failed to create %d-thread ggml pool\n",
                         kThreadCounts[slot]);
            return false;
        }
    }

    // One plan and work buffer per (type, thread count); runs alternate their
    // order between repetitions.
    ggml_cplan plans[kTypeCount][kThreadSlots] = {};
    std::vector<uint8_t> work[kTypeCount][kThreadSlots];
    std::vector<std::pair<int, int>> runs;
    for (int t = 0; t < kTypeCount; ++t) {
        if (!type_selected(options, t)) continue;
        for (int slot = 0; slot < kThreadSlots; ++slot) {
            if (!thread_selected(options, slot)) continue;
            plans[t][slot] = ggml_graph_plan(graphs[t], kThreadCounts[slot],
                                             resources.pools[slot]);
            if (plans[t][slot].work_size > 0) {
                work[t][slot].resize(plans[t][slot].work_size);
                plans[t][slot].work_data = work[t][slot].data();
            }
            runs.emplace_back(t, slot);
        }
    }

    std::vector<double> milliseconds[kTypeCount][kThreadSlots];
    for (int repetition = -options.warmups; repetition < options.repetitions;
         ++repetition) {
        for (size_t order = 0; order < runs.size(); ++order) {
            const size_t index = (repetition & 1) ? runs.size() - 1 - order : order;
            const int t = runs[index].first;
            const int slot = runs[index].second;
            ggml_threadpool_resume(resources.pools[slot]);
            const auto start = Clock::now();
            const ggml_status status = ggml_graph_compute(graphs[t], &plans[t][slot]);
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
            ggml_threadpool_pause(resources.pools[slot]);
            if (status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr,
                             "ERROR: ggml execution failed for %s M=%lld K=%lld "
                             "N=%lld threads=%d (status=%d)\n",
                             kWeightTypes[t].name, (long long)shape.m,
                             (long long)shape.k, (long long)n,
                             kThreadCounts[slot], (int)status);
                return false;
            }
            if (repetition >= 0) milliseconds[t][slot].push_back(elapsed_ms);
        }
    }

    const double operations = 2.0 * (double)shape.m * (double)shape.k * (double)n;
    for (const auto& run : runs) {
        const int t = run.first;
        const int slot = run.second;
        const Statistics stats = statistics(milliseconds[t][slot]);
        const double gflops = operations / (stats.median * 1.0e6);
        const double us_per_column = stats.median * 1000.0 / (double)n;
        char error[16] = "-";
        if (type_selected(options, kF32Index)) {
            std::snprintf(error, sizeof(error), "%.2e",
                          relative_error(outputs[t], outputs[kF32Index]));
        }
        std::printf("%-5s %5lld %5lld %5lld %7d %10.3f %10.3f %10.3f "
                    "%9.2f %10.3f %9.2f %9s\n",
                    kWeightTypes[t].name, (long long)shape.m, (long long)shape.k,
                    (long long)n, kThreadCounts[slot], stats.median,
                    stats.minimum, stats.maximum, gflops, stats.stdev,
                    us_per_column, error);
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    std::printf("GGML build: %s; AVX2=%d AVX512=%d AVX512_VNNI=%d AVX_VNNI=%d; "
                "persistent thread pools\n",
                ggml_version(), ggml_cpu_has_avx2(), ggml_cpu_has_avx512(),
                ggml_cpu_has_avx512_vnni(), ggml_cpu_has_avx_vnni());
    std::printf("Synthetic uniform weights x F32 activations; shapes are MxK, "
                "N is activation columns\n");
    std::printf("Warmups: %d; repetitions: %d (types and thread counts "
                "alternate); CPU mask: 0x%llx%s\n",
                options.warmups, options.repetitions,
                (unsigned long long)options.cpumask,
                options.cpumask == 0 ? " (default affinity)" : " (strict)");
    std::printf("%-5s %5s %5s %5s %7s %10s %10s %10s %9s %10s %9s %9s\n",
                "Type", "M", "K", "N", "Threads", "Median ms", "Min ms",
                "Max ms", "GFLOPS", "Stdev ms", "us/col", "RelErr");

    for (int i = 0; i < kShapeCount; ++i) {
        if (options.shape_index != kAll && i != options.shape_index) continue;
        for (int64_t n : kBatchSizes) {
            if (options.n != 0 && n != options.n) continue;
            if (!run_shape(kShapes[i], n, options)) return 1;
        }
    }
    return 0;
}
