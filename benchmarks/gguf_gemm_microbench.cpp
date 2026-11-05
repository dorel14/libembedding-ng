/*
 * libembedding - GGUF Q8_0 GEMM microbenchmark
 *
 * Measures isolated ggml Q8_0 weight by F32 activation matrix products.
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
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
    int64_t m;
    int64_t k;
};

struct Statistics {
    double minimum = 0.0;
    double median = 0.0;
    double maximum = 0.0;
    double stdev = 0.0;
};

struct Measurements {
    std::vector<double> milliseconds[3];
};

constexpr int kThreadCounts[] = {1, 2, 4};
constexpr int64_t kBatchSizes[] = {32, 64, 80, 128, 256, 512};
constexpr Shape kShapes[] = {
    {768, 768},
    {3072, 768},
    {768, 3072},
    {30522, 768},
};
constexpr int kWarmups = 2;
constexpr int kRepetitions = 5;
constexpr int kAllShapes = -1;
constexpr int64_t kAllBatchSizes = 0;

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
        "[--shape-index <0-3>] [--n <32|64|80|128|256|512>] "
        "[--threads <1|2|4>] [--cpumask <mask>]\n"
        "  --warmups <n>       Untimed executions per thread count (default: %d)\n"
        "  --repetitions <n>   Timed paired executions (default: %d)\n"
        "  --shape-index <n>   One shape: 0=768x768, 1=3072x768, "
        "2=768x3072, 3=30522x768\n"
        "  --n <n>             One activation column count (default: all)\n"
        "  --threads <n>       One worker count (default: benchmark all)\n"
        "  --cpumask <mask>    Strict worker affinity mask (hex such as 0x55)\n",
        executable, kWarmups, kRepetitions);
}

bool parse_options(int argc, char** argv, int& warmups, int& repetitions,
                   int& shape_index, int64_t& selected_n,
                   int& selected_threads, uint64_t& selected_cpumask) {
    warmups = kWarmups;
    repetitions = kRepetitions;
    shape_index = kAllShapes;
    selected_n = kAllBatchSizes;
    selected_threads = 0;
    selected_cpumask = 0;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", option);
            print_usage(argv[0]);
            return false;
        }
        int parsed = 0;
        const char* value = argv[++i];
        const std::string option_name(option);
        const int minimum =
            (option_name == "--warmups" || option_name == "--shape-index")
                ? 0 : 1;
        if (option_name == "--cpumask") {
            if (value[0] == '-') {
                std::fprintf(stderr, "Invalid CPU mask: %s\n", value);
                print_usage(argv[0]);
                return false;
            }
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed_mask =
                std::strtoull(value, &end, 0);
            if (!value[0] || !end || *end != '\0' || errno == ERANGE) {
                std::fprintf(stderr, "Invalid CPU mask: %s\n", value);
                print_usage(argv[0]);
                return false;
            }
            selected_cpumask = (uint64_t)parsed_mask;
            continue;
        }
        if (!parse_int(value, minimum, parsed)) {
            std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
            print_usage(argv[0]);
            return false;
        }
        if (option_name == "--warmups") {
            warmups = parsed;
        } else if (option_name == "--repetitions") {
            repetitions = parsed;
        } else if (option_name == "--shape-index" && parsed <= 3) {
            shape_index = parsed;
        } else if (option_name == "--n") {
            for (int64_t candidate : kBatchSizes) {
                if (candidate == parsed) selected_n = candidate;
            }
            if (selected_n == kAllBatchSizes) {
                std::fprintf(stderr, "Unsupported N value: %s\n", value);
                print_usage(argv[0]);
                return false;
            }
        } else if (option_name == "--threads") {
            for (int candidate : kThreadCounts) {
                if (candidate == parsed) selected_threads = candidate;
            }
            if (selected_threads == 0) {
                std::fprintf(stderr, "Unsupported thread count: %s\n", value);
                print_usage(argv[0]);
                return false;
            }
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", option);
            print_usage(argv[0]);
            return false;
        }
    }
    if (selected_cpumask != 0) {
        int active_cpus = 0;
        for (uint64_t mask = selected_cpumask; mask != 0; mask &= mask - 1) {
            ++active_cpus;
        }
        const int required_cpus = selected_threads == 0 ? 4 : selected_threads;
        if (active_cpus < required_cpus) {
            std::fprintf(stderr,
                         "CPU mask has %d processors; at least %d are required\n",
                         active_cpus, required_cpus);
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

bool run_shape(const Shape& shape, int64_t n, int warmups, int repetitions,
               int selected_threads, uint64_t selected_cpumask) {
    const size_t weight_floats = (size_t)shape.m * (size_t)shape.k;
    const size_t activation_floats = (size_t)shape.k * (size_t)n;
    const size_t output_floats = (size_t)shape.m * (size_t)n;
    const size_t tensor_bytes =
        ggml_row_size(GGML_TYPE_Q8_0, shape.k) * (size_t)shape.m +
        (activation_floats + output_floats) * sizeof(float);
    const size_t context_bytes = tensor_bytes + 64u * 1024u * 1024u;

    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ERROR: ggml_init failed for M=%lld K=%lld N=%lld\n",
                     (long long)shape.m, (long long)shape.k, (long long)n);
        return false;
    }

    ggml_tensor* weights = ggml_new_tensor_2d(
        ctx, GGML_TYPE_Q8_0, shape.k, shape.m);
    ggml_tensor* activations = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, shape.k, n);
    if (!weights || !activations || !weights->data || !activations->data) {
        std::fprintf(stderr, "ERROR: tensor allocation failed for M=%lld K=%lld N=%lld\n",
                     (long long)shape.m, (long long)shape.k, (long long)n);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> source_weights(weight_floats);
    uint32_t random_state = 0x4c454d42u;
    for (float& value : source_weights) value = random_value(random_state);
    float* activation_data = static_cast<float*>(activations->data);
    for (size_t i = 0; i < activation_floats; ++i) {
        activation_data[i] = random_value(random_state);
    }

    const size_t quantized_bytes = ggml_quantize_chunk(
        GGML_TYPE_Q8_0, source_weights.data(), weights->data, 0,
        shape.m, shape.k, nullptr);
    if (quantized_bytes != ggml_nbytes(weights)) {
        std::fprintf(stderr,
                     "ERROR: Q8_0 quantization wrote %zu bytes, expected %zu\n",
                     quantized_bytes, ggml_nbytes(weights));
        ggml_free(ctx);
        return false;
    }

    ggml_tensor* output = ggml_mul_mat(ctx, weights, activations);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    if (!output || !graph) {
        std::fprintf(stderr, "ERROR: graph construction failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_build_forward_expand(graph, output);

    ggml_threadpool_t threadpools[3] = {};
    ggml_cplan plans[3] = {};
    for (int slot = 0; slot < 3; ++slot) {
        if (selected_threads != 0 &&
            kThreadCounts[slot] != selected_threads) continue;
        ggml_threadpool_params thread_params =
            ggml_threadpool_params_default(kThreadCounts[slot]);
        thread_params.paused = true;
        if (selected_cpumask != 0) {
            for (int cpu = 0; cpu < 64; ++cpu) {
                thread_params.cpumask[cpu] =
                    (selected_cpumask & (UINT64_C(1) << cpu)) != 0;
            }
            thread_params.strict_cpu = true;
        }
        threadpools[slot] = ggml_threadpool_new(&thread_params);
        if (!threadpools[slot]) {
            std::fprintf(stderr, "ERROR: failed to create %d-thread ggml pool\n",
                         kThreadCounts[slot]);
            for (ggml_threadpool_t pool : threadpools) {
                if (pool) ggml_threadpool_free(pool);
            }
            ggml_free(ctx);
            return false;
        }

        plans[slot] = ggml_graph_plan(
            graph, kThreadCounts[slot], threadpools[slot]);
        if (plans[slot].work_size > 0) {
            plans[slot].work_data = static_cast<uint8_t*>(
                ggml_new_buffer(ctx, plans[slot].work_size));
            if (!plans[slot].work_data) {
                std::fprintf(stderr, "ERROR: failed to allocate ggml work buffer\n");
                for (ggml_threadpool_t pool : threadpools) {
                    if (pool) ggml_threadpool_free(pool);
                }
                ggml_free(ctx);
                return false;
            }
        }
    }

    Measurements measurements;
    for (int repetition = -warmups; repetition < repetitions; ++repetition) {
        for (int order = 0; order < 3; ++order) {
            const int actual_slot = (repetition & 1) ? 2 - order : order;
            if (selected_threads != 0 &&
                kThreadCounts[actual_slot] != selected_threads) continue;
            const int threads = kThreadCounts[actual_slot];
            ggml_threadpool_resume(threadpools[actual_slot]);
            const auto start = Clock::now();
            const ggml_status status =
                ggml_graph_compute(graph, &plans[actual_slot]);
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
            if (status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr,
                             "ERROR: ggml execution failed for M=%lld K=%lld N=%lld "
                             "threads=%d (status=%d)\n",
                             (long long)shape.m, (long long)shape.k,
                             (long long)n, threads, (int)status);
                for (ggml_threadpool_t pool : threadpools) {
                    if (pool) ggml_threadpool_free(pool);
                }
                ggml_free(ctx);
                return false;
            }
            if (repetition >= 0) {
                measurements.milliseconds[actual_slot].push_back(elapsed_ms);
            }
            ggml_threadpool_pause(threadpools[actual_slot]);
        }
    }

    const double operations =
        2.0 * (double)shape.m * (double)shape.k * (double)n;
    for (int slot = 0; slot < 3; ++slot) {
        if (selected_threads != 0 &&
            kThreadCounts[slot] != selected_threads) continue;
        const Statistics stats = statistics(measurements.milliseconds[slot]);
        const double gflops = operations / (stats.median * 1.0e6);
        std::printf("%5lld %5lld %5lld %7d %10.3f %10.3f %10.3f "
                    "%9.2f %10.3f\n",
                    (long long)shape.m, (long long)shape.k, (long long)n,
                    kThreadCounts[slot], stats.median, stats.minimum,
                    stats.maximum, gflops, stats.stdev);
    }

    for (ggml_threadpool_t pool : threadpools) {
        if (pool) ggml_threadpool_free(pool);
    }
    ggml_free(ctx);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    int warmups = 0;
    int repetitions = 0;
    int shape_index = kAllShapes;
    int64_t selected_n = kAllBatchSizes;
    int selected_threads = 0;
    uint64_t selected_cpumask = 0;
    if (!parse_options(argc, argv, warmups, repetitions, shape_index,
                       selected_n, selected_threads, selected_cpumask)) return 2;

    std::printf("GGML build: %s; VNNI=%s; persistent thread pools\n",
                ggml_version(),
                ggml_cpu_has_avx512_vnni() ? "available" : "unavailable");
    std::printf("Synthetic uniform Q8_0 weights x F32 activations; "
                "shapes are MxK, N is activation columns\n");
    std::printf("Warmups/thread count: %d; repetitions: %d%s; CPU mask: 0x%llx%s\n",
                warmups, repetitions,
                selected_threads == 0 ? " (all thread counts)" : "",
                (unsigned long long)selected_cpumask,
                selected_cpumask == 0 ? " (default affinity)" : " (strict)");
    std::printf("%5s %5s %5s %7s %10s %10s %10s %9s %10s\n",
                "M", "K", "N", "Threads", "Median ms", "Min ms",
                "Max ms", "GFLOPS", "Stdev ms");

    for (size_t i = 0; i < sizeof(kShapes) / sizeof(kShapes[0]); ++i) {
        if (shape_index != kAllShapes && (int)i != shape_index) continue;
        for (int64_t n : kBatchSizes) {
            if (selected_n != kAllBatchSizes && n != selected_n) continue;
            if (!run_shape(kShapes[i], n, warmups, repetitions,
                           selected_threads, selected_cpumask)) return 1;
        }
    }
    return 0;
}
