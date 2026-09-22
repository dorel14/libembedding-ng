/*
 * Mesure les copies memoire et le pooling
 */

#define LIBEMBEDDING_IMPLEMENTATION
#define LIBEMBEDDING_NO_DOWNLOAD
#include <libembedding/libembedding.h>
#include <libembedding/detail/pooling.hpp>
#include <libembedding/detail/normalize.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main() {
    printf("=== Profilage copies memoire ===\n\n");

    int bsz = 64;
    int seq_len = 12;
    int dim = 384;

    /* Simulate model output */
    size_t output_size = (size_t)bsz * seq_len * dim;
    std::vector<float> model_output(output_size);
    for (size_t i = 0; i < output_size; i++) model_output[i] = (float)(i % 100) / 100.0f;

    /* Attention mask */
    std::vector<int64_t> mask((size_t)bsz * seq_len, 1);

    /* Output buffer */
    std::vector<float> pooled((size_t)bsz * dim);

    printf("Batch: %d, seq_len: %d, dim: %d\n", bsz, seq_len, dim);
    printf("Taille output: %.2f MB\n", (double)output_size * 4 / (1024*1024));

    /* 1. Pooling mean */
    printf("\n--- 1. Pooling mean ---\n");
    {
        std::vector<double> times;
        for (int i = 0; i < 1000; i++) {
            double t0 = now_ms();
            lembed::detail::pool_mean(model_output.data(), mask.data(), bsz, seq_len, dim, 3, pooled.data());
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.4f ms (mediane sur 1000)\n", median(times));
    }

    /* 2. L2 Normalization */
    printf("\n--- 2. L2 Normalization ---\n");
    {
        std::vector<double> times;
        for (int i = 0; i < 1000; i++) {
            double t0 = now_ms();
            lembed::detail::l2_normalize(pooled.data(), bsz, dim);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.4f ms (mediane sur 1000)\n", median(times));
    }

    /* 3. Copie finale vers output */
    printf("\n--- 3. Copie memoire ---\n");
    {
        std::vector<float> output((size_t)bsz * dim);
        std::vector<double> times;
        for (int i = 0; i < 1000; i++) {
            double t0 = now_ms();
            memcpy(output.data(), pooled.data(), (size_t)bsz * dim * sizeof(float));
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.4f ms (mediane sur 1000)\n", median(times));
    }

    /* 4. Total pooling + norm + copie */
    printf("\n--- 4. Total post-processing ---\n");
    {
        std::vector<double> times;
        for (int i = 0; i < 1000; i++) {
            double t0 = now_ms();
            lembed::detail::pool_mean(model_output.data(), mask.data(), bsz, seq_len, dim, 3, pooled.data());
            lembed::detail::l2_normalize(pooled.data(), bsz, dim);
            std::vector<float> output((size_t)bsz * dim);
            memcpy(output.data(), pooled.data(), (size_t)bsz * dim * sizeof(float));
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("%.4f ms (mediane sur 1000)\n", median(times));
    }

    /* 5. Test avec differents batch sizes */
    printf("\n--- 5. Scaling post-processing ---\n");
    int batches[] = {1, 8, 32, 64, 128, 256};
    for (int bi = 0; bi < 6; bi++) {
        int b = batches[bi];
        size_t out_sz = (size_t)b * seq_len * dim;
        std::vector<float> out(out_sz, 0.5f);
        std::vector<int64_t> m((size_t)b * seq_len, 1);
        std::vector<float> p((size_t)b * dim);

        std::vector<double> times;
        for (int i = 0; i < 500; i++) {
            double t0 = now_ms();
            lembed::detail::pool_mean(out.data(), m.data(), b, seq_len, dim, 3, p.data());
            lembed::detail::l2_normalize(p.data(), b, dim);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }
        printf("batch=%-5d: %.3f ms total = %.4f ms/texte\n", b, median(times), median(times)/b);
    }

    printf("\n=== Termine ===\n");
    return 0;
}
