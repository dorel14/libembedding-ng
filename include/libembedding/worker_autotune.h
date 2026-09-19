/*
 * libembedding - worker_autotune.h
 * Auto-tuning for llama.cpp workers/sessions
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_WORKER_AUTOTUNE_H
#define LIBEMBEDDING_WORKER_AUTOTUNE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int optimal_workers;
    int optimal_threads;
    int physical_cores;
    int logical_cores;
} lembed_worker_config_t;

lembed_worker_config_t lembed_detect_optimal_workers(void);
int lembed_recommended_workers_for_model(const char* model_path);

#ifdef __cplusplus
}
#endif

#endif /* LIBEMBEDDING_WORKER_AUTOTUNE_H */
