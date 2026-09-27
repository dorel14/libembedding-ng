/*
 * libembedding - downloader.h
 * Model download C API
 *
 * Auteur: David Orel
 * Version: 1.8.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DOWNLOADER_H
#define LIBEMBEDDING_DOWNLOADER_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lookup additional files for a model from the registry.
 * Declaration only — implementation in detail/downloader_impl.hpp */
const char* const* lembed__get_additional_files(int model_enum, int model_type);

/* Download/ensure a text model is available in cache */
lembed_status_t lembed_ensure_text_model(
    lembed_text_model_t model,
    const char* cache_dir,      /* NULL = default */
    int show_progress,
    int offline,                /* 1 = skip downloads, cache only */
    char** model_dir_out);

lembed_status_t lembed_ensure_sparse_model(
    lembed_sparse_model_t model,
    const char* cache_dir,
    int show_progress,
    int offline,
    char** model_dir_out);

lembed_status_t lembed_ensure_image_model(
    lembed_image_model_t model,
    const char* cache_dir,
    int show_progress,
    int offline,
    char** model_dir_out);

lembed_status_t lembed_ensure_reranker_model(
    lembed_reranker_model_t model,
    const char* cache_dir,
    int show_progress,
    int offline,
    char** model_dir_out);

lembed_status_t lembed_ensure_gguf_model(
    const char* repo,
    const char* filename,
    const char* cache_dir,      /* NULL = default */
    int show_progress,
    int offline,                /* 1 = skip downloads, cache only */
    char** model_path_out);

lembed_status_t lembed_resolve_gguf_path(
    const char* name_or_path,   /* .gguf path, URL, or registry name */
    const char* cache_dir,      /* NULL = default */
    int offline,                /* 1 = skip downloads, cache only */
    char** model_path_out);     /* caller must free with lembed_free_string() */

void lembed_free_string(char* s);

/* =========================================================================
 * Model Cache Cleanup
 * ========================================================================= */

/* Clean up model cache: keep only specified models, remove others.
 * cache_dir: NULL = default cache directory
 * keep_models: NULL-terminated list of model identifiers to keep (e.g., "BAAI/bge-small-en-v1.5")
 * dry_run: 1 = only report what would be deleted, don't actually delete
 * deleted_count: output parameter for number of directories deleted (can be NULL)
 * freed_bytes: output parameter for total bytes freed (can be NULL)
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_cleanup_model_cache(
    const char* cache_dir,
    const char* const* keep_models,
    int dry_run,
    size_t* deleted_count,
    uint64_t* freed_bytes);

/* Clean up all models except the specified active one.
 * cache_dir: NULL = default cache directory
 * active_model_dir: model identifier to keep (e.g., "BAAI/bge-small-en-v1.5")
 * dry_run: 1 = only report, don't delete
 * deleted_count: output parameter for number of directories deleted (can be NULL)
 * freed_bytes: output parameter for total bytes freed (can be NULL)
 * Returns LEMBED_OK on success. */
lembed_status_t lembed_cleanup_model_cache_except(
    const char* cache_dir,
    const char* active_model_dir,
    int dry_run,
    size_t* deleted_count,
    uint64_t* freed_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ---- Implementation ---- */
#ifdef LIBEMBEDDING_IMPLEMENTATION

#include <cstdlib>
#include <cstring>
#include <string>
#include <filesystem>
#include <unordered_set>
#include <vector>
#include "detail/downloader_impl.hpp"
#include "detail/download_manager.hpp"
#include "gguf_registry.h"
#include "model_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lookup additional files for a model from the registry.
 * Returns nullptr if no additional files are needed. */
const char* const* lembed__get_additional_files(int model_enum, int model_type) {
    return lembed::detail::get_additional_files(model_enum, model_type);
}

/* Progress callback for GGUF downloads */
static void gguf_download_progress(float fraction, void* userdata) {
    lembed::detail::gguf_download_progress(fraction, userdata);
}

lembed_status_t lembed_ensure_text_model(
        lembed_text_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    try {
        std::string dir = lembed::detail::ensure_model_by_kind(
            (int)model, LEMBED__MODEL_TYPE_TEXT,
            cache_dir, show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

lembed_status_t lembed_ensure_sparse_model(
        lembed_sparse_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    try {
        std::string dir = lembed::detail::ensure_model_by_kind(
            (int)model, LEMBED__MODEL_TYPE_SPARSE,
            cache_dir, show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

#ifndef LIBEMBEDDING_NO_IMAGE
lembed_status_t lembed_ensure_image_model(
        lembed_image_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    try {
        std::string dir = lembed::detail::ensure_model_by_kind(
            (int)model, LEMBED__MODEL_TYPE_IMAGE,
            cache_dir, show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

#endif /* LIBEMBEDDING_NO_IMAGE */

lembed_status_t lembed_ensure_reranker_model(
        lembed_reranker_model_t model, const char* cache_dir,
        int show_progress, int offline, char** model_dir_out) {
    if (!model_dir_out) return LEMBED_ERROR_INVALID_ARGUMENT;
    try {
        std::string dir = lembed::detail::ensure_model_by_kind(
            (int)model, LEMBED__MODEL_TYPE_RERANKER,
            cache_dir, show_progress != 0, offline != 0);
        *model_dir_out = strdup(dir.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

lembed_status_t lembed_ensure_gguf_model(
        const char* repo, const char* filename,
        const char* cache_dir, int show_progress, int offline,
        char** model_path_out) {
    if (!repo || !repo[0] || !filename || !filename[0] || !model_path_out)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        std::string cache = lembed::detail::get_cache_dir(cache_dir);
        std::string repo_dir = lembed::detail::repo_to_dirname(repo);
        std::string model_dir = cache + "/models--" + repo_dir;
        lembed::detail::mkdirs(model_dir);

        std::string dest = model_dir + "/" + filename;

        /* Check if already cached */
        if (lembed::detail::file_exists(dest)) {
            *model_path_out = strdup(dest.c_str());
            return LEMBED_OK;
        }

        /* In offline mode, never attempt download */
        if (offline) {
            lembed::detail::set_error("GGUF model not in cache (offline mode): " + dest);
            return LEMBED_ERROR_DOWNLOAD;
        }

#ifndef LIBEMBEDDING_NO_DOWNLOAD
        /* Download the file with optional progress callback */
        lembed::detail::download_progress_fn pfn = show_progress ? &gguf_download_progress : nullptr;
        bool ok = lembed::detail::download_hf_file(repo, filename, dest, pfn, nullptr);
        if (!ok) {
            lembed::detail::set_error(std::string("Failed to download GGUF model: ") + repo + "/" + filename);
            return LEMBED_ERROR_DOWNLOAD;
        }
#else
        lembed::detail::set_error("GGUF model not in cache and downloading is disabled: " + dest);
        return LEMBED_ERROR_DOWNLOAD;
#endif

        *model_path_out = strdup(dest.c_str());
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

/* =========================================================================
 * Resolve a GGUF path/name/URL to a local cached file.
 * ========================================================================= */
lembed_status_t lembed_resolve_gguf_path(
        const char* name_or_path,
        const char* cache_dir,
        int offline,
        char** model_path_out) {
    if (!name_or_path || !name_or_path[0] || !model_path_out)
        return LEMBED_ERROR_INVALID_ARGUMENT;

    try {
        std::string input(name_or_path);

        /* Case 1: local .gguf file path */
        if (input.size() > 5) {
            std::string lower = input;
            for (auto& c : lower) c = (char)tolower(c);
            if (lower.substr(lower.size() - 5) == ".gguf" && lembed::detail::file_exists(input)) {
                *model_path_out = strdup(input.c_str());
                return LEMBED_OK;
            }
        }

        /* Case 2: URL starting with http(s) */
        if (input.find("http://") == 0 || input.find("https://") == 0) {
            /* Extract repo and filename from URL like:
             * https://huggingface.co/<repo>/resolve/main/<filename>.gguf */
            size_t repo_start = input.find("huggingface.co/");
            if (repo_start == std::string::npos) {
                lembed::detail::set_error("Unsupported GGUF URL: " + input);
                return LEMBED_ERROR_DOWNLOAD;
            }
            repo_start += strlen("huggingface.co/");
            size_t repo_end = input.find("/resolve/main/", repo_start);
            if (repo_end == std::string::npos) {
                lembed::detail::set_error("Unsupported GGUF URL: " + input);
                return LEMBED_ERROR_DOWNLOAD;
            }
            std::string repo = input.substr(repo_start, repo_end - repo_start);
            std::string filename = input.substr(repo_end + strlen("/resolve/main/"));

            return lembed_ensure_gguf_model(repo.c_str(), filename.c_str(),
                                            cache_dir, 0, offline, model_path_out);
        }

        /* Case 3: registry name or HF repo/filename shorthand */
        const lembed_gguf_model_info_t* info = lembed_find_gguf_model(input.c_str());
        if (info && info->gguf_url && info->gguf_url[0]) {
            /* Extract repo and filename from URL */
            std::string url(info->gguf_url);
            size_t repo_start = url.find("huggingface.co/");
            if (repo_start != std::string::npos) {
                repo_start += strlen("huggingface.co/");
                size_t repo_end = url.find("/resolve/main/", repo_start);
                if (repo_end != std::string::npos) {
                    std::string repo = url.substr(repo_start, repo_end - repo_start);
                    std::string filename = url.substr(repo_end + strlen("/resolve/main/"));
                    return lembed_ensure_gguf_model(repo.c_str(), filename.c_str(),
                                                    cache_dir, 0, offline, model_path_out);
                }
            }
        }

        /* Case 4: treat as HF repo shorthand "<repo>/<filename>" */
        {
            size_t slash = input.find('/');
            if (slash != std::string::npos && slash > 0 && slash < input.size() - 1) {
                std::string repo = input.substr(0, slash);
                std::string filename = input.substr(slash + 1);
                if (!filename.empty() && !repo.empty()) {
                    return lembed_ensure_gguf_model(repo.c_str(), filename.c_str(),
                                                    cache_dir, 0, offline, model_path_out);
                }
            }
        }

        lembed::detail::set_error("Cannot resolve GGUF model: " + input);
        return LEMBED_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        return LEMBED_ERROR_DOWNLOAD;
    }
}

void lembed_free_string(char* s) {
    free(s);
}

/* =========================================================================
 * Model Cache Cleanup Implementation
 * ========================================================================= */

lembed_status_t lembed_cleanup_model_cache(
        const char* cache_dir,
        const char* const* keep_models,
        int dry_run,
        size_t* deleted_count,
        uint64_t* freed_bytes) {
    if (!cache_dir && !keep_models && !dry_run) {
        /* Allow all-NULL for dry-run with defaults */
    }
    if (!keep_models) return LEMBED_ERROR_INVALID_ARGUMENT;

    size_t local_deleted = 0;
    uint64_t local_freed = 0;

    try {
        std::string cache = lembed::detail::get_cache_dir(cache_dir);
        if (!std::filesystem::is_directory(cache)) {
            /* Cache directory doesn't exist - nothing to do */
            if (deleted_count) *deleted_count = 0;
            if (freed_bytes) *freed_bytes = 0;
            return LEMBED_OK;
        }

        /* Build set of model directories to keep */
        std::unordered_set<std::string> keep_dirs;
        for (int i = 0; keep_models[i]; i++) {
            std::string repo_dir = lembed::detail::repo_to_dirname(keep_models[i]);
            keep_dirs.insert("models--" + repo_dir);
        }

        /* Also always keep tune_cache.json and any non-model directories */
        std::vector<std::string> to_delete;

        for (const auto& entry : std::filesystem::directory_iterator(cache)) {
            if (!entry.is_directory()) continue;
            std::string dir_name = entry.path().filename().string();
            if (dir_name.rfind("models--", 0) != 0) continue; /* Not a model dir */

            if (keep_dirs.find(dir_name) == keep_dirs.end()) {
                /* Calculate size before deletion */
                uint64_t dir_size = 0;
                for (const auto& f : std::filesystem::recursive_directory_iterator(entry.path())) {
                    if (f.is_regular_file()) {
                        dir_size += f.file_size();
                    }
                }
                to_delete.push_back(entry.path().string());
                local_freed += dir_size;
            }
        }

        if (dry_run) {
            /* Report what would be deleted */
            for (const auto& path : to_delete) {
                fprintf(stderr, "[cleanup] Would delete: %s\n", path.c_str());
            }
            fprintf(stderr, "[cleanup] Total: %zu directories, %llu bytes\n",
                    to_delete.size(), (unsigned long long)local_freed);
            if (deleted_count) *deleted_count = to_delete.size();
            if (freed_bytes) *freed_bytes = local_freed;
            return LEMBED_OK;
        }

        /* Actually delete */
        for (const auto& path : to_delete) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            if (!ec) local_deleted++;
        }

        if (deleted_count) *deleted_count = local_deleted;
        if (freed_bytes) *freed_bytes = local_freed;
        return LEMBED_OK;
    } catch (const std::exception& e) {
        lembed::detail::set_error(e.what());
        if (deleted_count) *deleted_count = 0;
        if (freed_bytes) *freed_bytes = 0;
        return LEMBED_ERROR_IO;
    }
}

lembed_status_t lembed_cleanup_model_cache_except(
        const char* cache_dir,
        const char* active_model_dir,
        int dry_run,
        size_t* deleted_count,
        uint64_t* freed_bytes) {
    if (!active_model_dir || !active_model_dir[0])
        return LEMBED_ERROR_INVALID_ARGUMENT;

    const char* keep_models[] = { active_model_dir, nullptr };
    return lembed_cleanup_model_cache(cache_dir, keep_models, dry_run, deleted_count, freed_bytes);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBEMBEDDING_IMPLEMENTATION */

#endif /* LIBEMBEDDING_DOWNLOADER_H */
