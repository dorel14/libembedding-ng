/*
 * libembedding - detail/ggml_cpu_backend.hpp
 * The ggml CPU backend, reached through ggml's backend registry.
 *
 * Why the registry instead of ggml-cpu.h
 * --------------------------------------
 * With GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS, ggml builds one CPU library
 * per instruction-set level (ggml-cpu-haswell, -skylakex, -icelake, ...) and,
 * at run time, loads the one whose ggml_backend_score() is highest on this
 * CPU. Those libraries are loaded with LoadLibrary/dlopen, never linked, so
 * ggml_backend_cpu_init(), ggml_backend_cpu_buffer_type() and
 * ggml_backend_cpu_set_n_threads() do not exist at link time. Everything here
 * goes through ggml_backend_dev_* and ggml_backend_reg_get_proc_address(),
 * which work the same with a statically linked CPU backend: the code below has
 * a single path for both builds.
 *
 * Where the variants are looked for
 * ---------------------------------
 * ggml's default search path is the directory of the *executable* and the
 * current directory (ggml-backend-reg.cpp, ggml_backend_load_best). Under
 * Python that is the directory of python.exe, which holds no ggml library. So
 * the directory of the module that contains this code -- libembedding.dll /
 * .so, or the executable in a header-only build -- is searched first, then the
 * directory of the ggml library (where a build tree has its variants), then
 * ggml's defaults.
 *
 * Auteur: David Orel
 * Version: 1.11.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_GGML_CPU_BACKEND_HPP
#define LIBEMBEDDING_DETAIL_GGML_CPU_BACKEND_HPP

#include <ggml.h>
#include <ggml-backend.h>

#include <mutex>
#include <string>

#ifdef GGML_BACKEND_DL
#  if defined(_WIN32)
#    ifndef NOMINMAX
#      define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#  else
#    include <dlfcn.h>
#  endif
#endif

namespace lembed {
namespace detail {

#ifdef GGML_BACKEND_DL
/* Directory of the module (DLL / shared object / executable) that contains
 * `address`, UTF-8, without a trailing separator. Empty when unknown. */
inline std::string ggml_cpu_backend_module_dir(const void* address) {
    std::string path;
#  if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(address), &module)) {
        return path;
    }
    wchar_t wide[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(module, wide, (DWORD)(sizeof(wide) / sizeof(wide[0])));
    if (n == 0 || n >= (DWORD)(sizeof(wide) / sizeof(wide[0]))) return path;
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide, (int)n, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) return path;
    path.resize((size_t)bytes);
    WideCharToMultiByte(CP_UTF8, 0, wide, (int)n, &path[0], bytes, nullptr, nullptr);
#  else
    Dl_info info{};
    if (!dladdr(address, &info) || !info.dli_fname) return path;
    path = info.dli_fname;
#  endif
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}
#endif

#ifdef GGML_BACKEND_DL
/* Logger installed while the variants are scored and loaded: ggml reports
 * every successful load at INFO level ("load_backend: loaded CPU backend
 * from ..."), which would land on the stderr of every process using the
 * library. Warnings and errors still reach the logger that was in place. */
struct GgmlLogForward {
    ggml_log_callback callback = nullptr;
    void* user_data = nullptr;
};

inline void ggml_cpu_backend_quiet_log(ggml_log_level level, const char* text, void* user_data) {
    if (level != GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_ERROR) return;
    const auto* forward = static_cast<const GgmlLogForward*>(user_data);
    if (forward && forward->callback) forward->callback(level, text, forward->user_data);
}
#endif

/* Makes the CPU backend available, once per process. Idempotent and
 * thread-safe; every other function here calls it. */
inline void ggml_cpu_backend_ensure_loaded() {
#ifdef GGML_BACKEND_DL
    static std::once_flag once;
    std::call_once(once, []() {
        /* The registry is process-wide (it lives in ggml.dll / libggml), but
         * this flag is per module: the library and an executable that also
         * compiles these headers each have one. Whoever comes second must not
         * register a second CPU backend. */
        if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) return;

        GgmlLogForward previous;
        ggml_log_get(&previous.callback, &previous.user_data);
        ggml_log_set(ggml_cpu_backend_quiet_log, &previous);

        /* 1. next to the module holding this code (libembedding, or the
         *    executable in a header-only build): the wheel layout;
         * 2. next to ggml itself, which is where ggml writes its variants in a
         *    build tree (and where a layout that keeps ggml apart puts them);
         * 3. ggml's defaults: the executable's directory and the cwd. */
        const std::string dirs[] = {
            ggml_cpu_backend_module_dir(
                reinterpret_cast<const void*>(&ggml_cpu_backend_ensure_loaded)),
            ggml_cpu_backend_module_dir(
                reinterpret_cast<const void*>(&ggml_backend_load_all_from_path)),
        };
        for (const std::string& dir : dirs) {
            if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) break;
            if (!dir.empty()) ggml_backend_load_all_from_path(dir.c_str());
        }
        if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            ggml_backend_load_all();
        }

        ggml_log_set(previous.callback, previous.user_data);
    });
#endif
}

/* The CPU device, or nullptr when no CPU backend could be loaded (a broken
 * install: the ggml-cpu-* libraries are missing next to libembedding). */
inline ggml_backend_dev_t ggml_cpu_device() {
    ggml_cpu_backend_ensure_loaded();
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

/* The error text callers report when ggml_cpu_device() is nullptr. */
inline const char* ggml_cpu_backend_missing_message() {
    return "no ggml CPU backend could be loaded: the ggml-cpu libraries must sit "
           "next to libembedding";
}

/* Host buffer type of the CPU device; nullptr without a CPU backend. */
inline ggml_backend_buffer_type_t ggml_cpu_buffer_type() {
    ggml_backend_dev_t dev = ggml_cpu_device();
    return dev ? ggml_backend_dev_buffer_type(dev) : nullptr;
}

/* A new CPU backend instance; nullptr without a CPU backend. */
inline ggml_backend_t ggml_cpu_backend_init() {
    ggml_backend_dev_t dev = ggml_cpu_device();
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

/* Sets the thread count of a CPU backend instance. Returns false when the
 * loaded backend does not export the setter, which a CPU backend always does. */
inline bool ggml_cpu_backend_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_dev_t dev = ggml_cpu_device();
    if (!dev || !backend) return false;
    auto fn = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads"));
    if (!fn) return false;
    fn(backend, n_threads);
    return true;
}

/* Instruction-set features the loaded CPU backend was compiled with, as
 * "AVX2 AVX512 AVX512_VNNI ...": with CPU dispatch this names the variant ggml
 * picked for this machine. Empty without a CPU backend. */
inline std::string ggml_cpu_backend_features() {
    std::string out;
    ggml_backend_dev_t dev = ggml_cpu_device();
    if (!dev) return out;
    auto fn = reinterpret_cast<ggml_backend_get_features_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_get_features"));
    if (!fn) return out;
    for (ggml_backend_feature* f = fn(ggml_backend_dev_backend_reg(dev)); f && f->name; ++f) {
        if (!f->value || std::string(f->value) != "1") continue;
        if (!out.empty()) out += ' ';
        out += f->name;
    }
    return out;
}

} // namespace detail
} // namespace lembed

/* Implementation of the public lembed_ggml_cpu_features() declared in
 * config.h, which stays free of ggml so a C consumer can read the library
 * version without it. Computed once: the loaded variant cannot change within a
 * process. */
#if defined(LIBEMBEDDING_IMPLEMENTATION) && !defined(LIBEMBEDDING_GGML_CPU_FEATURES_IMPL)
#define LIBEMBEDDING_GGML_CPU_FEATURES_IMPL
extern "C" const char* lembed_ggml_cpu_features(void) {
    static const std::string features = lembed::detail::ggml_cpu_backend_features();
    return features.c_str();
}
#endif

#endif // LIBEMBEDDING_DETAIL_GGML_CPU_BACKEND_HPP
