cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0144 NEW)
set(LIBEMBEDDING_VERSION "1.12.1")
project(libembedding
    VERSION ${LIBEMBEDDING_VERSION}
    DESCRIPTION "C/C++ embedding library using ONNX Runtime (header-only INTERFACE on Linux/macOS, shared library/DLL on Windows)"
    LANGUAGES C CXX
)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_UNUSED_CLI_VARIABLES ON)

if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# =========================================================================
# Options
# =========================================================================
option(LIBEMBEDDING_NO_DOWNLOAD   "Disable model downloading (offline only)"  OFF)
option(LIBEMBEDDING_NO_IMAGE      "Disable image embedding support"           OFF)
option(LIBEMBEDDING_BUILD_TESTS   "Build tests"                               ON)
option(LIBEMBEDDING_BUILD_EXAMPLES "Build examples"                           ON)
option(LIBEMBEDDING_BUILD_BENCHMARKS "Build benchmarks"                       OFF)
option(LIBEMBEDDING_BUILD_SHARED    "Build shared library for bindings"       OFF)
option(LIBEMBEDDING_CURL_STATIC    "libcurl is linked statically or its runtime is managed by the caller" OFF)
# One ggml CPU library per instruction-set level, the best one picked at run
# time (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS). OFF builds a single ggml-cpu
# for the build machine (GGML_NATIVE), which must not be shipped. macOS keeps
# the native build until the Apple Silicon variants are validated (P2.6).
if(APPLE)
    set(_lembed_cpu_dispatch_default OFF)
else()
    set(_lembed_cpu_dispatch_default ON)
endif()
option(LIBEMBEDDING_CPU_DISPATCH "Build every ggml CPU variant and select one at run time" ${_lembed_cpu_dispatch_default})
# libembedding's own code (pooling, similarity, SPLADE pooling, tokenizer...)
# targets the x86-64 baseline so that the shipped binaries run on any x64 CPU,
# as the ggml CPU dispatch above does: built with AVX2, they stop with an
# illegal instruction on CPUs without it (verified under Intel SDE on Nehalem
# and Sandy Bridge). The hot paths are in ONNX Runtime and ggml, which select
# their kernels at run time. ON is for local builds that never leave the
# machine, never for wheels or release binaries.
option(LIBEMBEDDING_ARCH_AVX2 "Compile libembedding's own code for AVX2 (not portable)" OFF)

# =========================================================================
# Dependencies
# =========================================================================

if(WIN32)
    # Allow overriding bundled third_party paths via environment variable
    # (e.g. for CI builds downloading onnxruntime from GitHub releases)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        string(REPLACE "/" "\\" _ort_root "$ENV{ONNXRUNTIME_ROOT}")
        set(ONNXRuntime_INCLUDE_DIR "${_ort_root}\\include")
        set(ONNXRuntime_LIBRARY "${_ort_root}\\lib\\onnxruntime.lib")
    elseif(NOT ONNXRuntime_INCLUDE_DIR)
        set(ONNXRuntime_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")
    endif()
    if(NOT ONNXRuntime_LIBRARY)
        set(ONNXRuntime_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib/onnxruntime.lib")
    endif()
    include_directories(${ONNXRuntime_INCLUDE_DIR})

    if(NOT CURL_INCLUDE_DIR)
        set(CURL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl/include")
    endif()
    if(NOT CURL_LIBRARY)
        set(CURL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl/lib/libcurl.dll.a")
    endif()
    include_directories(${CURL_INCLUDE_DIR})

    # Disable conflicting min/max macros globally on Windows
    add_compile_definitions(NOMINMAX)
endif()
# libcurl elsewhere is found below, only when downloading is enabled: an
# unconditional find_package(CURL REQUIRED) here made -DLIBEMBEDDING_NO_DOWNLOAD=ON
# fail to configure on platforms without libcurl (Android).

# -------------------------------------------------------------------------
# libcurl runtime DLL (Windows)
#
# third_party/curl ships headers + an import library only. The link
# succeeds, but libembedding.dll and every test/example import
# libcurl-x64.dll at load time: without it the process dies with
# 0xC0000135 (STATUS_DLL_NOT_FOUND), which ctest reports as a bogus
# "Timeout". Resolve the runtime here so the failure is a configure-time
# error with an actionable message instead.
#
# Search order: explicit override, bundled runtime, CURL_LIBRARY and its
# sibling bin/ (vcpkg layout), Python package dir (wheel/dev artifact).
# -------------------------------------------------------------------------
if(WIN32 AND NOT LIBEMBEDDING_NO_DOWNLOAD AND NOT LIBEMBEDDING_CURL_STATIC)
    if(CURL_RUNTIME_DLL AND EXISTS "${CURL_RUNTIME_DLL}")
        set(LIBEMBEDDING_CURL_RUNTIME_DLL "${CURL_RUNTIME_DLL}")
    else()
        set(_curl_runtime_candidates "")
        file(GLOB _bundled "${CMAKE_CURRENT_SOURCE_DIR}/third_party/curl/bin/libcurl*.dll")
        list(APPEND _curl_runtime_candidates ${_bundled})
        if(CURL_LIBRARY)
            get_filename_component(_curl_lib_dir "${CURL_LIBRARY}" DIRECTORY)
            file(GLOB _curl_same "${_curl_lib_dir}/libcurl*.dll")
            file(GLOB _curl_bin "${_curl_lib_dir}/../bin/libcurl*.dll")
            list(APPEND _curl_runtime_candidates ${_curl_same} ${_curl_bin})
        endif()
        file(GLOB _curl_python "${CMAKE_CURRENT_SOURCE_DIR}/python/src/libembedding/libcurl*.dll")
        list(APPEND _curl_runtime_candidates ${_curl_python})

        list(REMOVE_DUPLICATES _curl_runtime_candidates)
        if(_curl_runtime_candidates)
            list(GET _curl_runtime_candidates 0 LIBEMBEDDING_CURL_RUNTIME_DLL)
        endif()
        unset(_curl_runtime_candidates)
    endif()

    if(NOT LIBEMBEDDING_CURL_RUNTIME_DLL)
        message(FATAL_ERROR
            "libcurl runtime DLL not found.\n"
            "libembedding.dll links against libcurl and needs libcurl-x64.dll next to the "
            "executables at run time.\n"
            "Fix it with one of:\n"
            "  pwsh -File scripts/fetch_windows_libcurl.ps1        (installs the bundled runtime)\n"
            "  -DCURL_LIBRARY=<vcpkg>\\installed\\x64-windows\\lib\\libcurl.lib\n"
            "  -DCURL_RUNTIME_DLL=<path>\\libcurl-x64.dll\n"
            "  -DLIBEMBEDDING_NO_DOWNLOAD=ON                       (no download code at all)\n"
            "  -DLIBEMBEDDING_CURL_STATIC=ON                        (runtime managed by the caller)")
    endif()
    get_filename_component(LIBEMBEDDING_CURL_RUNTIME_DLL "${LIBEMBEDDING_CURL_RUNTIME_DLL}" ABSOLUTE)
    message(STATUS "libcurl runtime: ${LIBEMBEDDING_CURL_RUNTIME_DLL}")
endif()

# ONNX Runtime (required)
# Use our custom FindONNXRuntime module which handles downloaded archives
# and system installs correctly
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Hint Homebrew paths on macOS
if(APPLE AND EXISTS "/opt/homebrew/opt/onnxruntime")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/onnxruntime")
endif()

find_package(ONNXRuntime REQUIRED)

# cJSON (bundled, for parsing tokenizer.json and config files)
add_library(cjson STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cJSON/cJSON.c"
)
target_include_directories(cjson PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/cJSON>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(cjson PROPERTIES POSITION_INDEPENDENT_CODE ON)

# stb (bundled image loading/resizing, optional)
if(NOT LIBEMBEDDING_NO_IMAGE)
    add_library(libembedding_stb STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb/stb_impl.c"
    )
    target_include_directories(libembedding_stb PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    set_target_properties(libembedding_stb PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# libcurl (optional, for downloading)
if(NOT LIBEMBEDDING_NO_DOWNLOAD)
    find_package(CURL REQUIRED)
endif()

# =========================================================================
# llama.cpp (always enabled, bundled in third_party/)
# Must be declared BEFORE the main library target so that ggml/llama targets
# are available when the install(EXPORT) is generated.
#
# The bundled snapshot is trimmed: only CMakeLists.txt, LICENSE, AUTHORS,
# README.md, cmake/, ggml/, include/, src/, licenses/ and vendor/ are kept
# (see third_party/README.md and scripts/prune_llama_cpp.ps1). Every option
# below that points at a removed directory is pinned OFF so a stray -D cannot
# make the configure step fail on a missing add_subdirectory() target.
# =========================================================================
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")

# llama.cpp options
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_STATIC ON CACHE BOOL "" FORCE)
set(LLAMA_SHARED OFF CACHE BOOL "" FORCE)
set(LLAMA_CUDA OFF CACHE BOOL "" FORCE)
set(LLAMA_VULKAN OFF CACHE BOOL "" FORCE)
set(LLAMA_METAL OFF CACHE BOOL "" FORCE)

# Directories removed from the snapshot: keep their entry points OFF.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)  # common/      (server, CLI utils)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)   # tools/       (server, mtmd CLI)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)     # app/         (unified binary)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)    # tools/mtmd/  (multimodal CLI)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)    # ggml/tests/
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # ggml/examples/
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)         # ggml-opencl/ (kernels/embed_kernel.py)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)         # ggml-webgpu/ (wgsl-shaders/embed_wgsl.py)
set(GGML_VIRTGPU OFF CACHE BOOL "" FORCE)        # ggml-virtgpu/

# CPU instruction set. Without dispatch ggml compiles for the build machine
# (GGML_NATIVE: /arch:AVX512 on an AVX-512 runner), and the library then dies
# with an illegal instruction on any CPU that lacks it. With dispatch ggml
# builds every variant as a loadable module; GGML_NATIVE is incompatible with
# that mode (ggml-cpu/CMakeLists.txt), and BUILD_SHARED_LIBS is required.
if(LIBEMBEDDING_CPU_DISPATCH)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
endif()

# Android: ggml's OpenMP build links the NDK's libomp.so, which is not on the
# device ("library libomp.so not found", measured on an API 30 emulator) and
# which upstream does not support shipping either (docs/android.md). Upstream's
# Android release builds with GGML_OPENMP=OFF; ggml then uses its own threadpool.
if(ANDROID)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()

# llama.cpp's upstream install rules assume the unpruned source tree layout.
# The parent project installs the vendored targets and headers explicitly.
set(CMAKE_SKIP_INSTALL_RULES TRUE)
add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp-build EXCLUDE_FROM_ALL)
unset(CMAKE_SKIP_INSTALL_RULES)
set_target_properties(llama ggml PROPERTIES PUBLIC_HEADER "")

set_target_properties(llama PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${LLAMA_CPP_DIR}/include>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
# The ggml CPU libraries actually built: ggml-cpu alone, or one
# ggml-cpu-<variant> module per instruction-set level with CPU dispatch. Read
# from the build system rather than listed here, since the variant set depends
# on the compiler (MSVC builds fewer x86 variants than GCC/Clang).
set(LIBEMBEDDING_GGML_CPU_TARGETS "")
foreach(_dir "${LLAMA_CPP_DIR}/ggml/src" "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu")
    get_property(_dir_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_t IN LISTS _dir_targets)
        if(_t MATCHES "^ggml-cpu(-[A-Za-z0-9_.]+)?$")
            get_target_property(_t_type ${_t} TYPE)
            if(_t_type STREQUAL "SHARED_LIBRARY" OR _t_type STREQUAL "MODULE_LIBRARY"
               OR _t_type STREQUAL "STATIC_LIBRARY")
                list(APPEND LIBEMBEDDING_GGML_CPU_TARGETS ${_t})
            endif()
        endif()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES LIBEMBEDDING_GGML_CPU_TARGETS)
if(NOT LIBEMBEDDING_GGML_CPU_TARGETS)
    message(FATAL_ERROR "no ggml CPU backend target was found under ${LLAMA_CPP_DIR}/ggml/src")
endif()
message(STATUS "libembedding: CPU dispatch=${LIBEMBEDDING_CPU_DISPATCH}, ggml CPU targets: ${LIBEMBEDDING_GGML_CPU_TARGETS}")

set_target_properties(ggml ggml-base ${LIBEMBEDDING_GGML_CPU_TARGETS} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
        "$<BUILD_INTERFACE:${LLAMA_CPP_DIR}/ggml/include>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# =========================================================================
# Main library target (Forced as SHARED DLL for Windows/Python cffi)
# =========================================================================
if(WIN32)
    # Create a real DLL. If a cpp implementation file exists (ex: src/libembedding.cpp),
    # add it here instead of ""
    add_library(libembedding SHARED "${CMAKE_CURRENT_SOURCE_DIR}/python/csrc/libembedding_shlib.cpp")

    # Use explicit .def file for exports instead of WINDOWS_EXPORT_ALL_SYMBOLS
    # This allows LTCG (/GL /LTCG) to be used if desired
    #
    # The .def is filtered first: it names every public symbol, but a backend
    # compiled out (LIBEMBEDDING_NO_IMAGE) has no definition, and the linker
    # treats an unresolved name in a .def as a hard error rather than a warning.
    # Naming lembed_image_* while the image backend was disabled therefore failed
    # the link with LNK2001 on all 13 of them. Blocks bracketed by
    # ";@IF <cond>" / ";@ENDIF" are dropped when <cond> is false.
    set(_lembed_def_src "${CMAKE_CURRENT_SOURCE_DIR}/libembedding.def")
    set(_lembed_def_out "${CMAKE_CURRENT_BINARY_DIR}/libembedding.generated.def")
    file(READ "${_lembed_def_src}" _lembed_def_text)
    string(REPLACE "\r\n" "\n" _lembed_def_text "${_lembed_def_text}")
    # Every ";" is parked behind a sentinel before the text is split on ";", the
    # list separator: a line like ";@IF X" or "; ---- Image embedding ----"
    # would otherwise be torn in two and lose its semicolon.
    string(REPLACE ";" "@semi@" _lembed_def_text "${_lembed_def_text}")
    string(REPLACE "\n" ";" _lembed_def_lines "${_lembed_def_text}")

    set(_lembed_def_kept "")
    set(_lembed_def_skipping FALSE)
    foreach(_lembed_def_line IN LISTS _lembed_def_lines)
        # The condition is matched in two forms and evaluated by us rather than
        # handed to if() as one string: if(${c}) with c="NOT X" is a *single*
        # argument to if(), which reads as an unknown expression and is silently
        # false. Forms are ";@IF <var>" (keep when true) and ";@IF NOT <var>"
        # (keep when false). if(<var>) errors on a bad name, so a typo in a
        # marker stops the configure instead of silently dropping exports.
        if(_lembed_def_line MATCHES "^[ \t]*@semi@@IF[ \t]+NOT[ \t]+(.+)$")
            # ";@IF NOT X" keeps the block when X is false.
            if(${CMAKE_MATCH_1})
                set(_lembed_def_skipping TRUE)
            else()
                set(_lembed_def_skipping FALSE)
            endif()
        elseif(_lembed_def_line MATCHES "^[ \t]*@semi@@IF[ \t]+(.+)$")
            # ";@IF X" keeps the block when X is true.
            if(${CMAKE_MATCH_1})
                set(_lembed_def_skipping FALSE)
            else()
                set(_lembed_def_skipping TRUE)
            endif()
        elseif(_lembed_def_line MATCHES "^[ \t]*@semi@@ENDIF")
            set(_lembed_def_skipping FALSE)
        elseif(NOT _lembed_def_skipping)
            string(REPLACE "@semi@" ";" _lembed_def_line "${_lembed_def_line}")
            string(APPEND _lembed_def_kept "${_lembed_def_line}\n")
        endif()
    endforeach()
    file(WRITE "${_lembed_def_out}" "${_lembed_def_kept}")

    set_target_properties(libembedding PROPERTIES
        LINK_FLAGS "/DEF:${_lembed_def_out}"
    )

    # Force CMake to accept a shared library without initial source files
    # car elle va lier ONNXRuntime et CURL
    set_target_properties(libembedding PROPERTIES LINKER_LANGUAGE CXX)

    target_link_libraries(libembedding PRIVATE
        $<BUILD_INTERFACE:llama>
        $<BUILD_INTERFACE:ggml>
    )
    target_include_directories(libembedding PUBLIC
        $<BUILD_INTERFACE:${LLAMA_CPP_DIR}/include>
        $<BUILD_INTERFACE:${LLAMA_CPP_DIR}/ggml/include>
    )
else()
    # Header-only INTERFACE pour Linux / macOS
    add_library(libembedding INTERFACE)
    target_link_libraries(libembedding INTERFACE
        $<BUILD_INTERFACE:llama>
        $<BUILD_INTERFACE:ggml>
        $<INSTALL_INTERFACE:llama>
        $<INSTALL_INTERFACE:ggml>
    )
    target_include_directories(libembedding INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${LLAMA_CPP_DIR}/include>
        $<BUILD_INTERFACE:${LLAMA_CPP_DIR}/ggml/include>
    )
endif()

add_library(libembedding::libembedding ALIAS libembedding)

# =========================================================================
# CPU Optimizations
# =========================================================================
# Note: /GL (whole-program opt) and /LTCG can now be used on MSVC
# because we use an explicit .def file for exports instead of
# WINDOWS_EXPORT_ALL_SYMBOLS.
if(MSVC)
    target_compile_options(libembedding PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/fp:fast>
    )
    if(LIBEMBEDDING_ARCH_AVX2)
        target_compile_options(libembedding PRIVATE
            $<$<CONFIG:Release>:/arch:AVX2>
            $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
        )
    endif()
    target_compile_definitions(libembedding PRIVATE
        $<$<CONFIG:Release>:NDEBUG>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libembedding INTERFACE
        -O3
        -ffast-math
    )
    # No -mavx2 here: as an INTERFACE flag it reached every consumer of the
    # header-only library and the Python shared library. See
    # LIBEMBEDDING_ARCH_AVX2, applied to embedding_shared only.
endif()

# =========================================================================
# Configuration of Includes and Links (Adapted for Windows/DLL)
# =========================================================================
if(WIN32)
    # Mode DLL (SHARED) pour Windows
    target_include_directories(libembedding PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(libembedding PRIVATE
        ONNXRuntime::ONNXRuntime
        cjson
    )

if(NOT LIBEMBEDDING_NO_IMAGE)
        target_link_libraries(libembedding PRIVATE libembedding_stb)
    endif()

    if(NOT LIBEMBEDDING_NO_DOWNLOAD)
        target_link_libraries(libembedding PRIVATE ${CURL_LIBRARY})
    else()
        target_compile_definitions(libembedding PUBLIC LIBEMBEDDING_NO_DOWNLOAD)
    endif()

    if(LIBEMBEDDING_NO_IMAGE)
        target_compile_definitions(libembedding PUBLIC LIBEMBEDDING_NO_IMAGE)
    endif()
else()
    # Mode d'origine (INTERFACE) pour Linux / macOS
    target_include_directories(libembedding INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    target_link_libraries(libembedding INTERFACE
        ONNXRuntime::ONNXRuntime
        cjson
    )

    if(NOT LIBEMBEDDING_NO_IMAGE)
        target_link_libraries(libembedding INTERFACE libembedding_stb)
    endif()

    if(NOT LIBEMBEDDING_NO_DOWNLOAD)
        target_link_libraries(libembedding INTERFACE CURL::libcurl)
    else()
        target_compile_definitions(libembedding INTERFACE LIBEMBEDDING_NO_DOWNLOAD)
    endif()

    if(LIBEMBEDDING_NO_IMAGE)
        target_compile_definitions(libembedding INTERFACE LIBEMBEDDING_NO_IMAGE)
    endif()
endif()

# =========================================================================
# Helper: copy runtime DLLs next to executables on Windows
# =========================================================================
if(WIN32)
    set(_LEMBED_ONNX_DIR "${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib")
    function(copy_runtime_dlls target_name)
        set(_dlls "")

        # ONNX Runtime DLLs
        foreach(_dll onnxruntime.dll onnxruntime_providers_shared.dll)
            if(EXISTS "${_LEMBED_ONNX_DIR}/${_dll}")
                list(APPEND _dlls "${_LEMBED_ONNX_DIR}/${_dll}")
            endif()
        endforeach()

        # Curl DLL: the runtime resolved at configure time (bundled
        # third_party/curl/bin, vcpkg bin/, or CURL_RUNTIME_DLL). Copying it
        # is what keeps the test executables runnable.
        if(LIBEMBEDDING_CURL_RUNTIME_DLL)
            list(APPEND _dlls "${LIBEMBEDDING_CURL_RUNTIME_DLL}")
        endif()

        # llama.cpp runtime DLLs. With CPU dispatch the ggml-cpu-* variants
        # must sit next to libembedding.dll: that is where it looks for them.
        foreach(_target llama ggml ggml-base ${LIBEMBEDDING_GGML_CPU_TARGETS})
            if(TARGET ${_target})
                list(APPEND _dlls "$<TARGET_FILE:${_target}>")
            endif()
        endforeach()

        # libembedding.dll
        if(TARGET libembedding)
            list(APPEND _dlls "$<TARGET_FILE:libembedding>")
        endif()

        list(REMOVE_DUPLICATES _dlls)
        foreach(_dll ${_dlls})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_dll}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endforeach()
    endfunction()

    # Copy runtime DLLs next to libembedding.dll
    copy_runtime_dlls(libembedding)

    # For executables that define LIBEMBEDDING_IMPLEMENTATION themselves (the
    # examples and benchmarks): the DLL keeps its dependencies PRIVATE, so the
    # implementation compiled into the executable links them directly. On
    # Linux/macOS the INTERFACE target already carries them.
    function(link_implementation_deps target_name)
        target_link_libraries(${target_name} PRIVATE
            llama ggml cjson ONNXRuntime::ONNXRuntime)
        if(TARGET libembedding_stb)
            target_link_libraries(${target_name} PRIVATE libembedding_stb)
        endif()
        if(NOT LIBEMBEDDING_NO_DOWNLOAD AND CURL_LIBRARY)
            target_link_libraries(${target_name} PRIVATE ${CURL_LIBRARY})
        endif()
    endfunction()
endif()

# =========================================================================
# Helper: copy runtime shared libraries next to executables on Linux/macOS
# =========================================================================
if(NOT WIN32)
    function(copy_runtime_libs target_name)
        set(_to_copy "")

        # ONNX Runtime
        if(ONNXRuntime_LIBRARIES)
            get_filename_component(_ort_lib_dir "${ONNXRuntime_LIBRARIES}" DIRECTORY)

            if(APPLE)
                set(_soname "libonnxruntime.dylib")
                set(_provider "libonnxruntime_providers_shared.dylib")
            else()
                set(_soname "libonnxruntime.so")
                set(_provider "libonnxruntime_providers_shared.so")
            endif()

            if(EXISTS "${ONNXRuntime_LIBRARIES}")
                list(APPEND _to_copy "${ONNXRuntime_LIBRARIES}")
            endif()
            if(EXISTS "${_ort_lib_dir}/${_soname}")
                list(APPEND _to_copy "${_ort_lib_dir}/${_soname}")
            endif()
            if(EXISTS "${_ort_lib_dir}/${_provider}")
                list(APPEND _to_copy "${_ort_lib_dir}/${_provider}")
            endif()
        endif()

        # libcurl (optional, only if download is enabled)
        if(NOT LIBEMBEDDING_NO_DOWNLOAD)
            if(CURL_LIBRARIES AND EXISTS "${CURL_LIBRARIES}")
                get_filename_component(_curl_ext "${CURL_LIBRARIES}" EXT)
                if(_curl_ext MATCHES "^\\.(so|dylib)$")
                    list(APPEND _to_copy "${CURL_LIBRARIES}")
                endif()
            endif()
        endif()

        # CPU dispatch: the header-only build runs inside the executable, so
        # ggml looks for its ggml-cpu-* modules next to it.
        if(LIBEMBEDDING_CPU_DISPATCH)
            foreach(_cpu IN LISTS LIBEMBEDDING_GGML_CPU_TARGETS)
                list(APPEND _to_copy "$<TARGET_FILE:${_cpu}>")
                add_dependencies(${target_name} ${_cpu})
            endforeach()
        endif()

        list(REMOVE_DUPLICATES _to_copy)

        foreach(_lib IN LISTS _to_copy)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_lib}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endforeach()
    endfunction()
endif()

# =========================================================================
# Tests
# =========================================================================
if(LIBEMBEDDING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# =========================================================================
# Examples
# =========================================================================
if(LIBEMBEDDING_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =========================================================================
# Benchmarks
# =========================================================================
if(LIBEMBEDDING_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# =========================================================================
# Shared library (for Python/FFI bindings)
# =========================================================================
if(LIBEMBEDDING_BUILD_SHARED)
    add_library(embedding_shared SHARED python/csrc/libembedding_shlib.cpp)
    target_link_libraries(embedding_shared PRIVATE libembedding)
    set_target_properties(embedding_shared PROPERTIES
        OUTPUT_NAME "embedding"
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
    )
    if(LIBEMBEDDING_ARCH_AVX2 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
       AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        target_compile_options(embedding_shared PRIVATE -mavx2 -mfma)
    endif()
    if(APPLE)
        set_target_properties(embedding_shared PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(embedding_shared PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
    if(WIN32)
        # libembedding.dll keeps its dependencies PRIVATE, and this target
        # compiles the implementation again (libembedding_shlib.cpp), so it
        # must link ORT, ggml/llama, cJSON and curl itself, like the examples.
        link_implementation_deps(embedding_shared)
    elseif(ONNXRuntime_LIBRARIES)
        copy_runtime_libs(embedding_shared)
    endif()
endif()

# =========================================================================
# Install
# =========================================================================
install(DIRECTORY include/libembedding
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY "${LLAMA_CPP_DIR}/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY "${LLAMA_CPP_DIR}/ggml/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY third_party/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libembedding/detail
    FILES_MATCHING PATTERN "*.h"
)

set(_install_targets libembedding cjson llama ggml ggml-base)
if(NOT LIBEMBEDDING_CPU_DISPATCH)
    list(APPEND _install_targets ${LIBEMBEDDING_GGML_CPU_TARGETS})
    # Without dynamic backend loading, ggml links every backend it enabled, so
    # each one must be in the same export set. On macOS ggml enables BLAS
    # (Accelerate) and Metal by default; elsewhere these targets do not exist.
    foreach(_backend ggml-blas ggml-metal)
        if(TARGET ${_backend})
            list(APPEND _install_targets ${_backend})
        endif()
    endforeach()
endif()
if(NOT LIBEMBEDDING_NO_IMAGE AND TARGET libembedding_stb)
    list(APPEND _install_targets libembedding_stb)
endif()
if(LIBEMBEDDING_CPU_DISPATCH)
    # Loadable modules: nothing links them, so they are not part of the export
    # set. They go where ggml looks for them, next to the library that loads
    # them (bin/ on Windows, beside libembedding.dll).
    if(WIN32)
        set(_lembed_cpu_module_dir ${CMAKE_INSTALL_BINDIR})
    else()
        set(_lembed_cpu_module_dir ${CMAKE_INSTALL_LIBDIR})
    endif()
    install(TARGETS ${LIBEMBEDDING_GGML_CPU_TARGETS}
        LIBRARY DESTINATION ${_lembed_cpu_module_dir}
        RUNTIME DESTINATION ${_lembed_cpu_module_dir}
    )
endif()
install(TARGETS ${_install_targets}
    EXPORT libembedding-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT libembedding-targets
    NAMESPACE libembedding::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libembedding
)

if(WIN32 OR LIBEMBEDDING_NO_DOWNLOAD)
    set(LIBEMBEDDING_PACKAGE_REQUIRES_CURL OFF)
else()
    set(LIBEMBEDDING_PACKAGE_REQUIRES_CURL ON)
endif()
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libembeddingConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libembeddingConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libembedding"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libembeddingConfigVersion.cmake"
    VERSION ${LIBEMBEDDING_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libembeddingConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libembeddingConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libembedding"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindONNXRuntime.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libembedding/modules"
)
if(WIN32)
    foreach(_runtime_dll onnxruntime.dll onnxruntime_providers_shared.dll)
        if(EXISTS "${_LEMBED_ONNX_DIR}/${_runtime_dll}")
            install(FILES "${_LEMBED_ONNX_DIR}/${_runtime_dll}"
                DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
        endif()
    endforeach()
    if(LIBEMBEDDING_CURL_RUNTIME_DLL)
        install(FILES "${LIBEMBEDDING_CURL_RUNTIME_DLL}"
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()
