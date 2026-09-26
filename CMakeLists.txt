cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0144 NEW)
set(LIBEMBEDDING_VERSION "1.6.0")
project(libembedding
    VERSION ${LIBEMBEDDING_VERSION}
    DESCRIPTION "C/C++ embedding library using ONNX Runtime (header-only INTERFACE on Linux/macOS, shared library/DLL on Windows)"
    LANGUAGES C CXX
)

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
else()
    find_package(CURL REQUIRED)
endif()

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

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp-build EXCLUDE_FROM_ALL)


# =========================================================================
# Main library target (Forced as SHARED DLL for Windows/Python cffi)
# =========================================================================
if(WIN32)
    # Create a real DLL. If a cpp implementation file exists (ex: src/libembedding.cpp),
    # add it here instead of ""
    add_library(libembedding SHARED "${CMAKE_CURRENT_SOURCE_DIR}/python/csrc/libembedding_shlib.cpp")

    # Use explicit .def file for exports instead of WINDOWS_EXPORT_ALL_SYMBOLS
    # This allows LTCG (/GL /LTCG) to be used if desired
    set_target_properties(libembedding PROPERTIES
        LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/libembedding.def"
    )

    # Force CMake to accept a shared library without initial source files
    # car elle va lier ONNXRuntime et CURL
    set_target_properties(libembedding PROPERTIES LINKER_LANGUAGE CXX)

    target_link_libraries(libembedding PUBLIC llama ggml)
    target_include_directories(libembedding PUBLIC
        $<BUILD_INTERFACE:${LLAMA_CPP_DIR}/include>
        $<BUILD_INTERFACE:${LLAMA_CPP_DIR}/ggml/include>
    )
else()
    # Header-only INTERFACE pour Linux / macOS
    add_library(libembedding INTERFACE)
    target_link_libraries(libembedding INTERFACE llama ggml)
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
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
    )
    target_compile_definitions(libembedding PRIVATE
        $<$<CONFIG:Release>:NDEBUG>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libembedding INTERFACE
        -O3
        -ffast-math
    )
    # Enable AVX2/AVX-512 where available
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" _has_avx2)
        if(_has_avx2)
            target_compile_options(libembedding INTERFACE -mavx2)
        endif()
    endif()
endif()

# =========================================================================
# Configuration of Includes and Links (Adapted for Windows/DLL)
# =========================================================================
if(WIN32)
    # Mode DLL (SHARED) pour Windows
    target_include_directories(libembedding PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
        $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(libembedding PUBLIC
        ONNXRuntime::ONNXRuntime
        cjson
    )

if(NOT LIBEMBEDDING_NO_IMAGE)
        target_link_libraries(libembedding PUBLIC libembedding_stb)
    endif()

    if(NOT LIBEMBEDDING_NO_DOWNLOAD)
        target_link_libraries(libembedding PUBLIC ${CURL_LIBRARY})
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
        $<INSTALL_INTERFACE:include>
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

        # llama.cpp runtime DLLs
        foreach(_target llama ggml ggml-cpu ggml-base)
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
    if(APPLE)
        set_target_properties(embedding_shared PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(embedding_shared PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
    if(NOT WIN32 AND ONNXRuntime_LIBRARIES)
        copy_runtime_libs(embedding_shared)
    endif()
endif()

# =========================================================================
# Install
# =========================================================================
include(GNUInstallDirs)

install(DIRECTORY include/libembedding
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY third_party/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libembedding/detail
    FILES_MATCHING PATTERN "*.h"
)

set(_install_targets libembedding cjson)
if(NOT LIBEMBEDDING_NO_IMAGE AND TARGET libembedding_stb)
    list(APPEND _install_targets libembedding_stb)
endif()
install(TARGETS ${_install_targets}
    EXPORT libembedding-targets
)



