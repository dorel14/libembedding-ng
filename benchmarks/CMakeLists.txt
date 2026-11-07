# Benchmarks
# Canonical benchmarks only. Debug/one-off scripts have been moved to archive/.

add_executable(bench_libembedding bench_libembedding.cpp)
target_link_libraries(bench_libembedding PRIVATE libembedding)

add_executable(bench_reranker_compare bench_reranker_compare.cpp)
target_link_libraries(bench_reranker_compare PRIVATE libembedding)
target_compile_definitions(bench_reranker_compare PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

# Auto-tuner before/after harness. Public C API only, linked against the shared
# library, so the same source measures the tuners across a refactor.
add_executable(bench_autotune bench_autotune.cpp)
target_link_libraries(bench_autotune PRIVATE libembedding)

add_executable(fair_compare fair_compare.cpp)
target_link_libraries(fair_compare PRIVATE libembedding)
target_compile_definitions(fair_compare PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

# The microbenchmark drives ggml-cpu directly (thread pools, graph plans) to
# time one kernel build in isolation. With CPU dispatch there is no ggml-cpu to
# link against -- the variants are loadable modules -- so it needs
# -DLIBEMBEDDING_CPU_DISPATCH=OFF, which also pins the variant it measures to
# the build machine.
if(LIBEMBEDDING_CPU_DISPATCH)
    message(STATUS "gguf_gemm_microbench skipped: it needs -DLIBEMBEDDING_CPU_DISPATCH=OFF")
else()
    add_executable(gguf_gemm_microbench gguf_gemm_microbench.cpp)
    target_link_libraries(gguf_gemm_microbench PRIVATE ggml)
    if(MSVC)
        target_compile_options(gguf_gemm_microbench PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/arch:AVX2>
        )
    endif()
endif()

if(WIN32)
    set(_benchmark_targets bench_libembedding bench_reranker_compare fair_compare
        gguf_gemm_microbench bench_autotune)
    foreach(_t IN LISTS _benchmark_targets)
        if(TARGET ${_t})
            copy_runtime_dlls(${_t})
        endif()
    endforeach()
    # These three compile the implementation themselves (see the root helper).
    foreach(_t bench_libembedding bench_reranker_compare fair_compare)
        link_implementation_deps(${_t})
    endforeach()
elseif(ONNXRuntime_LIBRARIES)
    set(_benchmark_targets bench_libembedding bench_reranker_compare fair_compare
        gguf_gemm_microbench bench_autotune)
    foreach(_t IN LISTS _benchmark_targets)
        if(TARGET ${_t})
            copy_runtime_libs(${_t})
        endif()
    endforeach()
endif()
