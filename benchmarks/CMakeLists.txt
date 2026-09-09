# Benchmarks
# Canonical benchmarks only. Debug/one-off scripts have been moved to archive/.

add_executable(bench_libembedding bench_libembedding.cpp)
target_link_libraries(bench_libembedding PRIVATE libembedding)

add_executable(bench_reranker_compare bench_reranker_compare.cpp)
target_link_libraries(bench_reranker_compare PRIVATE libembedding)
target_compile_definitions(bench_reranker_compare PRIVATE
    LIBEMBEDDING_IMPLEMENTATION
)

add_executable(fair_compare fair_compare.cpp)
target_link_libraries(fair_compare PRIVATE libembedding)
target_compile_definitions(fair_compare PRIVATE LIBEMBEDDING_NO_DOWNLOAD)

if(WIN32)
    set(_benchmark_targets bench_libembedding bench_reranker_compare fair_compare)
    foreach(_t IN LISTS _benchmark_targets)
        if(TARGET ${_t})
            copy_runtime_dlls(${_t})
        endif()
    endforeach()
elseif(ONNXRuntime_LIBRARIES)
    set(_benchmark_targets bench_libembedding bench_reranker_compare fair_compare)
    foreach(_t IN LISTS _benchmark_targets)
        if(TARGET ${_t})
            copy_runtime_libs(${_t})
        endif()
    endforeach()
endif()
