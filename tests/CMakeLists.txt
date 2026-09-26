# Tests for libembedding
# Unit tests can run without model downloads
# Integration tests require network access

# =========================================================================
# Unit Tests (no downloads required)
# =========================================================================

# Test model registry
add_executable(test_model_registry test_model_registry.cpp)
target_link_libraries(test_model_registry PRIVATE libembedding)
add_test(NAME test_model_registry COMMAND test_model_registry)

# Test pooling and normalization
add_executable(test_pooling test_pooling.cpp)
target_link_libraries(test_pooling PRIVATE libembedding)
add_test(NAME test_pooling COMMAND test_pooling)

# Test similarity functions (no download required)
add_executable(test_similarity test_similarity.cpp)
target_link_libraries(test_similarity PRIVATE libembedding)
add_test(NAME test_similarity COMMAND test_similarity)

# Test local loading (no network required — tests error paths)
add_executable(test_local_loading test_local_loading.cpp)
target_link_libraries(test_local_loading PRIVATE libembedding)
# This test only exercises create_from_path error paths; no download code needed.
# Defining LIBEMBEDDING_NO_DOWNLOAD avoids pulling in libcurl.dll dependency.
target_compile_definitions(test_local_loading PRIVATE LIBEMBEDDING_NO_DOWNLOAD)
add_test(NAME test_local_loading COMMAND test_local_loading)

# Test the ABI of the structs shared with the cffi bindings (no network)
add_executable(test_struct_sizes test_struct_sizes.cpp)
target_link_libraries(test_struct_sizes PRIVATE libembedding)
add_test(NAME test_struct_sizes COMMAND test_struct_sizes)

# On Windows, copy runtime DLLs (onnxruntime, libcurl) next to test executables
if(WIN32)
    foreach(_t test_model_registry test_pooling test_similarity test_local_loading test_struct_sizes)
        if(TARGET ${_t})
            copy_runtime_dlls(${_t})
        endif()
    endforeach()
elseif(ONNXRuntime_LIBRARIES)
    foreach(_t test_model_registry test_pooling test_similarity test_local_loading test_struct_sizes)
        if(TARGET ${_t})
            copy_runtime_libs(${_t})
        endif()
    endforeach()
endif()

# =========================================================================
# Integration Tests (require model downloads)
# =========================================================================
option(LIBEMBEDDING_INTEGRATION_TESTS "Enable integration tests (require network)" OFF)

if(LIBEMBEDDING_INTEGRATION_TESTS)
    add_executable(test_text_embedding test_text_embedding.cpp)
    target_link_libraries(test_text_embedding PRIVATE libembedding)
    add_test(NAME test_text_embedding COMMAND test_text_embedding)

    add_executable(test_introspection test_introspection.cpp)
    target_link_libraries(test_introspection PRIVATE libembedding)
    add_test(NAME test_introspection COMMAND test_introspection)

    add_executable(test_sparse_embedding test_sparse_embedding.cpp)
    target_link_libraries(test_sparse_embedding PRIVATE libembedding)
    add_test(NAME test_sparse_embedding COMMAND test_sparse_embedding)

    add_executable(test_reranker test_reranker.cpp)
    target_link_libraries(test_reranker PRIVATE libembedding)
    add_test(NAME test_reranker COMMAND test_reranker)

    # llama.cpp backend tests (standalone - compiles implementation directly)
    add_executable(test_llama_embedding test_llama_embedding.cpp)
    target_include_directories(test_llama_embedding PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party
        ${CMAKE_SOURCE_DIR}/third_party/cJSON
        ${llama.cpp_SOURCE_DIR}/include
        ${llama.cpp_SOURCE_DIR}/ggml/include
    )
    target_compile_definitions(test_llama_embedding PRIVATE
        LIBEMBEDDING_IMPLEMENTATION
    )
    target_link_libraries(test_llama_embedding PRIVATE llama ggml)
    add_test(NAME test_llama_embedding COMMAND test_llama_embedding)

    if(NOT LIBEMBEDDING_NO_IMAGE)
        add_executable(test_image_embedding test_image_embedding.cpp)
        target_link_libraries(test_image_embedding PRIVATE libembedding)
        add_test(NAME test_image_embedding COMMAND test_image_embedding)
    endif()

    if(WIN32)
        foreach(_t test_text_embedding test_introspection test_sparse_embedding test_reranker)
            copy_runtime_dlls(${_t})
        endforeach()
        copy_runtime_dlls(test_llama_embedding)
        if(NOT LIBEMBEDDING_NO_IMAGE)
            copy_runtime_dlls(test_image_embedding)
        endif()
    elseif(ONNXRuntime_LIBRARIES)
        foreach(_t test_text_embedding test_introspection test_sparse_embedding test_reranker)
            copy_runtime_libs(${_t})
        endforeach()
        copy_runtime_libs(test_llama_embedding)
        if(NOT LIBEMBEDDING_NO_IMAGE)
            copy_runtime_libs(test_image_embedding)
        endif()
    endif()
endif()
