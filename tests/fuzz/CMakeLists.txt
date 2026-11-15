# libembedding - tests/fuzz/CMakeLists.txt
# libFuzzer harnesses, built only with -DLIBEMBEDDING_BUILD_FUZZERS=ON (clang).
# The sanitizer and coverage flags are set for the whole build in the root
# CMakeLists.txt; only the harness links libFuzzer's main().
#
# Auteur: David Orel
# Version: 1.14.0
#
# SPDX-License-Identifier: MIT

add_executable(fuzz_gguf_loader fuzz_gguf_loader.cpp)
target_link_libraries(fuzz_gguf_loader PRIVATE libembedding ggml)
target_compile_options(fuzz_gguf_loader PRIVATE -fsanitize=fuzzer)
target_link_options(fuzz_gguf_loader PRIVATE -fsanitize=fuzzer)
