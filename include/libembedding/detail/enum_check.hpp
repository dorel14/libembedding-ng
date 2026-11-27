/*
 * libembedding - detail/enum_check.hpp
 * Range validation of enum values that come from the caller.
 *
 * Auteur: David Orel
 * Version: 1.16.7
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_ENUM_CHECK_HPP
#define LIBEMBEDDING_ENUM_CHECK_HPP

#include "../types.h"
#include "../error.h"

/* Several implementation headers are included from inside an extern "C"
 * region, which forbids templates: force C++ linkage for everything below. */
extern "C++" {

#include <cstdio>
#include <cstring>

/*
 * Why this reads the bytes and not the enum.
 *
 * A C caller can pass any int where the header says `lembed_xxx_t`. In C++ an
 * enum object holding a value outside its range is undefined behaviour at the
 * moment it is *read* (UBSan `enum`), and the optimiser may assume the value is
 * in range and delete a `< 0 || >= COUNT` test written on the enum itself. So
 * the check must not load the enum-typed lvalue. enum_raw() binds a reference
 * (no load) and copies the object representation into an int with memcpy,
 * which is defined for any bit pattern. Once the raw value passed the range
 * test, the ordinary enum-typed code after it is well defined.
 *
 * None of the public enums has a fixed underlying type, and all of them are
 * int-sized on the supported compilers; the static_assert in enum_raw()
 * documents and enforces that.
 */
namespace lembed { namespace detail {

template <class E>
inline int enum_raw(const E& e) {
    static_assert(sizeof(E) == sizeof(int), "public enums are int-sized");
    int v;
    std::memcpy(&v, &e, sizeof v);
    return v;
}

/* LEMBED_OK when lo <= raw <= hi. Otherwise sets "Invalid <what> value <raw>
 * (expected <lo>..<hi>)" and returns LEMBED_ERROR_INVALID_ARGUMENT. */
inline lembed_status_t check_enum_range(int raw, int lo, int hi, const char* what) {
    if (raw >= lo && raw <= hi) return LEMBED_OK;
    char msg[160];
    std::snprintf(msg, sizeof msg, "Invalid %s value %d (expected %d..%d)",
                  what, raw, lo, hi);
    set_error(msg);
    return LEMBED_ERROR_INVALID_ARGUMENT;
}

template <class E>
inline lembed_status_t check_enum(const E& e, int lo, int hi, const char* what) {
    return check_enum_range(enum_raw(e), lo, hi, what);
}

/* An unknown provider keeps the status it always had: UNSUPPORTED, with the
 * message onnx_provider_precheck() gives ("not silently treated as CPU", see
 * test_backend_introspection). Only the read changes: it goes through the raw
 * value. This also covers the GGUF entry points, which never reach the ONNX
 * precheck. */
inline lembed_status_t check_provider(const lembed_execution_provider_t& v) {
    const int raw = enum_raw(v);
    if (raw >= LEMBED_PROVIDER_CPU && raw <= LEMBED_PROVIDER_LLAMACPP) return LEMBED_OK;
    char msg[240];
    std::snprintf(msg, sizeof msg,
                  "Execution provider %d was requested but is not available: "
                  "unknown lembed_execution_provider_t value. "
                  "Use LEMBED_PROVIDER_CPU (provider=\"cpu\") to run on the CPU.", raw);
    set_error(msg);
    return LEMBED_ERROR_UNSUPPORTED;
}
inline lembed_status_t check_backend(const lembed_backend_t& v) {
    return check_enum(v, LEMBED_BACKEND_ONNX, LEMBED_BACKEND_AUTO, "backend");
}
inline lembed_status_t check_text_model(const lembed_text_model_t& v) {
    return check_enum(v, 0, LEMBED_TEXT_MODEL_COUNT - 1, "text model");
}
inline lembed_status_t check_sparse_model(const lembed_sparse_model_t& v) {
    return check_enum(v, 0, LEMBED_SPARSE_MODEL_COUNT - 1, "sparse model");
}
inline lembed_status_t check_image_model(const lembed_image_model_t& v) {
    return check_enum(v, 0, LEMBED_IMAGE_MODEL_COUNT - 1, "image model");
}
inline lembed_status_t check_reranker_model(const lembed_reranker_model_t& v) {
    return check_enum(v, 0, LEMBED_RERANKER_MODEL_COUNT - 1, "reranker model");
}
inline lembed_status_t check_multivector_model(const lembed_multivector_model_t& v) {
    return check_enum(v, 0, LEMBED_MULTIVECTOR_MODEL_COUNT - 1, "multivector model");
}
inline lembed_status_t check_pooling(const lembed_pooling_t& v) {
    return check_enum(v, LEMBED_POOLING_CLS, LEMBED_POOLING_MEAN, "pooling");
}
/* int fields documented as "lembed_quantization_t" / "lembed_batch_strategy_t" */
inline lembed_status_t check_quantization(int raw) {
    return check_enum_range(raw, LEMBED_QUANTIZATION_NONE, LEMBED_QUANTIZATION_FP16,
                            "quantization");
}
inline lembed_status_t check_batch_strategy(int raw) {
    return check_enum_range(raw, LEMBED_BATCH_SEQUENTIAL, LEMBED_BATCH_LENGTH_BUCKET,
                            "batch strategy");
}

/* Whole-struct checks. `need_model` is false for the *_from_path / GGUF entry
 * points, which never read the registry model field. */
inline lembed_status_t check_text_options(const lembed_text_options_t* o, bool need_model) {
    lembed_status_t s = check_provider(o->provider);
    if (s == LEMBED_OK) s = check_backend(o->backend);
    if (s == LEMBED_OK && need_model) s = check_text_model(o->model);
    if (s == LEMBED_OK) s = check_batch_strategy(o->batch_strategy);
    return s;
}
inline lembed_status_t check_sparse_options(const lembed_sparse_options_t* o, bool need_model) {
    lembed_status_t s = check_provider(o->provider);
    if (s == LEMBED_OK && need_model) s = check_sparse_model(o->model);
    return s;
}
inline lembed_status_t check_image_options(const lembed_image_options_t* o, bool need_model) {
    lembed_status_t s = check_provider(o->provider);
    if (s == LEMBED_OK && need_model) s = check_image_model(o->model);
    return s;
}
inline lembed_status_t check_reranker_options(const lembed_reranker_options_t* o, bool need_model) {
    lembed_status_t s = check_provider(o->provider);
    if (s == LEMBED_OK) s = check_backend(o->backend);
    if (s == LEMBED_OK && need_model) s = check_reranker_model(o->model);
    return s;
}

}} /* namespace lembed::detail */

} /* extern "C++" */

#endif /* LIBEMBEDDING_ENUM_CHECK_HPP */
