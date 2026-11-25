/*
 * test_enum_validation.cpp - out-of-range enum values at the C API boundary.
 *
 * A C caller can hand the library any int where a header says `lembed_xxx_t`.
 * Every entry point must answer LEMBED_ERROR_INVALID_ARGUMENT with a message in
 * lembed_last_error(), before touching a model, the network or a table indexed
 * by that value. No model and no network are needed: validation runs first.
 *
 * The bad values are built with memcpy into the enum object. Passing them by
 * value is itself an enum-typed read, so the functions that do it carry
 * NO_ENUM_SAN: the point of the test is the library's behaviour, and UBSan
 * `enum` would otherwise report the test's own call sites.
 *
 * Auteur: David Orel
 * Version: 1.16.3
 *
 * SPDX-License-Identifier: MIT
 */

/* No LIBEMBEDDING_IMPLEMENTATION here. The autotune entry points
 * (lembed_autotune_ex, lembed_reranker_autotune...) are defined only in
 * python/csrc/libembedding_shlib.cpp, which defines the implementation itself:
 * on Windows it is the DLL, elsewhere tests/CMakeLists.txt compiles it into
 * this executable. Defining it here too would duplicate every definition. */
#include <libembedding/libembedding.h>

#include <cstdio>
#include <cstring>
#include <string>

#if defined(__clang__)
#define NO_ENUM_SAN __attribute__((no_sanitize("enum")))
#elif defined(__GNUC__)
#define NO_ENUM_SAN __attribute__((no_sanitize("undefined")))
#else
#define NO_ENUM_SAN
#endif

static int passes = 0, failures = 0;

/* An enum object holding `v`, whatever the enum's range. */
template <class E>
static NO_ENUM_SAN E bad(int v) {
    E e;
    std::memcpy(&e, &v, sizeof v);
    return e;
}

/* Store `v` in an enum-typed field without an enum-typed write. */
template <class E>
static void set_raw(E& field, int v) {
    std::memcpy(&field, &v, sizeof v);
}

/* `expected` and `token` default to the general contract. An unknown execution
 * provider is the exception: it has always been UNSUPPORTED (see
 * test_backend_introspection), so it keeps that status and its own message. */
static void expect_status(lembed_status_t s, lembed_status_t expected, const char* token,
                          const char* what, int v) {
    const std::string msg = lembed_last_error();
    if (s != expected) {
        std::fprintf(stderr, "FAIL: %s = %d returned status %d, expected %d\n",
                     what, v, (int)s, (int)expected);
        failures++;
    } else if (msg.find(token) == std::string::npos) {
        std::fprintf(stderr, "FAIL: %s = %d: last_error has no \"%s\": '%s'\n",
                     what, v, token, msg.c_str());
        failures++;
    } else {
        passes++;
    }
}

static const int kBadValues[] = {999, -1};

#define EXPECT_INVALID(call, what, v) \
    expect_status((call), LEMBED_ERROR_INVALID_ARGUMENT, "Invalid", (what), (v))
#define EXPECT_BAD_PROVIDER(call, what, v) \
    expect_status((call), LEMBED_ERROR_UNSUPPORTED, "unknown lembed_execution_provider_t", (what), (v))

static NO_ENUM_SAN void test_registry() {
    for (int v : kBadValues) {
        lembed_model_info_t info;
        EXPECT_INVALID(lembed_get_text_model_info(bad<lembed_text_model_t>(v), &info),
                       "get_text_model_info", v);
        EXPECT_INVALID(lembed_get_sparse_model_info(bad<lembed_sparse_model_t>(v), &info),
                       "get_sparse_model_info", v);
        EXPECT_INVALID(lembed_get_image_model_info(bad<lembed_image_model_t>(v), &info),
                       "get_image_model_info", v);
        EXPECT_INVALID(lembed_get_reranker_model_info(bad<lembed_reranker_model_t>(v), &info),
                       "get_reranker_model_info", v);

        EXPECT_INVALID(lembed_get_multivector_model_info(bad<lembed_multivector_model_t>(v), &info),
                       "get_multivector_model_info", v);

        /* A registry name as model_dir: the provider check must fire before
         * any registry lookup, so no download is attempted. */
        {
            lembed_multivector_options_t mo = lembed_multivector_options_default();
            mo.model_dir = "answerdotai/answerai-colbert-small-v1";
            set_raw(mo.provider, v);
            lembed_multivector_t* mv = nullptr;
            EXPECT_BAD_PROVIDER(lembed_multivector_create(&mo, &mv), "multivector_create.provider", v);
            if (mv != nullptr) { std::fprintf(stderr, "FAIL: multivector handle created\n"); failures++; }
        }

        char* dir = nullptr;
        EXPECT_INVALID(lembed_ensure_text_model(bad<lembed_text_model_t>(v), nullptr, 0, 1, &dir),
                       "ensure_text_model", v);
        EXPECT_INVALID(lembed_ensure_sparse_model(bad<lembed_sparse_model_t>(v), nullptr, 0, 1, &dir),
                       "ensure_sparse_model", v);
#ifndef LIBEMBEDDING_NO_IMAGE
        EXPECT_INVALID(lembed_ensure_image_model(bad<lembed_image_model_t>(v), nullptr, 0, 1, &dir),
                       "ensure_image_model", v);
#endif
        EXPECT_INVALID(lembed_ensure_reranker_model(bad<lembed_reranker_model_t>(v), nullptr, 0, 1, &dir),
                       "ensure_reranker_model", v);
    }

    /* The COUNT sentinel itself is out of range, the last real entry is not. */
    lembed_model_info_t info;
    EXPECT_INVALID(lembed_get_text_model_info(LEMBED_TEXT_MODEL_COUNT, &info),
                   "get_text_model_info(COUNT)", (int)LEMBED_TEXT_MODEL_COUNT);
    lembed_status_t ok = lembed_get_text_model_info(
        (lembed_text_model_t)(LEMBED_TEXT_MODEL_COUNT - 1), &info);
    if (ok == LEMBED_OK) passes++;
    else { std::fprintf(stderr, "FAIL: last text model rejected\n"); failures++; }
}

static NO_ENUM_SAN void test_text() {
    for (int v : kBadValues) {
        lembed_text_embedding_t* out = nullptr;
        {
            lembed_text_options_t o = lembed_text_options_default();
            set_raw(o.provider, v);
            EXPECT_BAD_PROVIDER(lembed_text_embedding_create(&o, &out), "text.provider", v);
            EXPECT_BAD_PROVIDER(lembed_text_embedding_create_from_path("no_such_dir", &o, &out),
                                "text.provider (from_path)", v);
            EXPECT_BAD_PROVIDER(lembed_text_embedding_create_from_gguf_path("x.gguf", &o, &out),
                                "text.provider (gguf_path)", v);
        }
        {
            lembed_text_options_t o = lembed_text_options_default();
            set_raw(o.backend, v);
            EXPECT_INVALID(lembed_text_embedding_create(&o, &out), "text.backend", v);
            EXPECT_INVALID(lembed_text_embedding_create_from_path("no_such_dir", &o, &out),
                           "text.backend (from_path)", v);
            EXPECT_INVALID(lembed_text_embedding_create_from_gguf_path("x.gguf", &o, &out),
                           "text.backend (gguf_path)", v);
        }
        {
            lembed_text_options_t o = lembed_text_options_default();
            set_raw(o.model, v);
            EXPECT_INVALID(lembed_text_embedding_create(&o, &out), "text.model", v);
        }
        {
            lembed_text_options_t o = lembed_text_options_default();
            o.batch_strategy = v;
            EXPECT_INVALID(lembed_text_embedding_create(&o, &out), "text.batch_strategy", v);
        }
        {
            lembed_text_options_v2_t o = lembed_text_options_default_v2();
            o.quantization = v;
            EXPECT_INVALID(lembed_text_embedding_create_v2(&o, &out), "text_v2.quantization", v);
            o = lembed_text_options_default_v2();
            set_raw(o.base.provider, v);
            EXPECT_BAD_PROVIDER(lembed_text_embedding_create_v2(&o, &out), "text_v2.provider", v);
        }
        {
            static const unsigned char blob[] = {0};
            lembed_user_defined_model_t m;
            std::memset(&m, 0, sizeof m);
            m.onnx_data = blob;
            m.onnx_data_size = sizeof blob;
            m.tokenizer_json = blob;
            m.tokenizer_json_size = sizeof blob;
            EXPECT_BAD_PROVIDER(lembed_text_embedding_create_custom(
                                    &m, bad<lembed_execution_provider_t>(v), 1, &out),
                                "custom.provider", v);
            set_raw(m.pooling, v);
            EXPECT_INVALID(lembed_text_embedding_create_custom(
                               &m, LEMBED_PROVIDER_CPU, 1, &out),
                           "custom.pooling", v);
        }
    }
}

static NO_ENUM_SAN void test_sparse() {
    for (int v : kBadValues) {
        lembed_sparse_embedding_ctx_t* out = nullptr;
        {
            lembed_sparse_options_t o = lembed_sparse_options_default();
            set_raw(o.provider, v);
            EXPECT_BAD_PROVIDER(lembed_sparse_text_embedding_create(&o, &out), "sparse.provider", v);
            EXPECT_BAD_PROVIDER(lembed_sparse_text_embedding_create_from_path("no_such_dir", &o, &out),
                                "sparse.provider (from_path)", v);
            EXPECT_BAD_PROVIDER(lembed_sparse_text_embedding_create_from_gguf_path("x.gguf", &o, &out),
                                "sparse.provider (gguf_path)", v);
        }
        {
            lembed_sparse_options_t o = lembed_sparse_options_default();
            set_raw(o.model, v);
            EXPECT_INVALID(lembed_sparse_text_embedding_create(&o, &out), "sparse.model", v);
        }
    }
}

#ifndef LIBEMBEDDING_NO_IMAGE
static NO_ENUM_SAN void test_image() {
    for (int v : kBadValues) {
        lembed_image_embedding_t* out = nullptr;
        {
            lembed_image_options_t o = lembed_image_options_default();
            set_raw(o.provider, v);
            EXPECT_BAD_PROVIDER(lembed_image_embedding_create(&o, &out), "image.provider", v);
            EXPECT_BAD_PROVIDER(lembed_image_embedding_create_from_path("no_such_dir", &o, &out),
                                "image.provider (from_path)", v);
        }
        {
            lembed_image_options_t o = lembed_image_options_default();
            set_raw(o.model, v);
            EXPECT_INVALID(lembed_image_embedding_create(&o, &out), "image.model", v);
        }
    }
}
#endif

static NO_ENUM_SAN void test_reranker() {
    for (int v : kBadValues) {
        lembed_reranker_t* out = nullptr;
        {
            lembed_reranker_options_t o = lembed_reranker_options_default();
            set_raw(o.provider, v);
            EXPECT_BAD_PROVIDER(lembed_reranker_create(&o, &out), "reranker.provider", v);
            EXPECT_BAD_PROVIDER(lembed_reranker_create_from_path("no_such_dir", &o, &out),
                                "reranker.provider (from_path)", v);
            EXPECT_BAD_PROVIDER(lembed_reranker_create_from_gguf_path("x.gguf", &o, &out),
                                "reranker.provider (gguf_path)", v);
        }
        {
            lembed_reranker_options_t o = lembed_reranker_options_default();
            set_raw(o.backend, v);
            EXPECT_INVALID(lembed_reranker_create(&o, &out), "reranker.backend", v);
        }
        {
            lembed_reranker_options_t o = lembed_reranker_options_default();
            set_raw(o.model, v);
            EXPECT_INVALID(lembed_reranker_create(&o, &out), "reranker.model", v);
        }
        {
            lembed_reranker_options_v2_t o;
            std::memset(&o, 0, sizeof o);
            o.base = lembed_reranker_options_default();
            o.quantization = v;
            EXPECT_INVALID(lembed_reranker_create_v2(&o, &out), "reranker_v2.quantization", v);
        }
    }
}

static NO_ENUM_SAN void test_mode_and_autotune() {
    for (int v : kBadValues) {
        const char* s = lembed_mode_to_string(bad<lembed_embedding_mode_t>(v));
        if (s && std::strcmp(s, "unknown") == 0) passes++;
        else { std::fprintf(stderr, "FAIL: mode_to_string(%d) = '%s'\n", v, s ? s : "(null)"); failures++; }

        lembed_text_model_t m = lembed_recommended_model_for_mode(bad<lembed_embedding_mode_t>(v));
        if ((int)m == (int)LEMBED_TEXT_MODEL_COUNT) passes++;
        else { std::fprintf(stderr, "FAIL: recommended_model_for_mode(%d) = %d\n", v, (int)m); failures++; }

        lembed_tuning_result_t tr;
        EXPECT_INVALID(lembed_autotune("sentence-transformers/all-MiniLM-L6-v2",
                                       bad<lembed_autotune_mode_t>(v), &tr),
                       "autotune.mode", v);
        const char* texts[] = {"a"};
        EXPECT_INVALID(lembed_autotune_custom("sentence-transformers/all-MiniLM-L6-v2", texts, 1,
                                              bad<lembed_autotune_mode_t>(v), &tr),
                       "autotune_custom.mode", v);

        lembed_unified_tuning_result_t ur;
        EXPECT_INVALID(lembed_autotune_unified(bad<lembed_task_t>(v), "m",
                                               LEMBED_AUTOTUNE_QUICK, &ur),
                       "autotune_unified.task", v);
        EXPECT_INVALID(lembed_autotune_unified(LEMBED_TASK_EMBEDDING, "m",
                                               bad<lembed_autotune_mode_t>(v), &ur),
                       "autotune_unified.mode", v);
        EXPECT_INVALID(lembed_autotune_unified_config(bad<lembed_task_t>(v), "m", 10.0, &ur),
                       "autotune_unified_config.task", v);
        lembed_autotune_unified_clear_cache(bad<lembed_task_t>(v), nullptr); /* void: no crash */

        {
            lembed_autotune_options_t o = lembed_autotune_options_default(LEMBED_TASK_EMBEDDING);
            set_raw(o.task, v);
            EXPECT_INVALID(lembed_autotune_ex("m", &o, &ur), "autotune_ex.task", v);
            o = lembed_autotune_options_default(LEMBED_TASK_EMBEDDING);
            set_raw(o.mode, v);
            EXPECT_INVALID(lembed_autotune_ex("m", &o, &ur), "autotune_ex.mode", v);
            o = lembed_autotune_options_default(LEMBED_TASK_EMBEDDING);
            set_raw(o.objective, v);
            EXPECT_INVALID(lembed_autotune_ex("m", &o, &ur), "autotune_ex.objective", v);
        }
        {
            /* An unknown task must not leak into the returned defaults. */
            lembed_autotune_options_t o = lembed_autotune_options_default(bad<lembed_task_t>(v));
            int raw_task;
            std::memcpy(&raw_task, &o.task, sizeof raw_task);
            if (raw_task == (int)LEMBED_TASK_EMBEDDING) passes++;
            else { std::fprintf(stderr, "FAIL: options_default(%d).task = %d\n", v, raw_task); failures++; }
        }

        lembed_reranker_tuning_result_t rr;
        EXPECT_INVALID(lembed_reranker_autotune("m", bad<lembed_autotune_mode_t>(v),
                                                LEMBED_OBJECTIVE_BALANCED, &rr),
                       "reranker_autotune.mode", v);
        EXPECT_INVALID(lembed_reranker_autotune("m", LEMBED_AUTOTUNE_QUICK,
                                                bad<lembed_objective_t>(v), &rr),
                       "reranker_autotune.objective", v);
        EXPECT_INVALID(lembed_reranker_auto_config("m", 100.0, bad<lembed_objective_t>(v), &rr),
                       "reranker_auto_config.objective", v);
        EXPECT_INVALID(lembed_reranker_auto_config_profile(
                           "m", bad<lembed_reranker_profile_t>(v), &rr),
                       "reranker_auto_config_profile.profile", v);
        EXPECT_INVALID(lembed_reranker_autotune_custom("m", texts, 1,
                                                       bad<lembed_autotune_mode_t>(v),
                                                       LEMBED_OBJECTIVE_BALANCED, &rr),
                       "reranker_autotune_custom.mode", v);
        EXPECT_INVALID(lembed_reranker_autotune_custom("m", texts, 1, LEMBED_AUTOTUNE_QUICK,
                                                       bad<lembed_objective_t>(v), &rr),
                       "reranker_autotune_custom.objective", v);

        lembed_sparse_tuning_result_t sr;
        EXPECT_INVALID(lembed_sparse_autotune("m", bad<lembed_autotune_mode_t>(v), &sr),
                       "sparse_autotune.mode", v);
#ifndef LIBEMBEDDING_NO_IMAGE
        lembed_image_tuning_result_t ir;
        EXPECT_INVALID(lembed_image_autotune("m", bad<lembed_autotune_mode_t>(v), &ir),
                       "image_autotune.mode", v);
#endif
    }
}

int main(void) {
    test_registry();
    test_text();
    test_sparse();
#ifndef LIBEMBEDDING_NO_IMAGE
    test_image();
#endif
    test_reranker();
    test_mode_and_autotune();

    std::printf("test_enum_validation: %d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
