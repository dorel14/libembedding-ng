/*
 * libembedding - detail/onnx_session_impl.hpp
 * ONNX Runtime C API wrapper
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_ONNX_SESSION_IMPL_HPP
#define LIBEMBEDDING_DETAIL_ONNX_SESSION_IMPL_HPP

#include <onnxruntime_c_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* RAII helper to release ORT objects */

/* Singleton ORT environment */
inline const OrtApi* ort_api() {
    return OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

inline OrtEnv* ort_env() {
    static OrtEnv* env = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() {
        const OrtApi* api = ort_api();
        OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "libembedding", &env);
        if (status) {
            const char* msg = api->GetErrorMessage(status);
            std::string err = std::string("Failed to create ORT env: ") + msg;
            api->ReleaseStatus(status);
            throw std::runtime_error(err);
        }
    });
    return env;
}

/* Check ORT status and throw on error */
inline void ort_check(OrtStatus* status) {
    if (status) {
        const OrtApi* api = ort_api();
        const char* msg = api->GetErrorMessage(status);
        std::string err(msg);
        api->ReleaseStatus(status);
        throw std::runtime_error(err);
    }
}

/* Wrapper around an ORT session */
class OnnxSession {
public:
    OnnxSession() : session_(nullptr), allocator_(nullptr) {}
    ~OnnxSession() {
        if (session_) ort_api()->ReleaseSession(session_);
    }

    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;
    OnnxSession(OnnxSession&& other) noexcept
        : session_(other.session_), allocator_(other.allocator_),
          input_names_(std::move(other.input_names_)),
          output_names_(std::move(other.output_names_)) {
        other.session_ = nullptr;
    }

    /* Create session from file path */
    void load_from_file(const char* model_path,
                        int num_threads = 0,
                        int provider = 0 /* CPU */,
                        int quantization = 0) {
        const OrtApi* api = ort_api();
        OrtSessionOptions* opts = nullptr;

        ort_check(api->CreateSessionOptions(&opts));
        ort_check(api->EnableMemPattern(opts));
        ort_check(api->EnableCpuMemArena(opts));

        /* Graph optimization level 3 (all optimizations) */
        ort_check(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

        /* Thread configuration */
        if (num_threads > 0) {
            ort_check(api->SetIntraOpNumThreads(opts, num_threads));
            ort_check(api->SetInterOpNumThreads(opts, 1));
        }

        /* Execution provider */
        apply_provider_config(api, opts, provider);
        apply_quantization_config(api, opts, quantization);

        #if defined(_WIN32) || defined(WIN32)
            std::wstring path_w;
            int n = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
            if (n > 0) {
                path_w.resize(n);
                MultiByteToWideChar(CP_UTF8, 0, model_path, -1, path_w.data(), n);
            }
            ort_check(api->CreateSession(ort_env(), path_w.c_str(), opts, &session_));
        #else
            ort_check(api->CreateSession(ort_env(), model_path, opts, &session_));
        #endif
        api->ReleaseSessionOptions(opts);

        ort_check(api->GetAllocatorWithDefaultOptions(&allocator_));
        query_io_names();
    }

    /* Create session from memory buffer */
    void load_from_memory(const void* data, size_t size,
                          int num_threads = 0,
                          int provider = 0,
                          int quantization = 0) {
        const OrtApi* api = ort_api();
        OrtSessionOptions* opts = nullptr;

        ort_check(api->CreateSessionOptions(&opts));
        ort_check(api->EnableMemPattern(opts));
        ort_check(api->EnableCpuMemArena(opts));
        ort_check(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

        if (num_threads > 0) {
            ort_check(api->SetIntraOpNumThreads(opts, num_threads));
            ort_check(api->SetInterOpNumThreads(opts, 1));
        }

        apply_provider_config(api, opts, provider);
        apply_quantization_config(api, opts, quantization);

        ort_check(api->CreateSessionFromArray(ort_env(), data, size, opts, &session_));
        api->ReleaseSessionOptions(opts);

        ort_check(api->GetAllocatorWithDefaultOptions(&allocator_));
        query_io_names();
    }

    /* Check if a specific input name exists */
    bool has_input(const char* name) const {
        for (const auto& n : input_names_) {
            if (n == name) return true;
        }
        return false;
    }

    /* Get output names */
    const std::vector<std::string>& output_names() const { return output_names_; }
    const std::vector<std::string>& input_names() const { return input_names_; }

    /* Run inference with named inputs
     * Returns output tensor data and shape info
     */
    struct TensorResult {
        std::vector<float> data;
        std::vector<int64_t> shape;
    };

    /* Run with input_ids and attention_mask (and optionally token_type_ids).
     * If preferred_output >= 0, only that output tensor is requested from ORT,
     * and the result vector will contain exactly one element at index 0. */
    std::vector<TensorResult> run(
            const int64_t* input_ids, const int64_t* attention_mask,
            const int64_t* token_type_ids, /* may be NULL */
            int batch_size, int seq_length,
            int preferred_output = -1) {

        const OrtApi* api = ort_api();

        /* Build input tensors */
        OrtMemoryInfo* mem_info = nullptr;
        ort_check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

        int64_t input_shape[2] = { batch_size, seq_length };
        size_t input_size = (size_t)batch_size * seq_length;

        std::vector<OrtValue*> input_values;
        std::vector<const char*> input_name_ptrs;

        /* input_ids */
        OrtValue* ids_val = nullptr;
        ort_check(api->CreateTensorWithDataAsOrtValue(
            mem_info, (void*)input_ids, input_size * sizeof(int64_t),
            input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ids_val));
        input_values.push_back(ids_val);
        input_name_ptrs.push_back("input_ids");

        /* attention_mask (if model accepts it) */
        OrtValue* mask_val = nullptr;
        if (has_input("attention_mask")) {
            ort_check(api->CreateTensorWithDataAsOrtValue(
                mem_info, (void*)attention_mask, input_size * sizeof(int64_t),
                input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &mask_val));
            input_values.push_back(mask_val);
            input_name_ptrs.push_back("attention_mask");
        }

        /* token_type_ids (if model accepts it) */
        OrtValue* type_val = nullptr;
        if (token_type_ids && has_input("token_type_ids")) {
            ort_check(api->CreateTensorWithDataAsOrtValue(
                mem_info, (void*)token_type_ids, input_size * sizeof(int64_t),
                input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &type_val));
            input_values.push_back(type_val);
            input_name_ptrs.push_back("token_type_ids");
        }

        /* Output names (request only the needed output if specified) */
        std::vector<const char*> output_name_ptrs;
        size_t num_outputs;
        if (preferred_output >= 0 && preferred_output < (int)output_names_.size()) {
            output_name_ptrs.push_back(output_names_[preferred_output].c_str());
            num_outputs = 1;
        } else {
            for (const auto& n : output_names_)
                output_name_ptrs.push_back(n.c_str());
            num_outputs = output_names_.size();
        }
        std::vector<OrtValue*> output_values(num_outputs, nullptr);

        /* Run inference */
        ort_check(api->Run(session_, nullptr,
                          input_name_ptrs.data(), input_values.data(),
                          input_name_ptrs.size(),
                          output_name_ptrs.data(), num_outputs,
                          output_values.data()));

        /* Extract results */
        std::vector<TensorResult> results;
        for (size_t i = 0; i < num_outputs; i++) {
            TensorResult tr;

            OrtTensorTypeAndShapeInfo* info = nullptr;
            ort_check(api->GetTensorTypeAndShape(output_values[i], &info));

            size_t ndim = 0;
            ort_check(api->GetDimensionsCount(info, &ndim));
            tr.shape.resize(ndim);
            ort_check(api->GetDimensions(info, tr.shape.data(), ndim));

            size_t total = 1;
            for (auto d : tr.shape) total *= (size_t)d;
            tr.data.resize(total);

            float* out_data = nullptr;
            ort_check(api->GetTensorMutableData(output_values[i], (void**)&out_data));
            std::memcpy(tr.data.data(), out_data, total * sizeof(float));

            api->ReleaseTensorTypeAndShapeInfo(info);
            api->ReleaseValue(output_values[i]);
            results.push_back(std::move(tr));
        }

        /* Cleanup inputs */
        for (auto v : input_values) api->ReleaseValue(v);
        api->ReleaseMemoryInfo(mem_info);

        return results;
    }

    /* Run with a single float input (for image models) */
    std::vector<TensorResult> run_float(
            const char* input_name,
            const float* data, const int64_t* shape, int ndim) {

        const OrtApi* api = ort_api();

        OrtMemoryInfo* mem_info = nullptr;
        ort_check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

        size_t total = 1;
        for (int i = 0; i < ndim; i++) total *= (size_t)shape[i];

        OrtValue* input_val = nullptr;
        ort_check(api->CreateTensorWithDataAsOrtValue(
            mem_info, (void*)data, total * sizeof(float),
            shape, ndim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_val));

        const char* input_names[] = { input_name };
        std::vector<const char*> output_name_ptrs;
        for (const auto& n : output_names_) {
            output_name_ptrs.push_back(n.c_str());
        }

        size_t num_outputs = output_names_.size();
        std::vector<OrtValue*> output_values(num_outputs, nullptr);

        ort_check(api->Run(session_, nullptr,
                          input_names, &input_val, 1,
                          output_name_ptrs.data(), num_outputs,
                          output_values.data()));

        std::vector<TensorResult> results;
        for (size_t i = 0; i < num_outputs; i++) {
            TensorResult tr;
            OrtTensorTypeAndShapeInfo* info = nullptr;
            ort_check(api->GetTensorTypeAndShape(output_values[i], &info));

            size_t od = 0;
            ort_check(api->GetDimensionsCount(info, &od));
            tr.shape.resize(od);
            ort_check(api->GetDimensions(info, tr.shape.data(), od));

            size_t ot = 1;
            for (auto d : tr.shape) ot *= (size_t)d;
            tr.data.resize(ot);

            float* out = nullptr;
            ort_check(api->GetTensorMutableData(output_values[i], (void**)&out));
            std::memcpy(tr.data.data(), out, ot * sizeof(float));

            api->ReleaseTensorTypeAndShapeInfo(info);
            api->ReleaseValue(output_values[i]);
            results.push_back(std::move(tr));
        }

        api->ReleaseValue(input_val);
        api->ReleaseMemoryInfo(mem_info);

        return results;
    }

    /* Select the best output tensor by key precedence */
    int select_output(const char* preferred_name = nullptr) const {
        /* If only one output, use it */
        if (output_names_.size() == 1) return 0;

        /* If preferred name given, look for it */
        if (preferred_name) {
            for (size_t i = 0; i < output_names_.size(); i++) {
                if (output_names_[i] == preferred_name) return (int)i;
            }
        }

        /* Output key precedence (matching fastembed-rs) */
        static const char* precedence[] = {
            "text_embeds",
            "last_hidden_state",
            "sentence_embedding",
            nullptr
        };

        for (int p = 0; precedence[p]; p++) {
            for (size_t i = 0; i < output_names_.size(); i++) {
                if (output_names_[i] == precedence[p]) return (int)i;
            }
        }

        /* Fallback to first output */
        return 0;
    }

private:
    OrtSession* session_;
    OrtAllocator* allocator_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    void query_io_names() {
        const OrtApi* api = ort_api();

        /* Input names */
        size_t num_inputs = 0;
        ort_check(api->SessionGetInputCount(session_, &num_inputs));
        for (size_t i = 0; i < num_inputs; i++) {
            char* name = nullptr;
            ort_check(api->SessionGetInputName(session_, i, allocator_, &name));
            input_names_.push_back(name);
            allocator_->Free(allocator_, name);
        }

        /* Output names */
        size_t num_outputs = 0;
        ort_check(api->SessionGetOutputCount(session_, &num_outputs));
        for (size_t i = 0; i < num_outputs; i++) {
            char* name = nullptr;
            ort_check(api->SessionGetOutputName(session_, i, allocator_, &name));
            output_names_.push_back(name);
            allocator_->Free(allocator_, name);
        }
    }

/* =========================================================================
 * Quantization configuration for quantized ONNX models.
 * Uses ORT session config entries (compatible with ORT >= 1.16).
 * For ORT >= 1.22+, consider migrating to OrtQuantizationConfig API.
 * ========================================================================= */
    void apply_quantization_config(const OrtApi* api, OrtSessionOptions* opts, int quantization) {
        switch (quantization) {
            case 1: /* LEMBED_QUANTIZATION_STATIC */
                ort_check(api->AddSessionConfigEntry(opts, "session.disable_quant_qdq", "0"));
                ort_check(api->AddSessionConfigEntry(opts, "session.enable_quant_qdq_cleanup", "1"));
                break;

            case 2: /* LEMBED_QUANTIZATION_DYNAMIC */
                ort_check(api->AddSessionConfigEntry(opts, "session.disable_prepacking", "0"));
                break;

            default: /* LEMBED_QUANTIZATION_NONE */
                break;
        }
    }

    /* =========================================================================
     * Provider configuration
     * 0 = CPU (default, no extra setup needed)
     * 1 = CUDA
     * 2 = CoreML
     * 3 = DirectML
     * 4 = TensorRT
     *
     * For non-CPU providers, the user must have the corresponding
     * ONNX Runtime provider library installed. We attempt to append
     * the provider and fall back silently to CPU on failure.
     * ========================================================================= */
    void apply_provider_config(const OrtApi* api, OrtSessionOptions* opts, int provider) {
#ifdef ORT_API_VERSION
        switch (provider) {
            case 1: { /* CUDA */
#ifdef USE_CUDA
                OrtCUDAProviderOptions cuda_opts;
                memset(&cuda_opts, 0, sizeof(cuda_opts));
                api->SessionOptionsAppendExecutionProvider_CUDA(opts, &cuda_opts);
#endif
                break;
            }
            case 2: { /* CoreML */
#if defined(__APPLE__)
                OrtStatus* s = OrtSessionOptionsAppendExecutionProvider_CoreML(opts, 0);
                if (s) api->ReleaseStatus(s); /* fall back to CPU */
#endif
                break;
            }
            case 3: { /* DirectML (Windows only, requires onnxruntime-dml) */
#if defined(_WIN32) || defined(WIN32)
#ifdef USE_DML
                OrtStatus* s = OrtSessionOptionsAppendExecutionProvider_Dml(opts, 0);
                if (s) api->ReleaseStatus(s); /* fall back to CPU */
#endif
#endif
                break;
            }
            case 4: { /* TensorRT */
#ifdef USE_TENSORRT
                OrtTensorRTProviderOptions trt_opts;
                memset(&trt_opts, 0, sizeof(trt_opts));
                api->SessionOptionsAppendExecutionProvider_TensorRT(opts, &trt_opts);
#endif
                break;
            }
            default:
                break;
        }
#endif
    }
};

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_DETAIL_ONNX_SESSION_IMPL_HPP */
