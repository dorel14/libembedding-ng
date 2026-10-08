/*
 * libembedding - detail/gguf/gguf_probe.hpp
 * Thin, typed adapter over ggml's GGUF reader.
 *
 * Deliberately this file re-implements nothing. The binary format is parsed by
 * the gguf reader already vendored with llama.cpp, which libembedding links
 * against; a fourth parser inside this project would be a second source of
 * truth for the same layout, free to drift out of step with the first.
 *
 * What this layer adds is the two things the C API of gguf.h cannot express
 * directly and that every caller would otherwise rewrite:
 *
 *   - *typed, checked* accessors, so reading a u32 out of a string value is a
 *     rejected read rather than a garbage number;
 *   - key lookup by architecture ("bert.vocab_size"), so a caller never has to
 *     hardcode the architecture prefix.
 *
 * No semantics live here. What the keys *mean* belongs to gguf_spec.hpp.
 *
 * Auteur: David Orel
 * Version: 1.10.1
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_GGUF_PROBE_HPP
#define LIBEMBEDDING_GGUF_PROBE_HPP

#include <gguf.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace lembed {
namespace gguf {

/* Read-only handle around a parsed GGUF header.
 *
 * Owns nothing but the gguf context and guarantees it is released, so a caller
 * cannot leak one on an early return. Values are read through this type only;
 * the raw context is not exposed, which keeps the ggml type out of the rest of
 * the library. */
class Probe {
public:
    Probe() = default;
    ~Probe() { close(); }

    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    Probe(Probe&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }

    Probe& operator=(Probe&& other) noexcept {
        if (this != &other) {
            close();
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }

    /* Opens a file for header inspection only.
     *
     * no_alloc is set because we want names, types and scalar metadata, never
     * tensor contents: allocating the data blob would read gigabytes to answer
     * a question about a few hundred header bytes. */
    bool open(const char* path) {
        close();
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;
        ctx_ = gguf_init_from_file(path, params);
        return ctx_ != nullptr;
    }

    void close() {
        if (ctx_) {
            gguf_free(ctx_);
            ctx_ = nullptr;
        }
    }

    /* Takes over a gguf context somebody else already opened, instead of
     * opening the file a second time. The sparse runtime needs the header *and*
     * the tensor data from one parse: gguf_init_from_file reads the data blob
     * only when no_alloc is false, and Probe::open reads it only when it is
     * true, so there is no single option that serves both. Taking the context
     * keeps one parse of the 30522-entry vocabulary. Ownership transfers to the
     * Probe, which frees it in close(). */
    bool adopt(gguf_context* ctx) {
        close();
        ctx_ = ctx;
        return ctx_ != nullptr;
    }

    /* Hands the context back to the caller and forgets it, so ownership can
     * move without a double free. Returns nullptr when there was nothing. */
    gguf_context* release() {
        gguf_context* ctx = ctx_;
        ctx_ = nullptr;
        return ctx;
    }

    /* The vocabulary lives in an array of strings, not a scalar, so it needs
     * its own accessor. Returns false when the key is absent or is not an
     * array of strings. */
    bool get_arr_str(const char* key, std::vector<std::string>& out) const {
        out.clear();
        if (!ctx_) return false;
        const int64_t id = gguf_find_key(ctx_, key);
        if (id < 0 || gguf_get_kv_type(ctx_, id) != GGUF_TYPE_ARRAY) return false;
        /* GGUF arrays are homogeneous and gguf_get_arr_type reports the single
         * element type, not a pointer to a per-element type list. */
        if (gguf_get_arr_type(ctx_, id) != GGUF_TYPE_STRING) return false;
        const int64_t n = gguf_get_arr_n(ctx_, id);
        if (n <= 0) return false;
        out.reserve((size_t)n);
        for (int64_t i = 0; i < n; i++) {
            const char* v = gguf_get_arr_str(ctx_, id, i);
            if (!v) return false;
            out.emplace_back(v);
        }
        return true;
    }

    bool valid() const { return ctx_ != nullptr; }

    int64_t n_tensors() const {
        return ctx_ ? gguf_get_n_tensors(ctx_) : 0;
    }

    /* ---- typed key access -------------------------------------------------
     *
     * Each accessor checks the stored type before converting. gguf_get_val_u32
     * on a string key returns whatever happens to be in that memory; here it
     * returns false and the caller reports the key as missing, which is the
     * difference between a clean refusal and a plausible wrong number. */

    bool has(const char* key) const {
        return ctx_ && gguf_find_key(ctx_, key) >= 0;
    }

    bool get_u32(const char* key, uint32_t& out) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_key(ctx_, key);
        if (id < 0 || gguf_get_kv_type(ctx_, id) != GGUF_TYPE_UINT32) return false;
        out = gguf_get_val_u32(ctx_, id);
        return true;
    }

    /* Signed and unsigned 32-bit are both accepted for the same reason bools
     * are: converters disagree on which they emit for a count or an id, and
     * rejecting the unsigned spelling would report a perfectly good file as
     * missing its vocabulary. The unsigned path is range-checked. */
    bool get_i32(const char* key, int32_t& out) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_key(ctx_, key);
        if (id < 0) return false;
        const gguf_type t = gguf_get_kv_type(ctx_, id);
        if (t == GGUF_TYPE_INT32) {
            out = gguf_get_val_i32(ctx_, id);
            return true;
        }
        if (t == GGUF_TYPE_UINT32) {
            const uint32_t v = gguf_get_val_u32(ctx_, id);
            if (v > (uint32_t)INT32_MAX) return false;
            out = (int32_t)v;
            return true;
        }
        return false;
    }

    bool get_str(const char* key, std::string& out) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_key(ctx_, key);
        if (id < 0 || gguf_get_kv_type(ctx_, id) != GGUF_TYPE_STRING) return false;
        const char* v = gguf_get_val_str(ctx_, id);
        if (!v) return false;
        out.assign(v);
        return true;
    }

    /* Floats are stored as F32, but a hand-written converter may well have used
     * F64 for something like a LayerNorm epsilon, so both are accepted. Reading
     * the wrong width out of a GGUF_TYPE_FLOAT32 slot yields a plausible
     * epsilon, which is precisely the failure this check exists to prevent. */
    bool get_f32(const char* key, float& out) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_key(ctx_, key);
        if (id < 0) return false;
        const gguf_type t = gguf_get_kv_type(ctx_, id);
        if (t == GGUF_TYPE_FLOAT32) {
            out = gguf_get_val_f32(ctx_, id);
            return true;
        }
        if (t == GGUF_TYPE_FLOAT64) {
            out = (float)gguf_get_val_f64(ctx_, id);
            return true;
        }
        return false;
    }

    bool get_arch_f32(const char* arch, const char* field, float& out) const {
        if (!arch || !*arch || !field) return false;
        const std::string key = std::string(arch) + "." + field;
        return get_f32(key.c_str(), out);
    }

    /* gguf stores booleans as int8; a converter is free to have written 0/1 as
     * a uint8 instead, so accept the unsigned spelling too. */
    bool get_bool(const char* key, bool& out) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_key(ctx_, key);
        if (id < 0) return false;
        const gguf_type t = gguf_get_kv_type(ctx_, id);
        if (t == GGUF_TYPE_BOOL) {
            out = gguf_get_val_bool(ctx_, id) != 0;
            return true;
        }
        if (t == GGUF_TYPE_UINT8) {
            out = gguf_get_val_u8(ctx_, id) != 0;
            return true;
        }
        return false;
    }

    /* Builds "<arch>.<field>" and looks it up.
     *
     * This is the indirection that keeps architecture names out of the rest of
     * the library: nothing else has to know that the vocabulary size lives under
     * "bert.vocab_size" today and would live under something else for another
     * architecture. */
    bool get_arch_u32(const char* arch, const char* field, uint32_t& out) const {
        if (!arch || !*arch || !field) return false;
        const std::string key = std::string(arch) + "." + field;
        return get_u32(key.c_str(), out);
    }

    bool get_arch_i32(const char* arch, const char* field, int32_t& out) const {
        if (!arch || !*arch || !field) return false;
        const std::string key = std::string(arch) + "." + field;
        return get_i32(key.c_str(), out);
    }

    bool get_arch_bool(const char* arch, const char* field, bool& out) const {
        if (!arch || !*arch || !field) return false;
        const std::string key = std::string(arch) + "." + field;
        return get_bool(key.c_str(), out);
    }

    bool has_arch(const char* arch, const char* field) const {
        if (!arch || !*arch || !field) return false;
        const std::string key = std::string(arch) + "." + field;
        return has(key.c_str());
    }

    /* ---- tensors --------------------------------------------------------- */

    bool has_tensor(const char* name) const {
        return ctx_ && gguf_find_tensor(ctx_, name) >= 0;
    }

    int64_t tensor_id(const char* name) const {
        return ctx_ ? gguf_find_tensor(ctx_, name) : -1;
    }

    /* Number of *elements* of a tensor. Returns false when the tensor is absent,
     * so a caller cannot size a buffer from a shape it imagined.
     *
     * Computed from the shape, not from gguf_get_tensor_size(): that function
     * returns ggml_nbytes, i.e. BYTES, so the first version of this accessor
     * reported a byte count under a name and a comment that both said
     * "elements". It happened to be harmless only because the one caller
     * compared it against a vocabulary size, and an F32 byte count of a
     * [1, hidden] weight lands near the vocabulary by coincidence rather than by
     * construction. Deriving the product from `ne` is the only way that holds for
     * every quantisation. */
    bool tensor_numel(const char* name, int64_t& out) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_tensor(ctx_, name);
        if (id < 0) return false;
        const int64_t* ne = gguf_get_tensor_ne(ctx_, id);
        if (!ne) return false;
        int64_t n = 1;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (ne[d] <= 0) return false;  /* a 0 dimension means a malformed file */
            n *= ne[d];
        }
        out = n;
        return true;
    }

    /* Dimensions of a tensor, most-significant first as ggml stores them.
     * ne[0] is the fastest-moving axis. Returns false when absent. */
    bool tensor_dims(const char* name, int64_t out_ne[GGML_MAX_DIMS],
                     int& out_n_dims) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_tensor(ctx_, name);
        if (id < 0) return false;
        const int64_t* ne = gguf_get_tensor_ne(ctx_, id);
        if (!ne) return false;
        out_n_dims = 0;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            out_ne[d] = ne[d];
            if (ne[d] > 1) out_n_dims = d + 1;
        }
        return true;
    }

private:
    gguf_context* ctx_ = nullptr;
};

} /* namespace gguf */
} /* namespace lembed */

#endif /* LIBEMBEDDING_GGUF_PROBE_HPP */