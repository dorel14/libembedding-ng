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

    /* Number of elements (not bytes) of a tensor. Returns false when the tensor
     * is absent, so a caller cannot size a buffer from a shape it imagined. */
    bool tensor_numel(const char* name, int64_t& out) const {
        if (!ctx_) return false;
        const int64_t id = gguf_find_tensor(ctx_, name);
        if (id < 0) return false;
        out = (int64_t)gguf_get_tensor_size(ctx_, id);
        return true;
    }

private:
    gguf_context* ctx_ = nullptr;
};

} /* namespace gguf */
} /* namespace lembed */

#endif /* LIBEMBEDDING_GGUF_PROBE_HPP */