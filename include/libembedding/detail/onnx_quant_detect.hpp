/*
 * libembedding - detail/onnx_quant_detect.hpp
 * Detects ONNX graphs that quantize their activations at run time
 *
 * Auteur: David Orel
 * Version: 1.16.7
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_ONNX_QUANT_DETECT_HPP
#define LIBEMBEDDING_DETAIL_ONNX_QUANT_DETECT_HPP

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace lembed { namespace detail {

/* Why this exists (T46). DynamicQuantizeLinear, and ORT's fused
 * DynamicQuantizeMatMul, compute their int8 scale over the whole input tensor.
 * In a padded batch that tensor holds every text of the batch, so a text's
 * vector depends on its neighbours and on the padding: measured on gte-base,
 * cosine 0.970 min against the same text run alone (FP32: 1.00000). Batching
 * such a model makes embeddings non-reproducible (the LRU cache and reference
 * vectors disagree with a fresh run) for a throughput gain that only exists on
 * short texts. Those models therefore run one text per ORT call.
 *
 * Op types are stored as plain strings in the ONNX protobuf, so a byte search
 * is exact enough: the marker cannot appear inside a tensor name by accident
 * in any exporter we know of, and a false positive only costs throughput. */
static const char kDynamicQuantMarker[] = "DynamicQuantize";

/* Boyer-Moore-Horspool: weight bytes rarely match the marker's last byte, so
 * the search mostly skips the marker's full length per step. A plain
 * std::search on a 540 MB FP32 file added ~1.3 s to model creation. */
inline bool onnx_bytes_use_dynamic_quantization(const void* data, size_t size) {
    if (!data || size == 0) return false;
    const char* begin = static_cast<const char*>(data);
    const char* end = begin + size;
    const char* m = kDynamicQuantMarker;
    const std::boyer_moore_horspool_searcher<const char*> searcher(
        m, m + (sizeof(kDynamicQuantMarker) - 1));
    return std::search(begin, end, searcher) != end;
}

/* Reads the file in chunks (no full copy in memory); a chunk keeps the tail of
 * the previous one so a marker split across two reads is still found. An
 * unreadable file counts as quantized: one text per run is the safe choice,
 * and the ORT load that follows reports the real error anyway. */
inline bool onnx_file_uses_dynamic_quantization(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return true;
    const size_t overlap = sizeof(kDynamicQuantMarker) - 2;
    const size_t chunk = (size_t)1 << 20;
    std::vector<char> buf(chunk + overlap);
    size_t kept = 0;
    for (;;) {
        in.read(buf.data() + kept, (std::streamsize)chunk);
        const size_t got = (size_t)in.gcount();
        if (got == 0) break;
        const size_t len = kept + got;
        if (onnx_bytes_use_dynamic_quantization(buf.data(), len)) return true;
        kept = std::min(overlap, len);
        std::memmove(buf.data(), buf.data() + len - kept, kept);
    }
    return in.bad();
}

}} // namespace lembed::detail

#endif /* LIBEMBEDDING_DETAIL_ONNX_QUANT_DETECT_HPP */
