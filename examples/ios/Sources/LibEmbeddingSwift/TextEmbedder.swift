//
// libembedding - TextEmbedder.swift
// Minimal Swift wrapper around the libembedding C API for dense text
// embeddings: create a context, embed strings, free it on deinit.
//
// Auteur: David Orel
// Version: 1.13.0
//
// SPDX-License-Identifier: MIT

import Foundation
import libembedding

/// A failed libembedding call: the status code and the thread-local message.
public struct LibEmbeddingError: Error, CustomStringConvertible {
    public let status: lembed_status_t
    public let message: String

    init(_ status: lembed_status_t) {
        self.status = status
        self.message = String(cString: lembed_last_error())
    }

    public var description: String {
        "libembedding error \(status.rawValue): \(message)"
    }
}

/// A dense text embedding model. A registry model is downloaded on first use
/// into $HOME/Library/Caches/libembedding; on iOS only light models (150 MB or
/// less, see `lembed_mobile_model_limit_mb()`) load by name.
public final class TextEmbedder {
    private let ctx: OpaquePointer
    public let dim: Int

    public init(model: lembed_text_model_t = LEMBED_TEXT_BGE_SMALL_EN_V15) throws {
        var options = lembed_text_options_default()
        options.model = model
        options.show_download_progress = 0
        var out: OpaquePointer?
        let status = lembed_text_embedding_create(&options, &out)
        guard status == LEMBED_OK, let created = out else {
            throw LibEmbeddingError(status)
        }
        ctx = created
        dim = Int(lembed_text_embedding_dim(created))
    }

    deinit {
        lembed_text_embedding_free(ctx)
    }

    /// One L2-normalised vector of `dim` floats per text.
    public func embed(_ texts: [String]) throws -> [[Float]] {
        let copies = texts.map { strdup($0) }
        defer { copies.forEach { free($0) } }
        let pointers: [UnsafePointer<CChar>?] = copies.map { UnsafePointer($0) }

        var result = lembed_embeddings_t()
        let status = lembed_text_embedding_embed(ctx, pointers, Int32(pointers.count), 0, &result)
        guard status == LEMBED_OK else {
            throw LibEmbeddingError(status)
        }
        defer { lembed_embeddings_free(&result) }

        let count = Int(result.num_embeddings)
        let width = Int(result.dim)
        let flat = UnsafeBufferPointer(start: result.data, count: count * width)
        return (0..<count).map { Array(flat[$0 * width ..< ($0 + 1) * width]) }
    }

    /// Cosine similarity of two vectors, computed by the library.
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "vectors of different sizes")
        return lembed_cosine_similarity(a, b, Int32(a.count))
    }
}
