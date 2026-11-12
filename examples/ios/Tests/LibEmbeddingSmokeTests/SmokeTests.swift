//
// libembedding - SmokeTests.swift
// XCTest smoke suite for libembedding on iOS, run on a simulator by
// .github/workflows/ios.yml (`xcodebuild test`).
//
// Auteur: David Orel
// Version: 1.13.0
//
// SPDX-License-Identifier: MIT

import Foundation
import XCTest
import libembedding
@testable import LibEmbeddingSwift

final class SmokeTests: XCTestCase {
    /// bge-small (133 MB, under the mobile limit) downloads, embeds and ranks
    /// a paraphrase above an unrelated sentence.
    func testEmbedAndRank() throws {
        let model = try TextEmbedder()
        XCTAssertEqual(model.dim, 384)

        let vectors = try model.embed([
            "The cat sits on the mat.",
            "A cat is sitting on a mat.",
            "Quarterly revenue grew by eight percent.",
        ])
        XCTAssertEqual(vectors.count, 3)
        for v in vectors {
            XCTAssertEqual(v.count, 384)
            let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
            XCTAssertEqual(norm, 1, accuracy: 1e-3)
        }
        let paraphrase = TextEmbedder.cosine(vectors[0], vectors[1])
        let unrelated = TextEmbedder.cosine(vectors[0], vectors[2])
        XCTAssertGreaterThan(paraphrase, unrelated)
    }

    /// Embeddings match the x86_64 reference (python/tests/data/
    /// parity_bge_small_en_v15.json). CI passes its path through
    /// TEST_RUNNER_LEMBED_PARITY_REF; without it the test is skipped.
    func testParityWithX86Reference() throws {
        guard let path = ProcessInfo.processInfo.environment["LEMBED_PARITY_REF"] else {
            throw XCTSkip("LEMBED_PARITY_REF is not set")
        }
        struct Reference: Decodable {
            let minCosine: Float
            let dim: Int
            let texts: [String]
            let embeddings: [[Float]]
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let reference = try decoder.decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: path)))

        let model = try TextEmbedder()
        XCTAssertEqual(model.dim, reference.dim)
        let vectors = try model.embed(reference.texts)
        for (i, (got, want)) in zip(vectors, reference.embeddings).enumerated() {
            let cosine = TextEmbedder.cosine(got, want)
            print("text \(i): cosine vs x86_64 reference = \(cosine)")
            XCTAssertGreaterThanOrEqual(cosine, reference.minCosine)
        }
    }

    /// On iOS a registry model over 150 MB is refused before any download.
    func testMobileLimitRefusesLargeModel() {
        XCTAssertEqual(lembed_mobile_model_limit_mb(), 150)
        XCTAssertThrowsError(try TextEmbedder(model: LEMBED_TEXT_BGE_LARGE_EN_V15)) { error in
            guard let error = error as? LibEmbeddingError else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(error.status, LEMBED_ERROR_UNSUPPORTED)
            XCTAssertTrue(error.message.contains("150 MB"), error.message)
        }
    }
}
