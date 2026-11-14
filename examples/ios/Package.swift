// swift-tools-version:5.9
//
// libembedding - examples/ios/Package.swift
// Swift package that uses libembedding.xcframework on iOS: a small Swift
// wrapper (LibEmbeddingSwift) and an XCTest smoke suite run on a simulator.
//
// libembedding.xcframework is not versioned: build it with scripts/build_ios.sh
// for both SDKs and `xcodebuild -create-xcframework` (see
// .github/workflows/ios.yml), then copy it next to this file.
//
// Auteur: David Orel
// Version: 1.13.0
//
// SPDX-License-Identifier: MIT

import PackageDescription

let package = Package(
    name: "LibEmbeddingSmoke",
    platforms: [.iOS("15.1")],  // the minimum of the ONNX Runtime pod
    products: [
        .library(name: "LibEmbeddingSwift", targets: ["LibEmbeddingSwift"]),
    ],
    targets: [
        // The framework's module is named after its binary: `import libembedding`.
        .binaryTarget(name: "libembedding", path: "libembedding.xcframework"),
        .target(name: "LibEmbeddingSwift", dependencies: ["libembedding"]),
        .testTarget(name: "LibEmbeddingSmokeTests", dependencies: ["LibEmbeddingSwift"]),
    ]
)
