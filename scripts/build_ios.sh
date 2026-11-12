#!/usr/bin/env bash
#
# libembedding - build_ios.sh
# Build libembedding for one iOS SDK, as a self-contained libembedding.dylib.
#
#   build_ios.sh <iphoneos|iphonesimulator> <work-dir> [tests]
#
# iphoneos is the device (arm64), iphonesimulator the simulator (arm64, the
# architecture of Apple Silicon Macs and of GitHub's macOS runners). With
# "tests", the unit tests are built too; they run on a simulator through
# `xcrun simctl spawn` (see .github/workflows/ios.yml).
#
# ONNX Runtime comes from Microsoft's onnxruntime-c pod archive. The newest
# one is 1.30.0 (1.31.0 is not published for iOS), so the iOS build uses the
# 1.30 headers (ORT_API_VERSION 30): compiling against them is what proves the
# code needs nothing newer. The archive holds a static library in an
# xcframework; it is linked into libembedding.dylib together with a static
# ggml/llama (BUILD_SHARED_LIBS=OFF), so the dylib depends on system
# frameworks only. iOS apps may not load arbitrary dylibs at run time, so the
# ggml CPU dispatch stays off (its default on Apple platforms) and Metal is
# off (GPU is out of scope before 1.x).
#
# Downloading: the iOS SDK has no libcurl, so build_curl_mbedtls.sh builds a
# static curl over mbedTLS, linked into the dylib. iOS exposes no certificate
# file and SecTrust is not available with mbedTLS: Mozilla's CA bundle, pinned
# by SHA-256 (the dated file of curl.se/ca), is compiled in
# (LIBEMBEDDING_CA_BUNDLE_EMBED). SSL_CERT_FILE / CURL_CA_BUNDLE override it.
# Models are cached in $HOME/Library/Caches/libembedding.
#
# Auteur: David Orel
# Version: 1.13.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <iphoneos|iphonesimulator> <work-dir> [tests]" >&2
    exit 2
fi
sdk="$1"
work="$2"
tests="${3:-}"

case "$sdk" in
    iphoneos)        slice=ios-arm64 ;;
    iphonesimulator) slice=ios-arm64_x86_64-simulator ;;
    *) echo "ERROR: unknown SDK '$sdk'" >&2; exit 2 ;;
esac

ORT_VERSION=1.30.0
ORT_SHA256=e6f1670c14406fd9f082bb400ab197a9b0a9646058ca6366e440642e2b54a2ea
CA_BUNDLE_DATE=2026-11-05   # Mozilla data as of 2026-11-05, published by curl.se
CA_BUNDLE_SHA256=427c064216761a1e72a8b30f0d954d08cb6e4435a82091fb865afd20c8497285
IOS_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET:-15.1}"  # the ORT pod's minimum
root="$(cd "$(dirname "$0")/.." && pwd)"

# ---- ONNX Runtime (onnxruntime-c pod) -----------------------------------------
ort="$work/ort"
if [ ! -f "$ort/onnxruntime.xcframework/Info.plist" ]; then
    mkdir -p "$ort"
    zip="$work/onnxruntime-c-$ORT_VERSION.zip"
    curl -fsSL --retry 3 -o "$zip" \
        "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-$ORT_VERSION.zip"
    echo "$ORT_SHA256  $zip" | shasum -a 256 -c -
    unzip -q -o "$zip" -d "$ort"
fi
ort_lib="$ort/onnxruntime.xcframework/$slice/onnxruntime.framework/onnxruntime"
if [ ! -f "$ort_lib" ]; then
    echo "ERROR: no ONNX Runtime slice $slice in the pod archive" >&2
    exit 1
fi

# ---- CA bundle and static libcurl over mbedTLS ----------------------------------
ca="$work/cacert-$CA_BUNDLE_DATE.pem"
if [ ! -f "$ca" ]; then
    curl -fsSL --retry 3 -o "$ca.part" "https://curl.se/ca/cacert-$CA_BUNDLE_DATE.pem"
    mv "$ca.part" "$ca"
fi
echo "$CA_BUNDLE_SHA256  $ca" | shasum -a 256 -c -

ios_toolchain=(
    -DCMAKE_SYSTEM_NAME=iOS
    -DCMAKE_OSX_SYSROOT="$sdk"
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET"
)
curl_work="$work/curl-$sdk"
bash "$root/scripts/build_curl_mbedtls.sh" "$curl_work" "${ios_toolchain[@]}"
curl_prefix="$curl_work/curl-prefix"

# ---- native build --------------------------------------------------------------
build="$work/cmake-$sdk"
targets=(embedding_shared)
test_flag=OFF
if [ "$tests" = tests ]; then
    test_flag=ON
    # test_embedding_parity is an integration test: it downloads bge-small.
    targets+=(test_similarity test_pooling test_tokenizer test_model_registry
              test_struct_sizes test_sparse_gguf test_embedding_parity)
fi

# The static ORT library needs the frameworks its CoreML provider uses:
# UIDevice (UIKit) and nw_path_* (Network) come from its prelinked objects.
# curl uses CoreFoundation and SystemConfiguration on Apple platforms.
apple_libs="-framework Foundation -framework CoreML -framework UIKit -framework Network"
apple_libs+=" -framework CoreFoundation -framework SystemConfiguration"
cmake -S "$root" -B "$build" \
    "${ios_toolchain[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_FIND_ROOT_PATH="$curl_prefix" \
    -DCMAKE_PREFIX_PATH="$curl_prefix" \
    -DLIBEMBEDDING_CA_BUNDLE_EMBED="$ca" \
    -DCMAKE_EXE_LINKER_FLAGS="$apple_libs" \
    -DCMAKE_SHARED_LINKER_FLAGS="$apple_libs" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=OFF \
    -DLIBEMBEDDING_CPU_DISPATCH=OFF \
    -DLIBEMBEDDING_BUILD_SHARED=ON \
    -DLIBEMBEDDING_BUILD_TESTS="$test_flag" \
    -DLIBEMBEDDING_INTEGRATION_TESTS="$test_flag" \
    -DLIBEMBEDDING_BUILD_EXAMPLES=OFF \
    -DLIBEMBEDDING_BUILD_BENCHMARKS=OFF \
    -DONNXRuntime_INCLUDE_DIR="$ort/Headers" \
    -DONNXRuntime_LIBRARY="$ort_lib"
cmake --build "$build" --parallel "$(sysctl -n hw.ncpu)" --target "${targets[@]}"

# An xcframework consumer finds the library through its own rpath.
dylib="$build/libembedding.dylib"
install_name_tool -id @rpath/libembedding.dylib "$dylib"
otool -L "$dylib"
# Nothing but system libraries and frameworks may remain.
if otool -L "$dylib" | tail -n +2 | grep -vE '^\s+(/usr/lib/|/System/Library/|@rpath/libembedding\.dylib)'; then
    echo "ERROR: libembedding.dylib depends on a non-system library" >&2
    exit 1
fi
