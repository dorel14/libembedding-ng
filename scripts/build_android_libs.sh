#!/usr/bin/env bash
#
# libembedding - build_android_libs.sh
# Build libembedding and its native dependencies for one Android ABI and copy
# them into the Python package, ready for the wheel. Run by cibuildwheel as the
# Android before-build step (.github/workflows/android.yml), once per ABI.
#
# The target ABI is taken from _PYTHON_HOST_PLATFORM (android-24-arm64_v8a),
# else from the cross compiler cibuildwheel exports in CC. The NDK is
# ANDROID_NDK_LATEST_HOME (GitHub runners) or ANDROID_NDK_HOME. ONNX Runtime
# comes from its official Android AAR. The API level is ANDROID_API_LEVEL, the
# variable cibuildwheel derives the wheel tag from, so the libraries and the tag
# cannot disagree. Android has no libcurl: build_android_curl.sh builds a static
# one over mbedTLS, linked into libembedding.so, so models download as on any
# other platform. The licence texts of what the wheel bundles are gathered into
# python/licenses/ by collect_wheel_licenses.sh.
#
# Auteur: David Orel
# Version: 1.13.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.31.0}"
ANDROID_API="${ANDROID_API_LEVEL:-24}"
root="$(cd "$(dirname "$0")/.." && pwd)"

# ---- target ABI ------------------------------------------------------------
host="${_PYTHON_HOST_PLATFORM:-}"
case "$host" in
    *arm64_v8a*) abi=arm64-v8a ;;
    *x86_64*)    abi=x86_64 ;;
    *)
        case "$(basename "${CC:-}")" in
            aarch64-linux-android*) abi=arm64-v8a ;;
            x86_64-linux-android*)  abi=x86_64 ;;
            *) echo "ERROR: cannot tell the Android ABI (_PYTHON_HOST_PLATFORM='$host', CC='${CC:-}')" >&2
               exit 1 ;;
        esac ;;
esac

ndk="${ANDROID_NDK_LATEST_HOME:-${ANDROID_NDK_HOME:-}}"
if [ -z "$ndk" ] || [ ! -f "$ndk/build/cmake/android.toolchain.cmake" ]; then
    echo "ERROR: no Android NDK (ANDROID_NDK_LATEST_HOME / ANDROID_NDK_HOME)" >&2
    exit 1
fi
echo "== ABI $abi, API $ANDROID_API, NDK $ndk"

# ---- ONNX Runtime from the AAR ---------------------------------------------
work="$root/build-android-$abi"
ort="$work/ort"
if [ ! -f "$ort/lib/libonnxruntime.so" ]; then
    mkdir -p "$ort/include" "$ort/lib" "$work/aar"
    aar="$work/onnxruntime-android-${ONNXRUNTIME_VERSION}.aar"
    curl -fsSL -o "$aar" \
        "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ONNXRUNTIME_VERSION}/onnxruntime-android-${ONNXRUNTIME_VERSION}.aar"
    unzip -q -o "$aar" -d "$work/aar"
    cp "$work/aar/headers/"*.h "$ort/include/"
    cp "$work/aar/jni/$abi/libonnxruntime.so" "$ort/lib/"
fi

# ---- static libcurl over mbedTLS ---------------------------------------------
env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS -u CPPFLAGS -u AR -u RANLIB \
    bash "$root/scripts/build_android_curl.sh" "$abi" "$ANDROID_API" "$ndk" "$work"
curl_prefix="$work/curl-prefix"

# ---- native build ------------------------------------------------------------
# The cross-compilation variables cibuildwheel exports for the Python build
# would override the NDK toolchain file: CMake must see a clean environment.
env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS -u CPPFLAGS -u AR -u RANLIB \
    cmake -S "$root" -B "$work/cmake" \
        -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI="$abi" \
        -DANDROID_PLATFORM="android-$ANDROID_API" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_FIND_ROOT_PATH="$curl_prefix" \
        -DCMAKE_PREFIX_PATH="$curl_prefix" \
        -DLIBEMBEDDING_BUILD_SHARED=ON \
        -DLIBEMBEDDING_BUILD_TESTS=OFF \
        -DLIBEMBEDDING_BUILD_EXAMPLES=OFF \
        -DLIBEMBEDDING_BUILD_BENCHMARKS=OFF \
        -DONNXRuntime_INCLUDE_DIR="$ort/include" \
        -DONNXRuntime_LIBRARY="$ort/lib/libonnxruntime.so"
env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u LDFLAGS -u CPPFLAGS -u AR -u RANLIB \
    cmake --build "$work/cmake" --parallel 4 --target embedding_shared

# ---- package contents ----------------------------------------------------------
# cibuildwheel builds every ABI from the same tree: drop the previous ABI's
# libraries before copying this one's.
pkg="$root/python/src/libembedding"
rm -f "$pkg"/*.so
cp "$work/cmake/libembedding.so" "$pkg/"
cp "$ort/lib/libonnxruntime.so" "$pkg/"
shopt -s nullglob
libs=("$work/cmake/bin/"*.so)
if [ ${#libs[@]} -eq 0 ]; then
    echo "ERROR: no ggml/llama shared libraries under $work/cmake/bin" >&2
    exit 1
fi
cp "${libs[@]}" "$pkg/"
ls -la "$pkg"/*.so

# ---- licence texts (python/licenses/, PEP 639 license-files) -------------------
bash "$root/scripts/collect_wheel_licenses.sh" "$ONNXRUNTIME_VERSION" \
    "$curl_prefix/share/licenses/curl/COPYING" curl.COPYING \
    "$curl_prefix/share/licenses/mbedtls/LICENSE" mbedtls.LICENSE
