#!/usr/bin/env bash
#
# libembedding - build_android_curl.sh
# Build a static libcurl for one Android ABI, so the Android library can
# download models like every other platform.
#
#   build_android_curl.sh <abi> <api-level> <ndk-dir> <work-dir>
#
# Android ships no libcurl and no OpenSSL for NDK code, so both come from
# source: curl over mbedTLS (the LTS branch), static and position independent,
# then linked into libembedding.so. HTTP only: models come from HuggingFace
# over HTTPS, nothing else is needed. Both archives are pinned by SHA-256;
# curl's matches the release signed by Daniel Stenberg (GPG key
# 27EDEAF22F3ABCEB50DB9A125CC908FDB71E12C2) and mbedTLS's the sum published
# with its release. The curl version is the one of the bundled Windows headers
# (third_party/curl/include/curl/curlver.h).
#
# At run time the certificates come from the system store,
# /system/etc/security/cacerts (CURLOPT_CAPATH, set by downloader_impl.hpp).
#
# The result is installed in <work-dir>/curl-prefix, which holds a CMake
# package for find_package(CURL); a second run reuses it.
#
# Licences: curl is under the curl licence (MIT-style), mbedTLS under
# Apache-2.0 (dual-licensed with GPL-2.0-or-later; used under Apache-2.0).
#
# Auteur: David Orel
# Version: 1.12.2
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [ $# -ne 4 ]; then
    echo "usage: $0 <abi> <api-level> <ndk-dir> <work-dir>" >&2
    exit 2
fi
abi="$1"
api="$2"
ndk="$3"
work="$4"

CURL_VERSION=8.21.0
CURL_SHA256=aa1b66a70eace83dc624508745646c08ae561de512ab403adffb93ac87fc72e6
MBEDTLS_VERSION=3.6.7
MBEDTLS_SHA256=a7e8bcbec0e6f761b4af24f25677626b35f762f68eef79c08677a363212d11f6

prefix="$work/curl-prefix"
if [ -f "$prefix/lib/libcurl.a" ] && [ -f "$prefix/lib/libmbedtls.a" ]; then
    echo "== libcurl for $abi already built in $prefix"
    exit 0
fi

src="$work/curl-src"
mkdir -p "$src"

fetch() {  # url file sha256
    if [ ! -f "$src/$2" ]; then
        curl -fsSL --retry 3 -o "$src/$2.part" "$1"
        mv "$src/$2.part" "$src/$2"
    fi
    echo "$3  $src/$2" | sha256sum -c -
}

fetch "https://github.com/Mbed-TLS/mbedtls/releases/download/mbedtls-$MBEDTLS_VERSION/mbedtls-$MBEDTLS_VERSION.tar.bz2" \
      "mbedtls-$MBEDTLS_VERSION.tar.bz2" "$MBEDTLS_SHA256"
fetch "https://curl.se/download/curl-$CURL_VERSION.tar.xz" \
      "curl-$CURL_VERSION.tar.xz" "$CURL_SHA256"

rm -rf "$src/mbedtls-$MBEDTLS_VERSION" "$src/curl-$CURL_VERSION"
tar -xjf "$src/mbedtls-$MBEDTLS_VERSION.tar.bz2" -C "$src"
tar -xJf "$src/curl-$CURL_VERSION.tar.xz" -C "$src"

common=(
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake"
    -DANDROID_ABI="$abi"
    -DANDROID_PLATFORM="android-$api"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_INSTALL_PREFIX="$prefix"
    # The NDK toolchain restricts find_*() to its sysroot: let it see the
    # prefix too, where curl looks for mbedTLS.
    -DCMAKE_FIND_ROOT_PATH="$prefix"
    -DCMAKE_PREFIX_PATH="$prefix"
)

# ---- mbedTLS ---------------------------------------------------------------
cmake -S "$src/mbedtls-$MBEDTLS_VERSION" -B "$work/mbedtls-build" "${common[@]}" \
    -DENABLE_TESTING=OFF \
    -DENABLE_PROGRAMS=OFF \
    -DGEN_FILES=OFF \
    -DMBEDTLS_FATAL_WARNINGS=OFF \
    -DUSE_SHARED_MBEDTLS_LIBRARY=OFF \
    -DUSE_STATIC_MBEDTLS_LIBRARY=ON
cmake --build "$work/mbedtls-build" --parallel 4
cmake --install "$work/mbedtls-build"

# ---- curl --------------------------------------------------------------------
cmake -S "$src/curl-$CURL_VERSION" -B "$work/curl-build" "${common[@]}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_CURL_EXE=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_LIBCURL_DOCS=OFF \
    -DBUILD_MISC_DOCS=OFF \
    -DENABLE_CURL_MANUAL=OFF \
    -DHTTP_ONLY=ON \
    -DCURL_USE_MBEDTLS=ON \
    -DCURL_USE_OPENSSL=OFF \
    -DCURL_USE_LIBPSL=OFF \
    -DCURL_USE_LIBSSH2=OFF \
    -DUSE_LIBIDN2=OFF \
    -DUSE_NGHTTP2=OFF \
    -DCURL_ZLIB=OFF \
    -DCURL_BROTLI=OFF \
    -DCURL_ZSTD=OFF \
    -DCURL_DISABLE_LDAP=ON \
    -DCURL_CA_BUNDLE=none \
    -DCURL_CA_PATH=none \
    -DMBEDTLS_INCLUDE_DIR="$prefix/include" \
    -DMBEDTLS_LIBRARY="$prefix/lib/libmbedtls.a" \
    -DMBEDX509_LIBRARY="$prefix/lib/libmbedx509.a" \
    -DMBEDCRYPTO_LIBRARY="$prefix/lib/libmbedcrypto.a"
cmake --build "$work/curl-build" --parallel 4
cmake --install "$work/curl-build"

ls -la "$prefix/lib"
