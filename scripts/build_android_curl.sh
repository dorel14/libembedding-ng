#!/usr/bin/env bash
#
# libembedding - build_android_curl.sh
# Build a static libcurl for one Android ABI, so the Android library can
# download models like every other platform.
#
#   build_android_curl.sh <abi> <api-level> <ndk-dir> <work-dir>
#
# Android ships no libcurl and no OpenSSL for NDK code: build_curl_mbedtls.sh
# builds curl over mbedTLS from source (pinned versions and hashes live
# there), here with the NDK toolchain. At run time the certificates come from
# the system store, /system/etc/security/cacerts (CURLOPT_CAPATH, set by
# downloader_impl.hpp).
#
# The result is installed in <work-dir>/curl-prefix.
#
# Auteur: David Orel
# Version: 1.13.0
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

exec bash "$(dirname "$0")/build_curl_mbedtls.sh" "$work" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$abi" \
    -DANDROID_PLATFORM="android-$api"
