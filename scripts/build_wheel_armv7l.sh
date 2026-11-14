#!/usr/bin/env bash
#
# libembedding - build_wheel_armv7l.sh
# Cross-compile libembedding for 32-bit ARM Linux (armv7l, Raspberry Pi OS
# 32-bit) on an x86_64 Ubuntu 22.04 host and build the raw Python wheel.
# Called by .github/workflows/build-wheel-armv7l.yml, which then repairs and
# tests the wheel in an arm32v7 Debian 12 container under QEMU.
#
#   build_wheel_armv7l.sh <onnxruntime-dir>
#
# <onnxruntime-dir> holds ONNX Runtime cross-compiled for armv7l (ORT publishes
# no armv7l build): lib/libonnxruntime.so*, include/*.h, as produced by the
# "ort" job of the workflow.
#
# Toolchain: cmake/toolchains/armv7l-linux-gnueabihf.cmake (ARMv7-A, NEON
# VFPv4, hard float). ggml has no CPU variant set for 32-bit ARM, so a single
# ggml-cpu is built for those flags (LIBEMBEDDING_CPU_DISPATCH is OFF by
# default there). Downloads go through a static libcurl over mbedTLS, as in
# the other Linux wheels.
#
# Expected environment:
#   ONNXRUNTIME_VERSION  version of <onnxruntime-dir>, for the licence texts
# Output: the raw wheel (tag linux_armv7l) in python/dist/, the build tree in
# build/ (its RUNPATH points auditwheel at libllama, libggml* and ORT).
#
# Auteur: David Orel
# Version: 1.13.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <onnxruntime-dir>" >&2
    exit 2
fi
: "${ONNXRUNTIME_VERSION:?ONNXRUNTIME_VERSION is required}"
ort_root="$(cd "$1" && pwd)"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
toolchain="$root/cmake/toolchains/armv7l-linux-gnueabihf.cmake"

ort_lib="$ort_root/lib/libonnxruntime.so"
if [ ! -e "$ort_lib" ] || [ ! -f "$ort_root/include/onnxruntime_c_api.h" ]; then
    echo "ERROR: no ONNX Runtime library and headers under $ort_root" >&2
    exit 1
fi

echo "== toolchain"
arm-linux-gnueabihf-g++ --version | sed -n 1p
cmake --version | sed -n 1p

echo "== static libcurl over mbedTLS (armv7l)"
curl_work="$root/build/curl"
bash scripts/build_curl_mbedtls.sh "$curl_work" -DCMAKE_TOOLCHAIN_FILE="$toolchain"
curl_prefix="$curl_work/curl-prefix"

echo "== shared library"
# The toolchain restricts find_*() to its sysroot: the curl prefix is added
# as a root. ONNX Runtime is given by path, which find_library would re-root.
cmake -S . -B build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_FIND_ROOT_PATH="$curl_prefix" \
    -DCMAKE_PREFIX_PATH="$curl_prefix" \
    -DLIBEMBEDDING_BUILD_SHARED=ON \
    -DLIBEMBEDDING_BUILD_TESTS=OFF \
    -DLIBEMBEDDING_BUILD_EXAMPLES=OFF \
    -DLIBEMBEDDING_BUILD_BENCHMARKS=OFF \
    -DONNXRUNTIME_ROOT="$ort_root" \
    -DONNXRuntime_INCLUDE_DIR="$ort_root/include" \
    -DONNXRuntime_LIBRARY="$ort_lib"
cmake --build build --parallel "$(nproc)" --target embedding_shared

lib=build/libembedding.so
file "$lib"
arm-linux-gnueabihf-readelf -A "$lib" | grep -E "Tag_CPU_arch:|Tag_ABI_VFP_args"
arm-linux-gnueabihf-objdump -p "$lib" | grep -E "NEEDED|RUNPATH|RPATH"
# libcurl must be linked in, not a dependency auditwheel would graft.
if arm-linux-gnueabihf-objdump -p "$lib" | grep -E 'NEEDED.*(curl|ssl|crypto)'; then
    echo "ERROR: libembedding.so links a system libcurl or TLS library" >&2
    exit 1
fi

echo "== licence texts"
bash scripts/collect_wheel_licenses.sh "$ONNXRUNTIME_VERSION" \
    "$curl_prefix/share/licenses/curl/COPYING" curl.COPYING \
    "$curl_prefix/share/licenses/mbedtls/LICENSE" mbedtls.LICENSE

echo "== package contents"
pkg=python/src/libembedding
cp "$lib" "$pkg/"
# ORT is linked through its SONAME (libonnxruntime.so.1): auditwheel grafts
# it from the build RUNPATH. The unversioned copy is the one the Python
# loader looks for, as in the other Linux wheels.
cp -L "$ort_lib" "$pkg/libonnxruntime.so"
ls -la "$pkg"

echo "== symbol versions required per library"
for so in "$pkg"/*.so build/bin/*.so; do
    [ -e "$so" ] || continue
    glibc=$(arm-linux-gnueabihf-objdump -T "$so" | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -n 1 || true)
    glibcxx=$(arm-linux-gnueabihf-objdump -T "$so" | grep -oE 'GLIBCXX_[0-9.]+' | sort -uV | tail -n 1 || true)
    echo "$(basename "$so"): ${glibc:-none} ${glibcxx:-}"
done

echo "== raw wheel"
rm -rf python/dist
(cd python && python3 -m build --wheel --outdir dist \
    -C--build-option=--plat-name=linux_armv7l)
ls -la python/dist
