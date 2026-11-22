#!/usr/bin/env bash
#
# libembedding - fetch_onnxruntime_armv7l_fallback.sh
# Rebuild the ort-out/ tree of the armv7l workflow without compiling ONNX
# Runtime, for when the upstream sources cannot be cloned (microsoft/onnxruntime
# answered "Repository not found" on 2026-11-22).
#
#   fetch_onnxruntime_armv7l_fallback.sh <version> <outdir>
#
# Sources:
#   - library: the armv7l wheel of libembedding-ng published on PyPI, which
#     bundles the ORT cross-compiled by build-wheel-armv7l.yml (--no_telemetry)
#     as libembedding_ng.libs/libonnxruntime-<hash>.so.1 (auditwheel renamed
#     it and patched its SONAME). The wheel URL and SHA-256 are pinned below.
#   - headers, LICENSE, ThirdPartyNotices.txt: the NuGet package
#     Microsoft.ML.OnnxRuntime of the same version.
#
# Output (same layout as the cross-compile step): <outdir>/lib/
# libonnxruntime.so.<version>, libonnxruntime.so.1 and libonnxruntime.so
# (symlinks), <outdir>/include/*.h, <outdir>/LICENSE, <outdir>/ThirdPartyNotices.txt.
# The SONAME is reset to libonnxruntime.so.1 with patchelf, which must be
# installed.
#
# Auteur: David Orel
# Version: 1.14.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <version> <outdir>" >&2
    exit 2
fi
ver="$1"
out="$2"

# Pinned wheels, one per ORT version. Add a case when ORT is bumped and a
# wheel bundling the new version has been published.
case "$ver" in
    1.31.0)
        wheel_url="https://files.pythonhosted.org/packages/f0/84/3dc3f5c0083d3636a16843c8e081689495e9e10d9fb82db477fe9a75fa8a/libembedding_ng-1.16.0-py3-none-manylinux_2_35_armv7l.whl"
        wheel_sha256="523a0c44233dab9d0456043a00b2a2415a509b8b1b62e57b5c497f6ff2e76a6c"
        ;;
    *)
        echo "ERROR: no pinned libembedding-ng armv7l wheel bundles ONNX Runtime $ver" >&2
        exit 1
        ;;
esac

command -v patchelf > /dev/null || { echo "ERROR: patchelf is required" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Zip extraction: python is on every runner, unzip is not guaranteed.
unzip_to() {
    local py
    for py in python3 python; do
        if "$py" -c 'import zipfile' > /dev/null 2>&1; then
            "$py" -m zipfile -e "$1" "$2"
            return
        fi
    done
    unzip -q -o "$1" -d "$2"
}

echo "Wheel: $wheel_url" >&2
curl -fsSL --retry 3 -o "$tmp/ort.whl" "$wheel_url"
echo "$wheel_sha256  $tmp/ort.whl" | sha256sum -c - >&2
mkdir -p "$tmp/whl"
unzip_to "$tmp/ort.whl" "$tmp/whl"

shopt -s nullglob
libs=("$tmp"/whl/libembedding_ng.libs/libonnxruntime-*.so.1)
if [ ${#libs[@]} -ne 1 ]; then
    echo "ERROR: expected exactly one libonnxruntime-*.so.1 in the wheel, found ${#libs[@]}" >&2
    exit 1
fi

nuget_url="https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ver/microsoft.ml.onnxruntime.$ver.nupkg"
echo "Headers: $nuget_url" >&2
curl -fsSL --retry 3 -o "$tmp/ort.nupkg" "$nuget_url"
mkdir -p "$tmp/nupkg"
unzip_to "$tmp/ort.nupkg" "$tmp/nupkg"

rm -rf "$out"
mkdir -p "$out/lib" "$out/include"
cp "$tmp"/nupkg/build/native/include/*.h "$out/include/"
cp "$tmp/nupkg/LICENSE" "$tmp/nupkg/ThirdPartyNotices.txt" "$out/"

cp "${libs[0]}" "$out/lib/libonnxruntime.so.$ver"
patchelf --set-soname libonnxruntime.so.1 "$out/lib/libonnxruntime.so.$ver"
(cd "$out/lib" &&
    ln -s "libonnxruntime.so.$ver" libonnxruntime.so.1 &&
    ln -s libonnxruntime.so.1 libonnxruntime.so)

echo "SONAME: $(patchelf --print-soname "$out/lib/libonnxruntime.so.$ver")" >&2
ls -la "$out/lib" >&2
