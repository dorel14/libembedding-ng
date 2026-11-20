#!/usr/bin/env bash
#
# libembedding - collect_wheel_licenses.sh
# Add the build-dependent licence texts to python/licenses/ before a wheel is
# built; pyproject.toml lists that directory in license-files (PEP 639), so the
# texts land in the wheel's .dist-info/licenses/.
#
#   collect_wheel_licenses.sh <onnxruntime-version> [<source> <name>]...
#
# ONNX Runtime's LICENSE and ThirdPartyNotices.txt are fetched at the release
# tag of the bundled version: the binary archives do not all carry them (the
# Android AAR does not). Each <source> <name> pair copies a file that the build
# produced, e.g. curl's and mbedTLS's licences from build_curl_mbedtls.sh's
# prefix, or vcpkg's copyright files on Windows.
#
# The stable texts (llama.cpp, cJSON, stb, THIRD_PARTY_NOTICES.md) are
# versioned in python/licenses/; the ones written here are git-ignored.
#
# Auteur: David Orel
# Version: 1.13.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [ $# -lt 1 ] || [ $(( ($# - 1) % 2 )) -ne 0 ]; then
    echo "usage: $0 <onnxruntime-version> [<source> <name>]..." >&2
    exit 2
fi
ort_version="$1"
shift

dest="$(cd "$(dirname "$0")/.." && pwd)/python/licenses"
mkdir -p "$dest"

# Fallbacks when GitHub does not serve the files (microsoft/onnxruntime answered
# 404 on 2026-11-18): the copy fetch_onnxruntime.sh left in $ONNXRUNTIME_ROOT,
# then the NuGet package of the same version, which carries both texts.
nupkg_dir=""
for file in LICENSE ThirdPartyNotices.txt; do
    out="$dest/onnxruntime.$file"
    if curl -fsSL --retry 3 -o "$out" \
        "https://raw.githubusercontent.com/microsoft/onnxruntime/v$ort_version/$file"; then
        continue
    fi
    if [ -n "${ONNXRUNTIME_ROOT:-}" ] && [ -s "$ONNXRUNTIME_ROOT/$file" ]; then
        echo "WARNING: $file from GitHub unavailable, using $ONNXRUNTIME_ROOT/$file" >&2
        cp "$ONNXRUNTIME_ROOT/$file" "$out"
        continue
    fi
    echo "WARNING: $file from GitHub unavailable, using the NuGet package" >&2
    if [ -z "$nupkg_dir" ]; then
        nupkg_dir="$(mktemp -d)"
        curl -fsSL --retry 3 -o "$nupkg_dir/ort.nupkg" \
            "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ort_version/microsoft.ml.onnxruntime.$ort_version.nupkg"
    fi
    # Each candidate must run: on Windows "python3" can be the Store stub.
    py=python3
    "$py" -c 'import zipfile' > /dev/null 2>&1 || py=python
    "$py" -c 'import sys, zipfile; sys.stdout.buffer.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))' \
        "$nupkg_dir/ort.nupkg" "$file" > "$out"
done
if [ -n "$nupkg_dir" ]; then rm -rf "$nupkg_dir"; fi

while [ $# -gt 0 ]; do
    if [ ! -s "$1" ]; then
        echo "ERROR: licence file '$1' is missing or empty" >&2
        exit 1
    fi
    cp "$1" "$dest/$2"
    shift 2
done

ls -la "$dest"
