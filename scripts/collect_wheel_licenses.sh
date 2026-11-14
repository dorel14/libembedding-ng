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

for file in LICENSE ThirdPartyNotices.txt; do
    curl -fsSL --retry 3 -o "$dest/onnxruntime.$file" \
        "https://raw.githubusercontent.com/microsoft/onnxruntime/v$ort_version/$file"
done

while [ $# -gt 0 ]; do
    if [ ! -s "$1" ]; then
        echo "ERROR: licence file '$1' is missing or empty" >&2
        exit 1
    fi
    cp "$1" "$dest/$2"
    shift 2
done

ls -la "$dest"
