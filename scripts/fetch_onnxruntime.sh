#!/usr/bin/env bash
#
# libembedding - fetch_onnxruntime.sh
# Download a prebuilt ONNX Runtime and lay it out as
# <dest>/onnxruntime-<platform>-<version>/{include,lib}, the layout of the
# official release archives that ONNXRUNTIME_ROOT points at.
#
#   fetch_onnxruntime.sh <platform> <version> [<dest>]
#
#   <platform>  linux-x64 | linux-aarch64 | osx-arm64 | osx-x86_64 | win-x64
#   <dest>      parent directory, default: current directory
#
# Primary source: the GitHub release asset of microsoft/onnxruntime.
# Fallback: the NuGet package Microsoft.ML.OnnxRuntime, used when GitHub does
# not serve the asset (the repository answered 404 on 2026-11-18 and every CI
# job stopped at this download). The NuGet package carries the same headers and
# shared libraries for linux-x64, linux-arm64, osx-arm64 and win-x64, but only
# the unversioned library name, so the fallback recreates the versioned names
# and SONAME links the release archives ship. osx-x86_64 has no NuGet build.
#
# The root path is printed on the last line of stdout.
#
# Auteur: David Orel
# Version: 1.14.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "usage: $0 <platform> <version> [<dest>]" >&2
    exit 2
fi
plat="$1"
ver="$2"
dest="${3:-.}"
root="$dest/onnxruntime-$plat-$ver"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$dest"

# Zip archives (the Windows release asset and every .nupkg): GNU tar cannot
# read them and unzip is missing from some images, Python is on every runner.
# Each candidate must run: on Windows "python3" can be the Microsoft Store stub.
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

fetch_github() {
    local ext=tgz
    [ "$plat" = win-x64 ] && ext=zip
    local url="https://github.com/microsoft/onnxruntime/releases/download/v$ver/onnxruntime-$plat-$ver.$ext"
    echo "ONNX Runtime $ver ($plat): GitHub $url" >&2
    curl -fsSL --retry 3 -o "$tmp/ort.$ext" "$url" || return 1
    if [ "$ext" = tgz ]; then
        tar xzf "$tmp/ort.$ext" -C "$dest"
    else
        unzip_to "$tmp/ort.$ext" "$dest"
    fi
    [ -d "$root/include" ] && [ -d "$root/lib" ]
}

fetch_nuget() {
    local rid
    case "$plat" in
        linux-x64)     rid=linux-x64 ;;
        linux-aarch64) rid=linux-arm64 ;;
        osx-arm64)     rid=osx-arm64 ;;
        win-x64)       rid=win-x64 ;;
        *) echo "ERROR: NuGet has no ONNX Runtime build for $plat" >&2; return 1 ;;
    esac
    local url="https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ver/microsoft.ml.onnxruntime.$ver.nupkg"
    echo "ONNX Runtime $ver ($plat): NuGet fallback $url" >&2
    curl -fsSL --retry 3 -o "$tmp/ort.nupkg" "$url" || return 1
    mkdir -p "$tmp/nupkg"
    unzip_to "$tmp/ort.nupkg" "$tmp/nupkg"

    rm -rf "$root"
    mkdir -p "$root/lib"
    cp -R "$tmp/nupkg/build/native/include" "$root/include"
    cp "$tmp/nupkg/runtimes/$rid/native/"* "$root/lib/"
    cp "$tmp/nupkg/LICENSE" "$tmp/nupkg/ThirdPartyNotices.txt" "$root/"

    # Same file names as the release archives: the library carries its
    # versioned name, the SONAME / install name (.so.1, .1.dylib) and the
    # unversioned link name point at it.
    case "$plat" in
        linux-*)
            (cd "$root/lib" &&
                mv libonnxruntime.so "libonnxruntime.so.$ver" &&
                ln -s "libonnxruntime.so.$ver" libonnxruntime.so.1 &&
                ln -s libonnxruntime.so.1 libonnxruntime.so) ;;
        osx-*)
            (cd "$root/lib" &&
                mv libonnxruntime.dylib "libonnxruntime.$ver.dylib" &&
                ln -s "libonnxruntime.$ver.dylib" libonnxruntime.1.dylib &&
                ln -s "libonnxruntime.$ver.dylib" libonnxruntime.dylib) ;;
    esac
}

if ! fetch_github; then
    echo "WARNING: GitHub release asset unavailable, trying NuGet" >&2
    rm -rf "$root"
    fetch_nuget
fi

ls -la "$root/lib" >&2
echo "$root"
