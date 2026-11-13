#!/usr/bin/env bash
#
# libembedding - build_wheel_manylinux.sh
# Build the shared library and the Python wheel inside a manylinux container,
# then repair the wheel with auditwheel. Called by .github/workflows/build-wheel.yml.
#
# Why a container: ggml's armv9.2 CPU variants need -march=armv9.2-a+sme, which
# only GCC 14 accepts. A GCC 14 installed on the Ubuntu runner links against its
# own libstdc++ and drags the wheel up to manylinux_2_39, which the runner's
# glibc 2.35 itself refuses. manylinux_2_28's gcc-toolset-14 keeps every
# GLIBCXX symbol within the glibc 2.28 baseline.
#
# Downloads go through a static libcurl over mbedTLS (build_curl_mbedtls.sh),
# linked into libembedding.so: the system libcurl would make auditwheel graft
# some thirty libraries (OpenSSL, krb5, LDAP, libssh, several under the LGPL)
# that an HTTPS model download does not need. The licence texts are gathered by
# collect_wheel_licenses.sh.
#
# Expected environment:
#   ONNXRUNTIME_VERSION  ONNX Runtime release to bundle (e.g. 1.31.0)
#   HOST_UID, HOST_GID   owner to give the produced files back to
#   PYTHON_TAG           interpreter under /opt/python (default cp312-cp312)
# Output: the repaired wheel in python/dist/, the build tree in build/.
#
# Auteur: David Orel
# Version: 1.13.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

: "${ONNXRUNTIME_VERSION:?ONNXRUNTIME_VERSION is required}"
: "${HOST_UID:?HOST_UID is required}"
: "${HOST_GID:?HOST_GID is required}"
PYTHON_TAG="${PYTHON_TAG:-cp312-cp312}"

cd /io
# The container runs as root; hand everything back so the runner's later steps
# (tag normalisation, artefact upload) can rename and read the files.
trap 'chown -R "$HOST_UID:$HOST_GID" /io' EXIT

case "$(uname -m)" in
    x86_64)  ort_arch=x64 ;;
    aarch64) ort_arch=aarch64 ;;
    *) echo "ERROR: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

echo "== toolchain"
# sed reads its whole input: `| head -1` would close the pipe early, and under
# pipefail the writer's SIGPIPE (exit 141) aborts the script.
gcc --version | sed -n 1p
cmake --version | sed -n 1p
ldd --version | sed -n 1p

echo "== system dependencies"
# bzip2 and xz unpack the curl and mbedTLS source archives.
dnf install -y bzip2 xz

echo "== ONNX Runtime ${ONNXRUNTIME_VERSION} (${ort_arch})"
curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ort_arch}-${ONNXRUNTIME_VERSION}.tgz" | tar xz
ort_root="$PWD/onnxruntime-linux-${ort_arch}-${ONNXRUNTIME_VERSION}"

echo "== static libcurl over mbedTLS"
curl_work="$PWD/build/curl"
bash scripts/build_curl_mbedtls.sh "$curl_work"
curl_prefix="$curl_work/curl-prefix"

echo "== shared library"
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$curl_prefix" \
    -DLIBEMBEDDING_BUILD_SHARED=ON \
    -DLIBEMBEDDING_BUILD_TESTS=OFF \
    -DLIBEMBEDDING_BUILD_EXAMPLES=OFF \
    -DLIBEMBEDDING_BUILD_BENCHMARKS=OFF \
    -DONNXRUNTIME_ROOT="$ort_root"
# Capped at 4 jobs: the bundled llama.cpp is memory-hungry to compile.
jobs=$(nproc)
if [ "$jobs" -gt 4 ]; then jobs=4; fi
cmake --build build --parallel "$jobs" --target embedding_shared
# libcurl must be linked in, not a dependency auditwheel would graft.
if objdump -p build/libembedding.so | grep -E 'NEEDED.*(curl|ssl|crypto)'; then
    echo "ERROR: libembedding.so links a system libcurl or TLS library" >&2
    exit 1
fi

echo "== licence texts"
bash scripts/collect_wheel_licenses.sh "$ONNXRUNTIME_VERSION" \
    "$curl_prefix/share/licenses/curl/COPYING" curl.COPYING \
    "$curl_prefix/share/licenses/mbedtls/LICENSE" mbedtls.LICENSE

echo "== package contents"
pkg=python/src/libembedding
cp build/libembedding.so "$pkg/"
cp "$ort_root/lib/libonnxruntime.so" "$pkg/"
cp "$ort_root/lib/libonnxruntime_providers_shared.so" "$pkg/" 2>/dev/null || true
# ggml CPU variants are dlopen()ed, never linked: auditwheel cannot find them
# through libembedding's dependencies, so they must already be in the package.
shopt -s nullglob
cpu_variants=(build/bin/libggml-cpu-*.so)
if [ ${#cpu_variants[@]} -eq 0 ]; then
    echo "ERROR: no libggml-cpu-*.so under build/bin" >&2
    exit 1
fi
for lib in "${cpu_variants[@]}"; do
    cp "$lib" "$pkg/"
    echo "Vendored ggml CPU variant: $(basename "$lib")"
done
ls -la "$pkg"

echo "== wheel"
python="/opt/python/${PYTHON_TAG}/bin/python"
raw_dir=$(mktemp -d)
(cd python && "$python" -m build --wheel --outdir "$raw_dir")
mkdir -p python/dist
# Highest glibc / libstdc++ symbol version each bundled library needs: the
# wheel's manylinux tag is set by the most demanding one, which may be a
# prebuilt dependency (ONNX Runtime) rather than our own code.
echo "== symbol versions required per library"
for lib in "$pkg"/*.so; do
    glibc=$(objdump -T "$lib" | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -n 1 || true)
    glibcxx=$(objdump -T "$lib" | grep -oE 'GLIBCXX_[0-9.]+' | sort -uV | tail -n 1 || true)
    echo "$(basename "$lib"): ${glibc:-none} ${glibcxx:-}"
done
show=$(auditwheel show "$raw_dir"/*.whl)
echo "$show"
# The image sets AUDITWHEEL_PLAT to its own policy (manylinux_2_28), which
# `repair` uses by default, so pass the tag auditwheel itself computed. Our own
# libraries stay within glibc 2.28; when the system libcurl was used, the
# libraries grafted for it (OpenSSL, krb5, libssh...) raised it to 2_31.
plat=$(echo "$show" | tr '\n' ' ' | grep -oE 'constrains the platform tag to "[^"]+"' \
       | grep -oE 'manylinux_[0-9]+_[0-9]+_[a-z0-9_]+' || true)
plat="${plat:-$AUDITWHEEL_PLAT}"
# Floor kept for Raspberry Pi OS bullseye (glibc 2.31): refuse anything newer.
glibc_minor=$(echo "$plat" | sed -E 's/manylinux_2_([0-9]+)_.*/\1/')
if [ "$glibc_minor" -gt 31 ]; then
    echo "ERROR: wheel needs $plat, above the manylinux_2_31 floor" >&2
    exit 1
fi
echo "== repairing to $plat"
auditwheel repair --plat "$plat" -w python/dist "$raw_dir"/*.whl
auditwheel show python/dist/*.whl
