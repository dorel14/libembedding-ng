#!/usr/bin/env bash
#
# libembedding - check_wheel_armv7l.sh
# Repair and test the armv7l wheel built by build_wheel_armv7l.sh. Runs inside
# an arm32v7 Debian 12 container (QEMU on an x86_64 runner, see
# .github/workflows/build-wheel-armv7l.yml), with the repository mounted at the
# same path as on the host so that the build tree's RUNPATH stays valid for
# auditwheel.
#
#   check_wheel_armv7l.sh <onnxruntime-dir>
#
# Steps: auditwheel repair (grafts libllama, libggml*, libonnxruntime and
# libgomp), then install the repaired wheel in a fresh venv and run the Python
# tests without the build tree or the ORT directory, as a user gets it. numpy
# and cffi publish no armv7l wheels on PyPI; Raspberry Pi OS takes them from
# piwheels.org, and so does this test.
#
# Output: the repaired wheel in python/dist/ (the raw one is removed).
#
# Auteur: David Orel
# Version: 1.13.0
#
# SPDX-License-Identifier: MIT

set -euo pipefail

ort_root="$(cd "$1" && pwd)"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

echo "== system"
uname -m
ldd --version | sed -n 1p
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    python3 python3-venv ca-certificates patchelf libgomp1 >/dev/null
python3 --version

python3 -m venv /tmp/venv
# shellcheck disable=SC1091
. /tmp/venv/bin/activate
pip install -q --upgrade pip
pip install -q auditwheel

echo "== repair"
raw=$(ls python/dist/*.whl)
auditwheel show "$raw"
mkdir -p /tmp/repaired
auditwheel repair -w /tmp/repaired "$raw"
rm -f "$raw"
cp /tmp/repaired/*.whl python/dist/
auditwheel show python/dist/*.whl
wheel=$(ls python/dist/*.whl)
case "$wheel" in
    *manylinux_2_3[0-5]_armv7l*) ;;
    *) echo "ERROR: unexpected platform tag: $wheel" >&2; exit 1 ;;
esac

echo "== install and test"
# The repaired wheel must stand alone: hide the build tree and ORT.
mv build build.moved
mv "$ort_root" "$ort_root.moved"
restore() {
    mv build.moved build 2>/dev/null || true
    mv "$ort_root.moved" "$ort_root" 2>/dev/null || true
}
trap restore EXIT

pip install -q --extra-index-url https://www.piwheels.org/simple "$wheel" pytest
export LIBEMBEDDING_TEST_INSTALLED=1
(cd /tmp && python -c "import libembedding, sys; print('Testing', libembedding.__file__, libembedding.__version__); sys.exit('site-packages' not in libembedding.__file__)")
cd python
# Offline suite, then the parity check against the x86_64 reference
# (downloads bge-small): a wrong ARM32 kernel shows up there.
python -m pytest tests -q -m "not network" -p no:cacheprovider
python -m pytest tests/test_parity.py -v -p no:cacheprovider
