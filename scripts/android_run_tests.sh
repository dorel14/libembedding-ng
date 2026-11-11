#!/usr/bin/env bash
#
# libembedding - android_run_tests.sh
# Push the Android unit-test binaries and their shared libraries to a running
# emulator or device with adb, run each test there and report PASS/FAIL.
# Called by .github/workflows/android.yml.
#
# Usage: android_run_tests.sh <staging-dir>
#   <staging-dir> holds the test executables and every .so they need
#   (libonnxruntime.so, libggml*.so, ...).
#
# Android has no /tmp and no HOME: TMPDIR points at a writable directory under
# /data/local/tmp so tests that write fixtures (test_sparse_gguf) and the model
# cache can live there.
#
# Auteur: David Orel
# Version: 1.12.1
#
# SPDX-License-Identifier: MIT

set -euo pipefail

stage="${1:?usage: android_run_tests.sh <staging-dir>}"
dest=/data/local/tmp/libembedding
# test_text_embedding downloads its model: it checks libcurl, mbedTLS and the
# system certificate store (/system/etc/security/cacerts) on the device.
# test_embedding_parity compares with the x86_64 reference pushed alongside
# (parity_bge_small_en_v15.json, cosine >= 0.999 per text).
tests=(test_similarity test_pooling test_tokenizer test_model_registry test_struct_sizes test_sparse_gguf test_text_embedding test_embedding_parity)

adb wait-for-device
adb shell "getprop ro.product.cpu.abi; getprop ro.build.version.sdk"
adb shell "rm -rf $dest && mkdir -p $dest/tmp"
adb push "$stage"/. "$dest/" > /dev/null

failed=0
for t in "${tests[@]}"; do
    out=$(adb shell "cd $dest && chmod 755 ./$t && TMPDIR=$dest/tmp HOME=$dest/tmp LD_LIBRARY_PATH=$dest LEMBED_PARITY_REF=$dest/parity_bge_small_en_v15.json ./$t; echo EXIT=\$?" || true)
    rc=$(echo "$out" | grep -oE 'EXIT=[0-9]+' | tail -n 1 | cut -d= -f2 || true)
    summary=$(echo "$out" | grep -oE '[0-9]+ passed[^[:cntrl:]]*' | tail -n 1 || true)
    if [ "${rc:-1}" = 0 ]; then
        echo "PASS  $t: $summary"
    else
        echo "FAIL  $t (exit ${rc:-?})"
        echo "$out" | tail -n 20 | sed 's/^/  /'
        failed=1
    fi
done
exit $failed
