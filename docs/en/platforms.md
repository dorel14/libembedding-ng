---
nav_exclude: true
---

# Supported platforms

> **Français :** [Plateformes](../platforms.html)

This page describes where libembedding runs, what CI tests, and what has never
been run on physical hardware. Names and numbers come from the code and the
workflows; anything not executed on real devices is stated as such.

## Matrix

| Platform | Deliverable | Tested in CI | Never run on physical hardware? |
|---|---|---|---|
| Windows x64 | PyPI wheel | yes (`windows-2022`) | n/a |
| Linux x86_64 | PyPI wheel | yes (`ubuntu-22.04`) | n/a |
| Linux aarch64 | PyPI wheel `manylinux_2_31_aarch64` | yes (ARM runner `ubuntu-22.04-arm`) | yes (Raspberry Pi) |
| macOS arm64 (Apple Silicon, macOS 14+) | PyPI wheel | yes (`macos-15`) | n/a |
| Android arm64-v8a | wheel `android_24_arm64_v8a` (CPython 3.13) | built; native tests on an x86_64 emulator only | yes (arm64 device) |
| Android x86_64 | wheel `android_24_x86_64` (CPython 3.13) | yes (emulator) | n/a |
| iOS 15.1+ (arm64, device and simulator) | native library `libembedding.xcframework` (no Python wheel) | yes (simulator) | yes (iPhone) |

Sources: `.github/workflows/ci.yml` (`python-tests` matrix),
`.github/workflows/tests.yml`, `.github/workflows/build-wheel.yml`,
`.github/workflows/build-wheel-android.yml`, `.github/workflows/android.yml`,
`.github/workflows/ios.yml`.

On iOS, ggml is built without Metal (`-DGGML_METAL=OFF` in
`scripts/build_ios.sh`): GPU is out of scope.

## Linux aarch64 and Raspberry Pi

The `manylinux_2_31_aarch64` wheel installs on **64-bit** ARM Linux with glibc
2.31 or later: Raspberry Pi 3/4/5 under 64-bit Raspberry Pi OS (bullseye,
bookworm), Ubuntu 20.04+, AWS Graviton, Ampere. The `manylinux_2_31` floor is
enforced by `scripts/build_wheel_manylinux.sh` ("above the manylinux_2_31
floor").

- ggml ships eight ARM CPU libraries (`ggml-cpu-armv8.0_1` …
  `ggml-cpu-armv9.2_2`) and loads the best one at run time;
  `libembedding.ggml_cpu_features()` names it.
- **32-bit Raspberry Pi OS (armv7l) is not supported.**
- CI builds the wheel in a `manylinux_2_28` container and tests it on GitHub's
  ARM runners. It has **not** been tested on a physical Raspberry Pi.
- HTTPS downloads use the system CA bundle
  (`/etc/ssl/certs/ca-certificates.crt`, `/etc/pki/tls/certs/ca-bundle.crt`,
  `/etc/ssl/ca-bundle.pem`, `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`,
  `/etc/ssl/cert.pem`, first one found). `SSL_CERT_FILE` or `CURL_CA_BUNDLE`
  override it (`include/libembedding/detail/downloader_impl.hpp`,
  `find_ca_source()`).

## Android

Wheels `android_24_arm64_v8a` and `android_24_x86_64` for **CPython 3.13**,
Android 7.0 (API 24) and later: the PEP 738 builds used by BeeWare and
Chaquopy. PyPI has no cffi or numpy for Android; they come from Chaquopy's
index:

```bash
pip install --extra-index-url https://chaquo.com/pypi-13.1 libembedding-ng
```

(`PIP_EXTRA_INDEX_URL=https://chaquo.com/pypi-13.1` in
`.github/workflows/build-wheel-android.yml`.)

**Build.** `scripts/build_android_libs.sh` (cibuildwheel before-build step)
builds, per ABI, libembedding, ggml/llama and a **static** libcurl over
**mbedTLS** (`scripts/build_android_curl.sh` → `scripts/build_curl_mbedtls.sh`);
ONNX Runtime comes from the official `onnxruntime-android` AAR 1.31.0
(`ONNXRUNTIME_VERSION`). The API level is `ANDROID_API_LEVEL` (24), which also
sets the `android_24_*` tag.

**Model downloads.** Android has no libcurl for native code, so downloads go
through the static libcurl above. Certificates come from the system store,
`/system/etc/security/cacerts` (`CURLOPT_CAPATH`), unless `SSL_CERT_FILE` or
`CURL_CA_BUNDLE` names an existing file. Without `HOME`, the cache is
`$TMPDIR/libembedding`, else `/data/local/tmp/libembedding` (`get_cache_dir()`).

**Tests.**

- `android.yml` builds the native libraries and tests for `arm64-v8a` and
  `x86_64`, then runs the unit tests on an **x86_64 emulator** (API 30) through
  `scripts/android_run_tests.sh`, including `test_text_embedding` (a real model
  download) and `test_embedding_parity`.
- `build-wheel-android.yml` tests the x86_64 wheel on cibuildwheel's emulator
  (`pytest -m "not network or download_smoke"`). The arm64 wheel is **built but
  not run**: an x86_64 runner has no fast arm64 emulator.
- Nothing has been run on a physical arm64 Android device.

## iOS

iOS (15.1 and later, arm64) ships as a **native library**, not as a Python
wheel.

### Building

```bash
bash scripts/build_ios.sh <iphoneos|iphonesimulator> <work-dir> [tests]
```

- `iphoneos`: device (arm64). `iphonesimulator`: simulator, arm64 only
  (`CMAKE_OSX_ARCHITECTURES=arm64`).
- `tests` (optional): also builds the unit tests (`test_similarity`,
  `test_pooling`, `test_tokenizer`, `test_model_registry`, `test_struct_sizes`,
  `test_sparse_gguf`, `test_embedding_parity`), to run on a simulator with
  `xcrun simctl spawn`.
- Deployment target: `IOS_DEPLOYMENT_TARGET`, **15.1** by default (the minimum
  of the ONNX Runtime pod).
- ONNX Runtime: the `onnxruntime-c` pod archive **1.30.0**, the newest
  published for iOS (1.31.0 is not), checked against a pinned SHA-256. The 1.30
  headers are what the code compiles against.
- The result is a self-contained `libembedding.dylib`: ONNX Runtime (static),
  ggml/llama (static, `BUILD_SHARED_LIBS=OFF`) and libcurl/mbedTLS (static) are
  linked in; only system libraries and frameworks remain as dependencies
  (checked with `otool -L` at the end of the script). ggml's CPU dispatch
  (`LIBEMBEDDING_CPU_DISPATCH=OFF`) and Metal are off. The install name is
  `@rpath/libembedding.dylib`.

### xcframework

The `.github/workflows/ios.yml` workflow (on pull requests to `dev`/`main` or
manual dispatch, `macos-15` runner) builds both slices, runs the tests on an
iPhone simulator, then assembles `libembedding.xcframework` (both dylibs plus
the public headers from `include/libembedding`) and uploads it as the
**`libembedding-ios-xcframework`** artifact (`libembedding-ios.xcframework.zip`).
The xcframework includes the licence texts of everything linked in: `LICENSE`,
`LICENSE.onnxruntime`, `LICENSE.llama.cpp`, `LICENSE.curl`, `LICENSE.mbedtls`,
and `NOTICE.ca-certificates` (Mozilla CA bundle, MPL 2.0).

### Using it in an Xcode project

Add `libembedding.xcframework` to the frameworks and libraries of the app
target. The dylib's install name is `@rpath/libembedding.dylib`
(`scripts/build_ios.sh`), so the app must embed it somewhere its rpaths cover.
The headers are the C API ones (`libembedding/libembedding.h`). This procedure
has not been validated in a real Xcode project, and the repository does not yet
contain a Swift example or an Xcode project.

### Cache and certificates

- Model cache: `$HOME/Library/Caches/libembedding` (the system may purge it; a
  purged model is downloaded again). `LIBEMBEDDING_CACHE_DIR` and
  `FASTEMBED_CACHE_DIR` take precedence.
- iOS exposes no certificate file, and the native verifier (SecTrust) is not
  available with mbedTLS. Mozilla's CA bundle (`cacert-2026-11-05.pem` from
  curl.se/ca, pinned by SHA-256 in `scripts/build_ios.sh`) is therefore
  **compiled into the dylib**: `-DLIBEMBEDDING_CA_BUNDLE_EMBED=<pem>`
  (`CMakeLists.txt`) defines `LIBEMBEDDING_EMBEDDED_CA_BUNDLE`, and
  `downloader_impl.hpp` hands it to libcurl through `CURLOPT_CAINFO_BLOB`.
- `SSL_CERT_FILE` or `CURL_CA_BUNDLE`, when they name an existing file,
  override that bundle.
- The embedded bundle ages: rebuild to change its date.

### Tests

CI runs the unit tests on a simulator, including `test_embedding_parity`,
which downloads `bge-small`: that validates the static libcurl, mbedTLS and the
embedded CA bundle end to end. Nothing has run on a physical iPhone.

## Model size limit on mobile

On Android and iOS the phone's memory is the constraint: a registry model whose
files exceed **150 MB** is refused **by name, before any download**, with an
error listing the light models of the same kind.

| Item | Detail |
|---|---|
| C constant | `LEMBED_MOBILE_MODEL_LIMIT_MB` (150), `include/libembedding/model_registry.h` |
| Active limit | `int lembed_mobile_model_limit_mb(void)`: 150 on Android and iOS, 0 (no limit) elsewhere |
| Model size | `int lembed_model_download_size_mb(model_code, model_file)`: model file plus `.onnx_data` external weights, in MB; `-1` if the pair is not in the registry |
| Python | `libembedding.mobile_model_limit_mb()`; `ModelInfo.download_size_mb` (`-1` if unknown) |
| Error | `LEMBED_ERROR_UNSUPPORTED` in C, `UnsupportedError` in Python |

Behaviour:

- The check applies to `lembed_ensure_text_model`, `lembed_ensure_sparse_model`,
  `lembed_ensure_image_model`, `lembed_ensure_reranker_model` and the
  constructors that use them
  (`include/libembedding/detail/download_manager.hpp`,
  `check_mobile_model_limit`), and to `lembed_ensure_gguf_model` for registry
  GGUF models (`include/libembedding/downloader.h`). A repo/file pair outside
  the registry has no known size and is not checked.
- A **local path** is never checked.
- `list_text_models()`, `list_sparse_models()`, `list_image_models()` and
  `list_reranker_models()` return only models of 150 MB or less when the limit
  is active (`python/src/libembedding/models.py`, `_list_models`).
- `LIBEMBEDDING_ALLOW_LARGE_MODELS=1` lifts the limit (it wins over the next
  one).
- `LIBEMBEDDING_FORCE_MOBILE_MODEL_LIMIT=1` applies the limit on a desktop, to
  test an application's behaviour. For both variables only the value `1`
  counts.

Light models, from the size table in `model_registry.h`
(`lembed__model_sizes`): text, among others `all-MiniLM-L6-v2` (90 MB, 23 MB
quantized), `all-MiniLM-L12-v2` (133 MB, 34 MB quantized), `bge-small-en-v1.5`
(133 MB, quantized variants 34 and 66 MB), `snowflake-arctic-embed-xs` and `-s`,
`bge-small-zh-v1.5` (95 MB) and the quantized variants of several base models
(`nomic-embed-text-v1.5` 137 MB, `gte-base-en-v1.5` 147 MB,
`snowflake-arctic-embed-m` 110 MB); image: ResNet-50 (94 MB) and CLIP ViT-B-32
INT8 (89 MB); reranker: Jina Reranker V1 Turbo quantized (38 MB). **No sparse
model is light** (SPLADE++: 532 MB). For the exact, current list, call
`list_*_models()` on the target platform or read `ModelInfo.download_size_mb`.

```python
import libembedding

print(libembedding.mobile_model_limit_mb())  # 150 on Android/iOS, 0 elsewhere
for m in libembedding.list_text_models():
    print(m.model_name, m.download_size_mb)
```

## Numerical parity across platforms

A reference computed on x86_64 (Windows, ONNX Runtime 1.31.0, fp32 weights)
serves as the check: `python/tests/data/parity_bge_small_en_v15.json` holds the
`bge-small-en-v1.5` embeddings (dimension 384) of three texts, with
`"min_cosine": 0.999`. Every platform must give the same dimension and a cosine
**of at least 0.999** per text against that reference: the compiler, SIMD
kernels, ONNX Runtime build or tokenizer may move the last digits, never the
vector. The threshold is read from the JSON.

Two tests read that file:

- `tests/test_embedding_parity.cpp` (C++ integration test: it downloads the
  model; reference path: `argv[1]`, else `LEMBED_PARITY_REF`, else the compiled-in
  path); run in CI on the Android emulator (`scripts/android_run_tests.sh`) and
  the iOS simulator (`ios.yml`);
- `python/tests/test_parity.py` (pytest, markers `network` and
  `download_smoke`); run in the wheel test job on Linux x64, Linux aarch64,
  macOS and Windows (`ci.yml` → `build-wheel.yml`), and on the Android
  emulator through `download_smoke`.

## See also

- [Getting started](getting_started.html) — installation
- [C API — Model Downloading](c_api/downloader.html) — certificates, mobile limit
- [Available models](models.html) — registry
