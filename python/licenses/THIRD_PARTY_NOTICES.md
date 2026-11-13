# Third-party components in the libembedding-ng wheels

libembedding-ng itself is under the MIT licence (`LICENSE`). Each wheel also
ships native code from the projects below; their licence texts are in this
directory (`<name>-<version>.dist-info/licenses/licenses/` once installed).

| Component | Licence | Wheels | Licence file |
|---|---|---|---|
| ONNX Runtime | MIT, with its own third-party notices | all | `onnxruntime.LICENSE`, `onnxruntime.ThirdPartyNotices.txt` |
| llama.cpp / ggml | MIT | all | `llama.cpp.LICENSE` |
| cJSON | MIT | all | `cJSON.LICENSE` |
| stb_image, stb_image_resize2 | MIT or public domain | all | `stb.LICENSE` |
| curl (libcurl, static, HTTP only) | curl licence (MIT-style) | Linux, Android | `curl.COPYING` |
| Mbed TLS (static) | Apache-2.0 (dual-licensed with GPL-2.0-or-later; used under Apache-2.0) | Linux, Android | `mbedtls.LICENSE` |
| curl (libcurl.dll, from vcpkg) | curl licence (MIT-style) | Windows | `curl.COPYING` |
| zlib (z.dll, from vcpkg) | zlib licence | Windows | `zlib.LICENSE` |
| libgomp (GCC OpenMP runtime) | GPL-3.0-or-later WITH GCC-exception-3.1 | Linux | see below |

The macOS wheel uses the system libcurl and bundles no download library.

libgomp is grafted by auditwheel from the manylinux toolchain, unmodified. The
GCC Runtime Library Exception 3.1 allows it to be distributed with programs
under any licence: <https://www.gnu.org/licenses/gcc-exception-3.1.html>.

The model files libembedding downloads at run time are not part of the wheels;
each comes with its own licence on its HuggingFace page.
