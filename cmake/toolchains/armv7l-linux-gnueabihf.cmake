# libembedding - armv7l-linux-gnueabihf.cmake
# Cross toolchain for 32-bit ARM Linux (Raspberry Pi OS 32-bit, Debian armhf):
# ARMv7-A, NEON with VFPv4 (fused multiply-add), hard-float ABI. Every ARMv7
# Raspberry Pi (2 v1.2 and later) has them.
#
# Used to cross-compile ONNX Runtime (.github/workflows/ort-armv7l.yml), the
# static libcurl over mbedTLS (scripts/build_curl_mbedtls.sh) and libembedding
# itself (scripts/build_wheel_armv7l.sh). Ubuntu 22.04 package:
# g++-arm-linux-gnueabihf (GCC 11, glibc 2.35 sysroot in /usr/arm-linux-gnueabihf).
#
# Auteur: David Orel
# Version: 1.13.0
#
# SPDX-License-Identifier: MIT

set(CMAKE_SYSTEM_NAME Linux)
# ONNX Runtime and ggml both select their ARM32 code from this value.
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

set(CMAKE_C_FLAGS_INIT "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS_INIT "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard")

# Libraries and headers from the armhf sysroot (and from any prefix given in
# CMAKE_FIND_ROOT_PATH by the caller, such as the curl prefix); programs from
# the build machine.
list(APPEND CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
